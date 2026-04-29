from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from .model_utils import ReluSparseAutoencoder, TopKSparseAutoencoder
from .train_utils import resample_dead_features, firing_rate_kl_loss

class SAETrainingModule(ABC):
    def __init__(self, resample_dead_neurons: bool):
        super().__init__()
        self.resample_dead_neurons = resample_dead_neurons

    @abstractmethod
    def _train_step(self, xb: torch.Tensor, cfg, effective_lambda_l1: float):
        # Returns (loss_tensor, z_tensor, metrics_dict)
        # metrics_dict keys: loss, recon, l1, kl  (all floats, not tensors)
        ...

    @abstractmethod
    def _eval_step(self, xb: torch.Tensor, cfg):
        # Returns (x_hat_tensor, z_tensor, metrics_dict)
        # metrics_dict keys: loss, recon, l1, kl  (all floats, not tensors)
        ...

    def train_sae(self, train_loader, val_loader, opt, scheduler, device, cfg, run=None, show_curves=False):
        history = {k: [] for k in [
            "train_loss", "train_recon", "train_l1", "train_kl",
            "val_loss", "val_recon", "val_l1", "val_kl",
            "val_active", "val_nonzero", "val_fve", "val_dead_frac",
        ]}

        d_latent = self.sae.encoder.weight.shape[0]
        fired_this_interval = torch.zeros(d_latent, dtype=torch.bool, device=device)
        resample_interval = getattr(cfg, "resample_interval_epochs", 0)

        for epoch in range(1, cfg.num_epochs + 1):
            self.sae.train()
            running = {"loss": 0.0, "recon": 0.0, "l1": 0.0, "kl": 0.0}
            seen = 0

            lambda_l1 = getattr(cfg, "lambda_l1", 0.0)
            lambda_l1_warmup = getattr(cfg, "lambda_l1_warmup_epochs", 0)
            effective_lambda_l1 = (
                lambda_l1 * min(1.0, epoch / lambda_l1_warmup)
                if lambda_l1_warmup > 0
                else lambda_l1
            )

            for (xb,) in train_loader:
                xb = xb.to(device, non_blocking=True)
                loss, z, batch_metrics = self._train_step(xb, cfg, effective_lambda_l1)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.sae.parameters(), 1.0)
                opt.step()

                if self.resample_dead_neurons and resample_interval > 0:
                    fired_this_interval |= (z.detach() > 0).any(dim=0)

                bs = xb.size(0)
                for k in running:
                    running[k] += batch_metrics[k] * bs
                seen += bs

            scheduler.step()

            if self.resample_dead_neurons and resample_interval > 0 and epoch % resample_interval == 0:
                dead_mask = ~fired_this_interval
                n_resampled = resample_dead_features(self.sae, train_loader, opt, device, dead_mask)
                print(f"  [resample] epoch {epoch}: resampled {n_resampled} dead features")
                if run:
                    run.log({"resample/n_resampled": n_resampled, "epoch": epoch})
                fired_this_interval.zero_()

            train_metrics = {k: v / seen for k, v in running.items()}
            val_metrics = self.evaluate_sae(val_loader, device, cfg)

            for k in ["loss", "recon", "l1", "kl"]:
                history[f"train_{k}"].append(train_metrics[k])
            for k in ["loss", "recon", "l1", "kl", "active", "nonzero", "fve", "dead_frac"]:
                history[f"val_{k}"].append(val_metrics[k])

            if run:
                run.log({
                    "train/loss": train_metrics["loss"],
                    "train/recon": train_metrics["recon"],
                    "train/l1": train_metrics["l1"],
                    "train/kl": train_metrics["kl"],
                    "val/loss": val_metrics["loss"],
                    "val/recon": val_metrics["recon"],
                    "val/l1": val_metrics["l1"],
                    "val/kl": val_metrics["kl"],
                    "val/active_features": val_metrics["active"],
                    "val/nonzero_features": val_metrics["nonzero"],
                    "val/fve": val_metrics["fve"],
                    "val/dead_frac": val_metrics["dead_frac"],
                    "train/effective_lambda_l1": effective_lambda_l1,
                    "epoch": epoch,
                })

            kl_str = (
                f", kl {train_metrics['kl']:.6f} / {val_metrics['kl']:.6f}"
                if train_metrics["kl"] > 0
                else ""
            )
            print(
                f"Epoch {epoch:02d} | "
                f"train loss {train_metrics['loss']:.6f} "
                f"(recon {train_metrics['recon']:.6f}, l1 {train_metrics['l1']:.6f}{kl_str}) | "
                f"val loss {val_metrics['loss']:.6f} "
                f"(recon {val_metrics['recon']:.6f}, l1 {val_metrics['l1']:.6f}) | "
                f"fve {val_metrics['fve']:.3f} | dead {val_metrics['dead_frac']:.2%}"
            )

        if show_curves:
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.plot(history["train_recon"], label="train")
            plt.plot(history["val_recon"], label="val")
            plt.title("Reconstruction MSE")
            plt.legend()
            plt.subplot(1, 3, 2)
            plt.plot(history["train_l1"], label="train")
            plt.plot(history["val_l1"], label="val")
            plt.title("Latent L1")
            plt.legend()
            plt.subplot(1, 3, 3)
            plt.plot(history["val_active"])
            plt.title("Val active features/token")
            plt.tight_layout()
            plt.show()

        return self.sae, history

    def evaluate_sae(self, loader, device, cfg) -> dict:
        self.sae.eval()
        totals = {"loss": 0.0, "recon": 0.0, "l1": 0.0, "kl": 0.0, "active": 0.0, "nonzero": 0.0}
        x_sum = x_sq_sum = None
        total_ss_res = 0.0
        ever_active = None
        n = 0

        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(device, non_blocking=True)
                x_hat, z, step_metrics = self._eval_step(xb, cfg)

                az = z.abs()
                frac = az / az.sum(dim=1, keepdim=True).clamp(min=1e-8)
                active = (frac > cfg.mass_frac_threshold).float().sum(dim=1).mean()
                nonzero = (z > 0).float().sum(dim=-1).mean()

                total_ss_res += ((x_hat - xb) ** 2).sum().item()
                x_sum = xb.sum(dim=0) if x_sum is None else x_sum + xb.sum(dim=0)
                x_sq_sum = (xb ** 2).sum(dim=0) if x_sq_sum is None else x_sq_sum + (xb ** 2).sum(dim=0)

                fired = (z > 0).any(dim=0)
                ever_active = fired if ever_active is None else (ever_active | fired)

                bs = xb.size(0)
                for k in ["loss", "recon", "l1", "kl"]:
                    totals[k] += step_metrics[k] * bs
                totals["active"] += active.item() * bs
                totals["nonzero"] += nonzero.item() * bs
                n += bs

        mean_x = x_sum / n
        ss_tot = (x_sq_sum - n * mean_x ** 2).sum().item()
        return {
            **{k: v / n for k, v in totals.items()},
            "fve": 1.0 - total_ss_res / max(ss_tot, 1e-8),
            "dead_frac": (~ever_active).float().mean().item(),
        }


class ReluSAETrainingModule(SAETrainingModule):
    """
    ReLU-SAE. Supports L1, KL-divergence sparsity losses, and optional L1 warmup.
    resample_dead_neurons defaults to False since resampling is less impactful for ReLU-SAEs.
    """
    def __init__(self, d_in: int, d_latent: int, device, normalize_decoder: bool = True, resample_dead_neurons: bool = False):
        super().__init__(resample_dead_neurons=resample_dead_neurons)
        self.sae = ReluSparseAutoencoder(d_in=d_in, d_latent=d_latent, normalize_decoder=normalize_decoder).to(device)

    def _train_step(self, xb, cfg, effective_lambda_l1):
        if cfg.lambda_kl > 0:
            x_hat, z, h = self.sae(xb, return_pre_relu=True)
            kl = firing_rate_kl_loss(h, cfg.target_firing_rate)
        else:
            x_hat, z = self.sae(xb)
            kl = torch.zeros((), device=xb.device)
        recon = F.mse_loss(x_hat, xb)
        l1 = z.abs().mean()
        loss = recon + effective_lambda_l1 * l1 + cfg.lambda_kl * kl
        return loss, z, {"loss": loss.item(), "recon": recon.item(), "l1": l1.item(), "kl": kl.item()}

    def _eval_step(self, xb, cfg):
        if cfg.lambda_kl > 0:
            x_hat, z, h = self.sae(xb, return_pre_relu=True)
            kl = firing_rate_kl_loss(h, cfg.target_firing_rate)
        else:
            x_hat, z = self.sae(xb)
            kl = torch.zeros((), device=xb.device)
        recon = F.mse_loss(x_hat, xb)
        l1 = z.abs().mean()
        loss = recon + cfg.lambda_l1 * l1 + cfg.lambda_kl * kl
        return x_hat, z, {"loss": loss.item(), "recon": recon.item(), "l1": l1.item(), "kl": kl.item()}


class TopKSAETrainingModule(SAETrainingModule):
    """
    TopK-SAE. Sparsity is enforced structurally (hard top-k); no L1/KL losses.
    resample_dead_neurons defaults to True since dead features are a common problem with TopK-SAEs.
    """
    def __init__(self, d_in: int, d_latent: int, k: int, device, normalize_decoder: bool = True, resample_dead_neurons: bool = True):
        super().__init__(resample_dead_neurons=resample_dead_neurons)
        self.sae = TopKSparseAutoencoder(d_in=d_in, d_latent=d_latent, k=k, normalize_decoder=normalize_decoder).to(device)

    def _train_step(self, xb, _cfg, _effective_lambda_l1):
        x_hat, z = self.sae(xb)
        loss = F.mse_loss(x_hat, xb)
        return loss, z, {"loss": loss.item(), "recon": loss.item(), "l1": 0.0, "kl": 0.0}

    def _eval_step(self, xb, _cfg):
        x_hat, z = self.sae(xb)
        loss = F.mse_loss(x_hat, xb)
        return x_hat, z, {"loss": loss.item(), "recon": loss.item(), "l1": 0.0, "kl": 0.0}
