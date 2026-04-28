import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from .model_utils import firing_rate_kl_loss


def evaluate(
    model,
    loader,
    device,
    lambda_l1=1e-4,
    mass_frac_threshold=0.01,
    lambda_kl: float = 0.0,
    target_firing_rate: float = 0.01,
):
    model.eval()
    total_loss, total_recon, total_l1, total_kl, total_active, total_nonzero, n = (
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0,
    )
    # FVE accumulators — kept as tensors to avoid d_in-sized Python loops
    x_sum = None
    x_sq_sum = None
    total_ss_res = 0.0
    ever_active = None  # [d_latent] bool, tracks which features fired at least once

    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device, non_blocking=True)
            if lambda_kl > 0:
                x_hat, z, h = model(xb, return_pre_relu=True)
                kl = firing_rate_kl_loss(h, target_firing_rate)
            else:
                x_hat, z = model(xb)
                kl = torch.zeros((), device=device)

            recon = F.mse_loss(x_hat, xb)
            l1 = z.abs().mean()
            loss = recon + lambda_l1 * l1 + lambda_kl * kl

            az = z.abs()
            row_sum = az.sum(dim=1, keepdim=True).clamp(min=1e-8)
            frac = az / row_sum
            active = (frac > mass_frac_threshold).float().sum(dim=1).mean()
            nonzero = (z > 0).float().sum(dim=-1).mean()

            # FVE: accumulate per-dimension sums for SS_tot, and SS_res
            total_ss_res += ((x_hat - xb) ** 2).sum().item()
            if x_sum is None:
                x_sum = xb.sum(dim=0)
                x_sq_sum = (xb ** 2).sum(dim=0)
            else:
                x_sum += xb.sum(dim=0)
                x_sq_sum += (xb ** 2).sum(dim=0)

            # Dead features: union of active features across all batches
            fired = (z > 0).any(dim=0)
            ever_active = fired if ever_active is None else (ever_active | fired)

            bs = xb.size(0)
            total_loss += loss.item() * bs
            total_recon += recon.item() * bs
            total_l1 += l1.item() * bs
            total_kl += kl.item() * bs
            total_active += active.item() * bs
            total_nonzero += nonzero.item() * bs
            n += bs

    # FVE = 1 - SS_res / SS_tot, where SS_tot = sum(x^2) - n*mean(x)^2
    mean_x = x_sum / n
    ss_tot = (x_sq_sum - n * mean_x ** 2).sum().item()
    fve = 1.0 - total_ss_res / max(ss_tot, 1e-8)

    dead_frac = (~ever_active).float().mean().item()

    out = {
        "loss": total_loss / n,
        "recon": total_recon / n,
        "l1": total_l1 / n,
        "active": total_active / n,
        "nonzero": total_nonzero / n,
        "fve": fve,
        "dead_frac": dead_frac,
    }
    out["kl"] = total_kl / n if lambda_kl > 0 else 0.0
    return out

def resample_dead_features(sae, train_loader, opt, device, dead_mask, max_resample_frac=0.1):
    all_dead_indices = dead_mask.nonzero(as_tuple=True)[0]
    if len(all_dead_indices) == 0:
        return 0

    # Subsample dead indices to avoid disrupting too much of the decoder at once
    n_to_resample = max(1, int(len(all_dead_indices) * max_resample_frac))
    perm = torch.randperm(len(all_dead_indices), device=device)[:n_to_resample]
    dead_indices = all_dead_indices[perm]

    # Collect reconstruction errors over enough batches to sample from
    sae.eval()
    all_errors, all_inputs = [], []
    with torch.no_grad():
        for (xb,) in train_loader:
            xb = xb.to(device)
            x_hat, _ = sae(xb)
            all_errors.append(((xb - x_hat) ** 2).sum(dim=-1))
            all_inputs.append(xb)
            if sum(e.size(0) for e in all_errors) >= n_to_resample * 10:
                break

    all_errors = torch.cat(all_errors)
    all_inputs = torch.cat(all_inputs)

    # Sample tokens weighted by reconstruction error
    probs = (all_errors / all_errors.sum()).clamp(min=0)
    sampled = all_inputs[torch.multinomial(probs, n_to_resample, replacement=True)]
    new_dirs = sampled / sampled.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    # Scale new encoder weights well below alive features so they don't immediately
    # dominate Top-K selection — they grow toward useful directions via gradient updates.
    alive_mask = ~dead_mask
    alive_scale = sae.encoder.weight[alive_mask].norm(dim=-1).mean() if alive_mask.any() else torch.tensor(1.0)
    scale = alive_scale * 0.1

    with torch.no_grad():
        sae.encoder.weight[dead_indices] = new_dirs * scale
        sae.encoder.bias[dead_indices] = 0.0
        sae.decoder.weight[:, dead_indices] = new_dirs.T

        # Reset Adam moment estimates for reinitialised parameters
        for param, row_idx, col_idx in [
            (sae.encoder.weight, dead_indices, None),
            (sae.encoder.bias,   dead_indices, None),
            (sae.decoder.weight, None,         dead_indices),
        ]:
            if param not in opt.state:
                continue
            state = opt.state[param]
            for key in ("exp_avg", "exp_avg_sq"):
                if key not in state:
                    continue
                if col_idx is None:
                    state[key][row_idx] = 0.0
                else:
                    state[key][:, col_idx] = 0.0

    sae.train()
    return n_to_resample


def train_sae(
    sae,
    train_loader,
    val_loader,
    opt,
    scheduler,
    device,
    lambda_l1=1e-4,
    show_curves=True,
    mass_frac_threshold=0.01,
    lambda_kl: float = 0.0,
    target_firing_rate: float = 0.01,
    num_epochs=20,
    lambda_l1_warmup_epochs=0,
    resample_interval_epochs=0,
    run=None,
):

    history = {
        "train_loss": [],
        "train_recon": [],
        "train_l1": [],
        "train_kl": [],
        "val_loss": [],
        "val_recon": [],
        "val_l1": [],
        "val_kl": [],
        "val_active": [],
        "val_nonzero": [],
        "val_fve": [],
        "val_dead_frac": [],
    }

    d_latent = sae.encoder.weight.shape[0]
    fired_this_interval = torch.zeros(d_latent, dtype=torch.bool, device=device)

    for epoch in range(1, num_epochs + 1):
        sae.train()
        running_loss, running_recon, running_l1, running_kl, seen = 0.0, 0.0, 0.0, 0.0, 0

        if lambda_l1_warmup_epochs > 0:
            effective_lambda_l1 = lambda_l1 * min(1.0, epoch / lambda_l1_warmup_epochs)
        else:
            effective_lambda_l1 = lambda_l1

        for (xb,) in train_loader:
            xb = xb.to(device, non_blocking=True)

            if lambda_kl > 0:
                x_hat, z, h = sae(xb, return_pre_relu=True)
                kl = firing_rate_kl_loss(h, target_firing_rate)
            else:
                x_hat, z = sae(xb)
                kl = torch.zeros((), device=device)

            recon = F.mse_loss(x_hat, xb)
            l1 = z.abs().mean()
            loss = recon + effective_lambda_l1 * l1 + lambda_kl * kl

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
            opt.step()

            if resample_interval_epochs > 0:
                fired_this_interval |= (z.detach() > 0).any(dim=0)

            bs = xb.size(0)
            running_loss += loss.item() * bs
            running_recon += recon.item() * bs
            running_l1 += l1.item() * bs
            running_kl += kl.item() * bs
            seen += bs

        scheduler.step()

        if resample_interval_epochs > 0 and epoch % resample_interval_epochs == 0:
            dead_mask = ~fired_this_interval
            n_resampled = resample_dead_features(sae, train_loader, opt, device, dead_mask)
            print(f"  [resample] epoch {epoch}: resampled {n_resampled} dead features")
            if run:
                run.log({"resample/n_resampled": n_resampled, "epoch": epoch})
            fired_this_interval.zero_()

        train_metrics = {
            "loss": running_loss / seen,
            "recon": running_recon / seen,
            "l1": running_l1 / seen,
            "kl": running_kl / seen,
        }
        val_metrics = evaluate(
            sae,
            val_loader,
            device,
            lambda_l1=lambda_l1,
            mass_frac_threshold=mass_frac_threshold,
            lambda_kl=lambda_kl,
            target_firing_rate=target_firing_rate,
        )

        history["train_loss"].append(train_metrics["loss"])
        history["train_recon"].append(train_metrics["recon"])
        history["train_l1"].append(train_metrics["l1"])
        history["train_kl"].append(train_metrics["kl"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_recon"].append(val_metrics["recon"])
        history["val_l1"].append(val_metrics["l1"])
        history["val_kl"].append(val_metrics["kl"])
        history["val_active"].append(val_metrics["active"])
        history["val_nonzero"].append(val_metrics["nonzero"])
        history["val_fve"].append(val_metrics["fve"])
        history["val_dead_frac"].append(val_metrics["dead_frac"])

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
            if lambda_kl > 0
            else ""
        )
        print(
            f"Epoch {epoch:02d} | "
            f"train loss {train_metrics['loss']:.6f} (recon {train_metrics['recon']:.6f}, l1 {train_metrics['l1']:.6f}{kl_str}) | "
            f"val loss {val_metrics['loss']:.6f} (recon {val_metrics['recon']:.6f}, l1 {val_metrics['l1']:.6f}) | "
            f"fve {val_metrics['fve']:.3f} | dead {val_metrics['dead_frac']:.2%}"
        )

    # ---- Curves ----
    if show_curves:
        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1)
        plt.plot(history["train_recon"], label="train")
        plt.plot(history["val_recon"], label="val")
        plt.title("Reconstruction MSE")
        plt.legend()

        plt.subplot(1,3,2)
        plt.plot(history["train_l1"], label="train")
        plt.plot(history["val_l1"], label="val")
        plt.title("Latent L1")
        plt.legend()

        plt.subplot(1,3,3)
        plt.plot(history["val_active"])
        plt.title("Val active features/token")

        plt.tight_layout()
        plt.show()
    
    return sae, history