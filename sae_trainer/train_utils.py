import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

def evaluate(model, loader, device, lambda_l1=1e-4, mass_frac_threshold=0.01):
    model.eval()
    total_loss, total_recon, total_l1, total_active, n = 0.0, 0.0, 0.0, 0.0, 0

    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device, non_blocking=True)
            x_hat, z = model(xb)

            recon = F.mse_loss(x_hat, xb)
            l1 = z.abs().mean()
            loss = recon + lambda_l1 * l1

            az = z.abs()
            row_sum = az.sum(dim=1, keepdim=True).clamp(min=1e-8)
            frac = az / row_sum
            active = (frac > mass_frac_threshold).float().sum(dim=1).mean()
            #active = (z > 1e-6).float().sum(dim=1).mean()  # avg active features per token

            bs = xb.size(0)
            total_loss += loss.item() * bs
            total_recon += recon.item() * bs
            total_l1 += l1.item() * bs
            total_active += active.item() * bs
            n += bs

    return {
        "loss": total_loss / n,
        "recon": total_recon / n,
        "l1": total_l1 / n,
        "active": total_active / n,
    }

def train_sae(sae, train_loader, val_loader, opt, scheduler, device, lambda_l1=1e-4, show_curves=True, mass_frac_threshold=0.01):

    epochs = 20
    history = {"train_loss": [], "train_recon": [], "train_l1": [],
               "val_loss": [], "val_recon": [], "val_l1": [], "val_active": []}

    for epoch in range(1, epochs + 1):
        sae.train()
        running_loss, running_recon, running_l1, seen = 0.0, 0.0, 0.0, 0

        for (xb,) in train_loader:
            xb = xb.to(device, non_blocking=True)

            x_hat, z = sae(xb)
            recon = F.mse_loss(x_hat, xb)
            l1 = z.abs().mean()
            loss = recon + lambda_l1 * l1

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
            opt.step()

            bs = xb.size(0)
            running_loss += loss.item() * bs
            running_recon += recon.item() * bs
            running_l1 += l1.item() * bs
            seen += bs

        scheduler.step()

        train_metrics = {
            "loss": running_loss / seen,
            "recon": running_recon / seen,
            "l1": running_l1 / seen,
        }
        val_metrics = evaluate(sae, val_loader, device, lambda_l1=lambda_l1, mass_frac_threshold=mass_frac_threshold)

        history["train_loss"].append(train_metrics["loss"])
        history["train_recon"].append(train_metrics["recon"])
        history["train_l1"].append(train_metrics["l1"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_recon"].append(val_metrics["recon"])
        history["val_l1"].append(val_metrics["l1"])
        history["val_active"].append(val_metrics["active"])

        print(
            f"Epoch {epoch:02d} | "
            f"train loss {train_metrics['loss']:.6f} (recon {train_metrics['recon']:.6f}, l1 {train_metrics['l1']:.6f}) | "
            f"val loss {val_metrics['loss']:.6f} (recon {val_metrics['recon']:.6f}, l1 {val_metrics['l1']:.6f}) | "
            f"val active {val_metrics['active']:.1f}"
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