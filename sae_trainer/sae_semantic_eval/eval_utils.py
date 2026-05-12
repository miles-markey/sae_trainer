import math
import numpy as np
import torch
import matplotlib.pyplot as plt

# Optional (for latent-space scatter)
try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

# -------------------------
# 1) Adapt this wrapper to your SAE API
# -------------------------
def sae_forward(sae, x):
    """
    Returns:
      x_hat: [B, D] reconstruction
      z:     [B, K] latent activations (post-activation sparse codes)
    """
    out = sae(x)

    # common patterns
    if isinstance(out, tuple) and len(out) >= 2:
        x_hat, z = out[0], out[1]
    elif isinstance(out, dict):
        x_hat = out.get("x_hat", out.get("recon", out.get("reconstruction")))
        z = out.get("z", out.get("latents", out.get("codes")))
        if x_hat is None or z is None:
            raise ValueError("Dict output missing reconstruction/latent keys.")
    else:
        raise ValueError("Unknown SAE output format. Update sae_forward().")

    return x_hat, z

# -------------------------
# 2) Core metrics
# -------------------------
@torch.no_grad()
def evaluate_sae(sae, dataloader, device, mass_frac_threshold=0.01, show_metrics=False):
    sae.eval()
    n_tokens = 0

    mse_sum = 0.0
    var_sum = 0.0
    l1_sum = 0.0
    active_sum = 0.0
    dead_counter = None
    usage_counter = None

    # We'll collect small sample for plots
    z_samples = []

    for (xb,) in dataloader:
        n_tokens += xb.size(0)
        xb = xb.to(device)
        x_hat, z = sae_forward(sae, xb)

        # Reconstruction
        err = (xb - x_hat)
        mse_sum += (err.pow(2).mean(dim=1)).sum().item()
        var_sum += xb.var(dim=1, unbiased=False).sum().item()

        # Sparsity
        l1_sum += z.abs().mean(dim=1).sum().item()
        az = z.abs()
        row_sum = az.sum(dim=1, keepdim=True).clamp(min=1e-8)
        frac = az / row_sum
        active = (frac > mass_frac_threshold).float()  # pass mass_frac_threshold into evaluate_sae
        active_sum += active.sum(dim=1).sum().item()

        # Feature usage/dead features
        fired = (active.sum(dim=0) > 0).float()   # per feature in this batch
        if dead_counter is None:
            dead_counter = torch.zeros_like(fired)
            usage_counter = torch.zeros_like(fired)

        dead_counter += (1.0 - fired)             # count batches where feature never fired
        usage_counter += active.sum(dim=0).detach().cpu()

        # keep sample for visualizations
        if len(z_samples) < 10:
            z_samples.append(z.detach().float().cpu())

    mse = mse_sum / n_tokens
    mean_var = var_sum / n_tokens
    explained_var = max(0.0, 1.0 - (mse / (mean_var + 1e-12)))

    avg_l1 = l1_sum / n_tokens
    avg_active_per_token = active_sum / n_tokens

    total_feature_activations = usage_counter.sum().item() + 1e-12
    p = (usage_counter / total_feature_activations).numpy()
    p_nonzero = p[p > 0]
    usage_entropy = float(-(p_nonzero * np.log(p_nonzero)).sum())
    usage_perplexity = float(np.exp(usage_entropy))  # "effective number of used features"

    dead_feature_rate = float((usage_counter == 0).float().mean().item())

    z_sample = torch.cat(z_samples, dim=0) if len(z_samples) else None

    metrics = {
        "mse": mse,
        "explained_variance": explained_var,
        "avg_l1_latent": avg_l1,
        "avg_active_features_per_token": avg_active_per_token,
        "dead_feature_rate": dead_feature_rate,
        "usage_entropy": usage_entropy,
        "usage_perplexity": usage_perplexity,
        "usage_counter": usage_counter.numpy(),
        "z_sample": z_sample.numpy() if z_sample is not None else None,
    }

    if show_metrics:
        print("=== SAE Metrics ===")
        for k in [
            "mse",
            "explained_variance",
            "avg_l1_latent",
            "avg_active_features_per_token",
            "dead_feature_rate",
            "usage_entropy",
            "usage_perplexity",
        ]:
            print(f"{k:>32}: {metrics[k]:.6f}")

    return metrics

# -------------------------
# 3) Visualizations
# -------------------------
def visualize_sae(metrics):
    usage = metrics["usage_counter"]
    z_sample = metrics["z_sample"]

    # Feature usage histogram
    plt.figure(figsize=(7,4))
    plt.hist(usage, bins=50)
    plt.title("Feature Usage Counts")
    plt.xlabel("Activation count across sampled tokens")
    plt.ylabel("Number of features")
    plt.show()

    # Top most-used features
    top_k = 30 # try 100
    top_idx = np.argsort(-usage)[:top_k]
    plt.figure(figsize=(10,4))
    plt.bar(np.arange(top_k), usage[top_idx])
    plt.title(f"Top {top_k} Most Used Features")
    plt.xlabel("Ranked feature")
    plt.ylabel("Usage count")
    plt.show()

    if z_sample is not None:
        # Latent activation value distribution
        plt.figure(figsize=(7,4))
        plt.hist(z_sample.flatten(), bins=100)
        plt.title("Latent Activation Value Distribution")
        plt.xlabel("z value")
        plt.ylabel("Count")
        plt.yscale("log")
        plt.show()

        # Sequence-position heatmap-like view (first 200 tokens, first 64 features)
        n_tokens = min(200, z_sample.shape[0])
        n_feats = min(64, z_sample.shape[1])
        plt.figure(figsize=(12,4))
        plt.imshow(z_sample[:n_tokens, :n_feats].T, aspect="auto", interpolation="nearest")
        plt.title("Latent Activations Heatmap (features x tokens)")
        plt.xlabel("Token index")
        plt.ylabel("Feature index")
        plt.colorbar()
        plt.show()

        # Optional UMAP projection of token latents
        if HAS_UMAP and z_sample.shape[0] > 200:
            z_for_umap = z_sample[:5000]
            reducer = umap.UMAP(n_neighbors=30, min_dist=0.05, metric="cosine", random_state=42)
            z_2d = reducer.fit_transform(z_for_umap)

            plt.figure(figsize=(6,6))
            plt.scatter(z_2d[:,0], z_2d[:,1], s=2, alpha=0.5)
            plt.title("UMAP of SAE Latent Activations")
            plt.xlabel("UMAP-1")
            plt.ylabel("UMAP-2")
            plt.show()
        elif not HAS_UMAP:
            print("UMAP not installed. `pip install umap-learn` to enable latent-space plot.")