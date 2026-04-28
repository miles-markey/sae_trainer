import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional
import torch.nn as nn
import torch.nn.functional as F


def firing_rate_kl_loss(
    pre_relu: torch.Tensor,
    target_rate: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    KL( Bernoulli(p_hat) || Bernoulli(r) ) averaged over latents.

    p_hat_j = mean_batch sigmoid(pre_relu[:, j]) — soft proxy for "how often
    this latent is on"; r = target_rate (e.g. 0.01–0.05).

    Use pre-ReLU encoder logits so gradients flow through dead ReLU units.
    """
    r = float(target_rate)
    r = max(r, eps)
    r = min(r, 1.0 - eps)
    p_hat = torch.sigmoid(pre_relu).mean(dim=0).clamp(eps, 1.0 - eps)
    kl = p_hat * torch.log(p_hat / r) + (1.0 - p_hat) * torch.log((1.0 - p_hat) / (1.0 - r))
    return kl.mean()


# ---- Simple ReLU SAE ----
class SparseAutoencoder(nn.Module):
    def __init__(self, d_in: int, d_latent: int, normalize_decoder: bool = True):
        super().__init__()
        self.encoder = nn.Linear(d_in, d_latent, bias=True)
        self.decoder = nn.Linear(d_latent, d_in, bias=False)
        self.normalize_decoder = normalize_decoder

        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)

    def forward(self, x, return_pre_relu: bool = False):
        h = self.encoder(x)
        z = F.relu(h)  # [B, d_latent], sparse nonnegative code
        if self.normalize_decoder:
            # decoder.weight: [d_in, d_latent] — one column per dictionary atom
            W = self.decoder.weight
            scale = W.norm(dim=0, keepdim=True).clamp(min=1e-8)
            W = W / scale
            x_hat = F.linear(z, W, self.decoder.bias)
        else:
            x_hat = self.decoder(z)
        if return_pre_relu:
            return x_hat, z, h
        return x_hat, z


# ---- Top-K SAE ----
class TopKSparseAutoencoder(nn.Module):
    def __init__(self, d_in: int, d_latent: int, k: int, normalize_decoder: bool = True):
        super().__init__()
        self.k = k
        self.normalize_decoder = normalize_decoder
        self.encoder = nn.Linear(d_in, d_latent, bias=True)
        self.decoder = nn.Linear(d_latent, d_in, bias=False)

        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)

    def forward(self, x, return_pre_relu: bool = False):
        h = self.encoder(x)
        # Keep only the top-k activations per token, zero the rest
        topk_vals, topk_idx = torch.topk(h, self.k, dim=-1)
        topk_vals = F.relu(topk_vals)  # ensure non-negative
        z = torch.zeros_like(h).scatter_(-1, topk_idx, topk_vals)

        if self.normalize_decoder:
            W = self.decoder.weight
            scale = W.norm(dim=0, keepdim=True).clamp(min=1e-8)
            W = W / scale
            x_hat = F.linear(z, W, self.decoder.bias)
        else:
            x_hat = self.decoder(z)

        return x_hat, z