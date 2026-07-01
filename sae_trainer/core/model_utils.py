from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager

class SparseAutoencoder(nn.Module, ABC):
    def __init__(self, d_in: int, d_latent: int, normalize_decoder: bool = True):
        super().__init__()
        self.normalize_decoder = normalize_decoder
        self.encoder = nn.Linear(d_in, d_latent, bias=True)
        self.decoder = nn.Linear(d_latent, d_in, bias=False)
        self.d_latent = d_latent

        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        if self.normalize_decoder:
            W = self.decoder.weight
            scale = W.norm(dim=0, keepdim=True).clamp(min=1e-8)
            W = W / scale
            return F.linear(z, W, self.decoder.bias)
        return self.decoder(z)
    
    def get_input_dim(self):
        return self.encoder.in_features

    def get_latent_dim(self):
        return self.d_latent

    @abstractmethod
    def forward(self, x: torch.Tensor):
        ...


# ---- ReLU SAE ----
class ReluSparseAutoencoder(SparseAutoencoder):
    def forward(self, x: torch.Tensor, return_pre_relu: bool = False):
        h = self.encoder(x)
        z = F.relu(h)
        x_hat = self._decode(z)
        if return_pre_relu:
            return x_hat, z, h
        return x_hat, z


# ---- Top-K SAE ----
class TopKSparseAutoencoder(SparseAutoencoder):
    def __init__(self, d_in: int, d_latent: int, k: int, normalize_decoder: bool = True):
        super().__init__(d_in, d_latent, normalize_decoder)
        self.k = k

    def forward(self, x: torch.Tensor):
        h = self.encoder(x)
        topk_vals, topk_idx = torch.topk(h, self.k, dim=-1)
        topk_vals = F.relu(topk_vals)
        z = torch.zeros_like(h).scatter_(-1, topk_idx, topk_vals)
        return self._decode(z), z

@contextmanager
def sae_inserted_llm(llm, sae, layer_idx, device):
    """
    Context manager that temporarily inserts the SAE into the LLM at layer_idx.
    Within the block, hidden states at that layer are replaced with SAE reconstructions.
    """
    def _hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        flat = h.reshape(-1, h.shape[-1])
        with torch.no_grad():
            x_hat = sae(flat)[0]          # [0] = reconstruction
        x_hat = x_hat.reshape(h.shape)
        if isinstance(output, tuple):
            return (x_hat,) + output[1:]  # preserve KV cache etc.
        return x_hat

    # Reuse FeatureTracer's layer-finding logic
    if hasattr(llm, "transformer") and hasattr(llm.transformer, "h"):
        layer = llm.transformer.h[layer_idx]         # GPT-2
    elif hasattr(llm, "model") and hasattr(llm.model, "layers"):
        layer = llm.model.layers[layer_idx]           # Qwen / LLaMA
    else:
        raise AttributeError(f"Cannot find layer {layer_idx} on {type(llm).__name__}")

    handle = layer.register_forward_hook(_hook)
    try:
        yield
    finally:
        handle.remove()

