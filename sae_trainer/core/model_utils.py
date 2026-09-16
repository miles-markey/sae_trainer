from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
from typing import Optional

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

def _get_layer(llm, layer_idx: int) -> nn.Module:
    if hasattr(llm, "transformer") and hasattr(llm.transformer, "h"):
        return llm.transformer.h[layer_idx]
    if hasattr(llm, "model") and hasattr(llm.model, "layers"):
        return llm.model.layers[layer_idx]
    raise AttributeError(f"Cannot find layer {layer_idx} on {type(llm).__name__}")


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

    handle = _get_layer(llm, layer_idx).register_forward_hook(_hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def _capture_hidden_for_training(llm, layer_idx: int):
    """
    Capture hidden states at layer_idx while retaining the computation graph.

    Unlike the inference hooks, this does NOT detach or use torch.no_grad(),
    so gradients flow back through the captured tensor to update LLM parameters.
    """
    buf: list[torch.Tensor] = []

    def _hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        buf.append(h)  # no detach — must stay in graph for backward

    handle = _get_layer(llm, layer_idx).register_forward_hook(_hook)
    try:
        yield buf
    finally:
        handle.remove()


def sae_regularized_finetune(
    llm,
    sae,
    tokenizer,
    texts: list[str],
    layer_idx: int,
    target_features: dict[int, float],
    lambda_sae: float = 0.1,
    n_epochs: int = 3,
    batch_size: int = 4,
    lr: float = 2e-5,
    max_length: int = 512,
    device: str = "mps",
    log_every: int = 10,
    eval_sae_drift_every: Optional[int] = None,
    run=None
) -> dict:
    """
    Fine-tune an LLM with a frozen SAE used as a feature-space regulariser.

        Loss = L_CE  +  lambda_sae * L_SAE

    L_SAE is the MSE between each target feature's per-token activation and its
    target value, averaged over all non-padding token positions in the batch.

        target_features = {
            42:  2.0,   # encourage feature 42 to activate at strength 2.0
            107: 0.0,   # suppress feature 107
        }

    The SAE is fully frozen (requires_grad=False) throughout. Gradients for L_SAE
    flow *through* the SAE's encoder weights (as fixed linear ops) back to the LLM
    hidden states, updating the LLM so it naturally produces the desired activations.

    Parameters
    ----------
    texts               : list of training strings (plain text, not pre-tokenised)
    layer_idx           : layer at which the SAE was trained / should be applied
    target_features     : {feature_id: target_mean_activation} — see above
    lambda_sae          : weight of L_SAE relative to L_CE; start small (0.01–0.1)
    eval_sae_drift_every: if set, evaluates CE loss with SAE inserted every N steps
                          to monitor how much the LLM has drifted from the SAE's
                          expected input distribution

    Returns
    -------
    history dict with keys:
      "ce_loss", "sae_loss", "total_loss"  — one entry per optimizer step
      "feature_activations"                — {fid: [mean activation per step]}
      "sae_drift_ce"                       — CE loss under SAE insertion, if evaluated
    """
    from torch.optim import AdamW

    sae.requires_grad_(False)
    sae.eval()
    llm.train()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    optimizer = AdamW(llm.parameters(), lr=lr)

    target_fids = list(target_features.keys())
    target_vals = torch.tensor(
        [target_features[f] for f in target_fids], dtype=torch.float32, device=device
    )  # (n_target_features,)

    history: dict = {
        "ce_loss": [], "sae_loss": [], "total_loss": [],
        "feature_activations": {fid: [] for fid in target_fids},
        "sae_drift_ce": [],
    }

    global_step = 0
    indices = list(range(len(texts)))

    for epoch in range(n_epochs):
        perm = torch.randperm(len(indices)).tolist()

        for batch_start in range(0, len(texts), batch_size):
            batch = [texts[perm[i]] for i in range(
                batch_start, min(batch_start + batch_size, len(texts))
            )]

            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)

            # Mask padding positions from CE loss
            labels = inputs["input_ids"].clone()
            labels[inputs["attention_mask"] == 0] = -100

            # --- Forward pass ---
            # _capture_hidden_for_training keeps h in the computation graph so
            # L_SAE gradients reach LLM parameters through the SAE encoder.
            with _capture_hidden_for_training(llm, layer_idx) as buf:
                outputs = llm(**inputs, labels=labels)

            L_CE = outputs.loss

            # --- SAE regularisation loss ---
            h = buf[0]                              # (batch, seq, d_model)
            mask = inputs["attention_mask"].bool()  # (batch, seq)
            h_valid = h[mask]                       # (n_tokens, d_model) — no padding

            # SAE is frozen: encoder/decoder act as fixed linear ops here.
            # Gradients flow through h_valid → h → LLM params.
            _, z = sae(h_valid)                     # (n_tokens, d_latent)

            z_target_feats = z[:, target_fids]      # (n_tokens, n_target_features)
            L_SAE = F.mse_loss(
                z_target_feats,
                target_vals.unsqueeze(0).expand_as(z_target_feats),
            )

            loss = L_CE + lambda_sae * L_SAE

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(llm.parameters(), max_norm=1.0)
            optimizer.step()

            # --- Logging ---
            history["ce_loss"].append(L_CE.item())
            history["sae_loss"].append(L_SAE.item())
            history["total_loss"].append(loss.item())
            with torch.no_grad():
                for fid in target_fids:
                    history["feature_activations"][fid].append(
                        z[:, fid].mean().item()
                    )

            # --- SAE drift check ---
            # Measures whether the LLM's hidden states have drifted away from the
            # distribution the SAE was trained on, by computing CE loss when the
            # SAE is inserted in the forward path.
            if eval_sae_drift_every and (global_step + 1) % eval_sae_drift_every == 0:
                llm.eval()
                with torch.no_grad(), sae_inserted_llm(llm, sae, layer_idx, device):
                    drift_out = llm(**inputs, labels=labels)
                history["sae_drift_ce"].append(drift_out.loss.item())
                llm.train()

            global_step += 1

            if global_step % log_every == 0:
                feat_str = "  ".join(
                    f"f{fid}={history['feature_activations'][fid][-1]:.3f}"
                    for fid in target_fids
                )
                print(
                    f"epoch {epoch+1}/{n_epochs}  step {global_step}"
                    f"  L_CE={L_CE.item():.4f}"
                    f"  L_SAE={L_SAE.item():.4f}"
                    f"  total={loss.item():.4f}"
                    f"  {feat_str}"
                )
                if run:
                    run.log({
                        "train/ce_loss": L_CE.item(),
                        "train/sae_loss": L_SAE.item(),
                        "train/total_loss": loss.item(),
                    })

    sae.requires_grad_(True)
    return llm, history


@contextmanager
def sae_steered_llm(llm, sae, layer_idx, feature_id: int, delta: float, device=None):
    """
    Context manager that steers a single SAE feature during LLM inference.

    At each forward pass through layer_idx, the hidden state is encoded through
    the SAE, the target feature's activation is shifted by `delta`, then the
    modified latent is decoded back to hidden-state space.

    delta > 0  — amplify / activate the feature
    delta < 0  — suppress / ablate the feature
    delta = 0  — equivalent to plain SAE insertion with no steering

    Uses additive steering (z[feature_id] += delta) so the intervention works
    even when the feature is not currently active (z = 0 after TopK sparsity).
    To ablate completely regardless of current activation, use delta = -z[feature_id],
    or clamp after adding: z[:, feature_id].clamp_(min=0).
    """
    def _hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        flat = h.reshape(-1, h.shape[-1])
        with torch.no_grad():
            _, z = sae(flat)
            z[:, feature_id] = (z[:, feature_id] + delta).clamp(min=0)
            x_hat = sae._decode(z)
        x_hat = x_hat.reshape(h.shape)
        if isinstance(output, tuple):
            return (x_hat,) + output[1:]
        return x_hat

    handle = _get_layer(llm, layer_idx).register_forward_hook(_hook)
    try:
        yield
    finally:
        handle.remove()

