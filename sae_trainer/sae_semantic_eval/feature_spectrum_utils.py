import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Optional

from .feature_tracer_utils import FeatureTracer
from .feature_tracer_eval_utils import build_token_feature_spectrum


def build_sequence_dataset(
    tracer: FeatureTracer,
    n_features: int,
) -> list[np.ndarray]:
    """
    Extract per-prompt feature activation sequences from a tracer.

    Returns a list of float32 arrays, each of shape (T_i, n_features), where T_i
    is the number of traced token positions for prompt i. Prompts with fewer than
    2 traced positions are excluded.
    """
    sequences = []
    for pid in tracer.to_dataframe()["prompt_id"].unique():
        spectrum = build_token_feature_spectrum(pid, tracer, n_features)
        if len(spectrum) < 2:
            continue
        sequences.append(spectrum.values.astype(np.float32))
    return sequences


class FeatureSequenceDataset(Dataset):
    """
    PyTorch Dataset wrapping a list of feature activation sequences.
    Shorter sequences are right-padded with zeros to max_seq_len.
    The padding mask (True = padded position) is returned alongside each sequence.
    """

    def __init__(self, sequences: list[np.ndarray], max_seq_len: Optional[int] = None):
        self.sequences = sequences
        self.max_seq_len = max_seq_len or max(len(s) for s in sequences)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq = self.sequences[idx]  # (T, n_features)
        T, n_features = seq.shape

        if T < self.max_seq_len:
            pad = np.zeros((self.max_seq_len - T, n_features), dtype=np.float32)
            seq_padded = np.vstack([seq, pad])
            padding_mask = [False] * T + [True] * (self.max_seq_len - T)
        else:
            seq_padded = seq[: self.max_seq_len]
            padding_mask = [False] * self.max_seq_len

        return (
            torch.tensor(seq_padded, dtype=torch.float32),
            torch.tensor(padding_mask, dtype=torch.bool),
        )


class FeatureTransformer(nn.Module):
    """
    Small causal transformer for next-token feature vector prediction.

    Projects each input feature vector (n_features) into a lower-dimensional
    model space (d_model), applies causal self-attention over the sequence,
    then projects back to n_features at each position.

    Input:  (batch, seq_len, n_features)
    Output: (batch, seq_len, n_features) — predicted feature vector at each next position
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        n_heads: int = 2,
        n_layers: int = 2,
        max_seq_len: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(d_model, n_features)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.input_proj(x) + self.pos_embedding(positions)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h = self.transformer(
            h,
            mask=causal_mask,
            src_key_padding_mask=src_key_padding_mask,
            is_causal=True,
        )
        return self.output_proj(h)


def train_feature_transformer(
    tracer: FeatureTracer,
    n_features: int,
    d_model: int = 64,
    n_heads: int = 2,
    n_layers: int = 2,
    dropout: float = 0.1,
    n_epochs: int = 30,
    batch_size: int = 16,
    lr: float = 1e-3,
    val_frac: float = 0.2,
    device: str = "cpu",
) -> tuple["FeatureTransformer", dict]:
    """
    Train a FeatureTransformer on per-prompt feature activation sequences.

    Sequences are split by prompt (not by step) to avoid data leakage. At each
    training step the model sees tokens 0..t-1 and predicts token t (teacher forcing).

    Returns:
      model   — trained FeatureTransformer
      history — dict with "train_loss" and "val_loss" lists (one entry per epoch)
    """
    sequences = build_sequence_dataset(tracer, n_features)
    if len(sequences) < 2:
        raise ValueError("Need at least 2 prompts to form a train/val split.")

    n_val = max(1, int(len(sequences) * val_frac))
    rng = np.random.default_rng(42)
    shuffled = rng.permutation(len(sequences))
    val_seqs   = [sequences[i] for i in shuffled[:n_val]]
    train_seqs = [sequences[i] for i in shuffled[n_val:]]

    max_seq_len = max(len(s) for s in sequences)

    train_dl = DataLoader(
        FeatureSequenceDataset(train_seqs, max_seq_len),
        batch_size=batch_size, shuffle=True,
    )
    val_dl = DataLoader(
        FeatureSequenceDataset(val_seqs, max_seq_len),
        batch_size=batch_size,
    )

    model = FeatureTransformer(
        n_features=n_features,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_seq_len=max_seq_len,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    for epoch in range(n_epochs):
        model.train()
        train_losses = []
        for seqs, pad_mask in train_dl:
            seqs, pad_mask = seqs.to(device), pad_mask.to(device)
            x, y     = seqs[:, :-1, :], seqs[:, 1:, :]
            x_mask   = pad_mask[:, :-1]
            y_mask   = pad_mask[:, 1:]
            preds    = model(x, src_key_padding_mask=x_mask)
            valid    = ~y_mask
            loss     = loss_fn(preds[valid], y[valid])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for seqs, pad_mask in val_dl:
                seqs, pad_mask = seqs.to(device), pad_mask.to(device)
                x, y   = seqs[:, :-1, :], seqs[:, 1:, :]
                x_mask = pad_mask[:, :-1]
                y_mask = pad_mask[:, 1:]
                preds  = model(x, src_key_padding_mask=x_mask)
                valid  = ~y_mask
                loss   = loss_fn(preds[valid], y[valid])
                val_losses.append(loss.item())

        history["train_loss"].append(float(np.mean(train_losses)))
        history["val_loss"].append(float(np.mean(val_losses)))

        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch {epoch + 1}/{n_epochs}  "
                f"train_loss={history['train_loss'][-1]:.4f}  "
                f"val_loss={history['val_loss'][-1]:.4f}"
            )

    return model, history


def evaluate_feature_transformer(
    model: FeatureTransformer,
    tracer: FeatureTracer,
    n_features: int,
    k: int = 8,
    device: str = "cpu",
) -> dict:
    """
    Evaluate a trained FeatureTransformer on all sequences in a tracer.

    Returns cosine_sim and precision@k consistent with the Ridge baseline
    metrics from build_next_token_dataset, so results can be compared directly.
    """
    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

    sequences = build_sequence_dataset(tracer, n_features)

    cos_sims: list[float] = []
    prec_at_k: list[float] = []

    model.eval()
    with torch.no_grad():
        for seq in sequences:
            x = torch.tensor(seq[:-1], dtype=torch.float32).unsqueeze(0).to(device)
            y_true = seq[1:]  # (T-1, n_features)
            preds  = model(x).squeeze(0).cpu().numpy()  # (T-1, n_features)

            cos_sims.extend(np.diag(sk_cosine(preds, y_true)).tolist())

            top_pred = np.argsort(preds,  axis=1)[:, -k:]
            top_true = np.argsort(y_true, axis=1)[:, -k:]
            prec_at_k.extend(
                len(set(top_pred[i]) & set(top_true[i])) / k
                for i in range(len(y_true))
            )

    return {
        "cosine_sim":   float(np.mean(cos_sims)),
        "precision@k":  float(np.mean(prec_at_k)),
        "n_samples":    len(cos_sims),
    }
