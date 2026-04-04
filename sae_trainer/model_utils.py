import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional
import torch.nn as nn
import torch.nn.functional as F

# ---- Simple ReLU SAE ----
class SparseAutoencoder(nn.Module):
    def __init__(self, d_in: int, d_latent: int):
        super().__init__()
        self.encoder = nn.Linear(d_in, d_latent, bias=True)
        self.decoder = nn.Linear(d_latent, d_in, bias=False)

        # Optional: decoder row norm stabilization can help
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)

    def forward(self, x):
        z = F.relu(self.encoder(x))   # sparse nonnegative code
        x_hat = self.decoder(z)
        return x_hat, z