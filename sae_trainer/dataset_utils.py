from torch.utils.data import Dataset
from typing import List
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split

class TextDataset(Dataset):
    def __init__(self, texts: List[str]):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

def make_collate_fn(tokenizer, max_length=256):
    def collate(batch_texts):
        return tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
    return collate

def get_data_loaders(accum, layer_idx, batch_size=2048):
    x = accum["residual_out"][layer_idx].float().cpu()

    # Normalize by mean L2 norm so reconstruction and sparsity losses are balanced in scale.
    # Scale factor is returned so callers can store it in checkpoints for denormalization at inference.
    scale = x.norm(dim=-1).mean().clamp(min=1e-8)
    x = x / scale

    dataset = TensorDataset(x)
    n_total = len(dataset)
    n_val = max(1, int(0.05 * n_total))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, x.shape[1], scale