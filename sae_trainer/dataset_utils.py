from torch.utils.data import Dataset
from typing import List


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