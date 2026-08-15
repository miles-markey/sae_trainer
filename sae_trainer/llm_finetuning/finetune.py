from pathlib import Path
import sys
import argparse
from types import SimpleNamespace
import yaml

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datasets import load_dataset
from types import SimpleNamespace
import wandb

from sae_trainer.core.model_utils import ReluSparseAutoencoder, TopKSparseAutoencoder, sae_inserted_llm
from sae_trainer.sae_training.train import training_wrapper, get_model, get_dataloader, collect_activations
from sae_trainer.core.model_utils import sae_regularized_finetune

def load_config(path: str) -> SimpleNamespace:
    with open(path) as f:
        data = yaml.safe_load(f)
    return SimpleNamespace(**data)

def get_sae(cfg, device):
    # Load SAE checkpoint — pick whichever layer you want to analyse (3, 6, 9, or 11)
    ckpt = torch.load(f"model_weights_files/{cfg.sae_type}_sae_{cfg.model_name}_{cfg.dataset_name}_layer{cfg.layer_idx}.pt", map_location=device)
    if cfg.sae_type == 'relu':
        sae = ReluSparseAutoencoder(d_in=ckpt["d_in"], d_latent=ckpt["d_latent"], normalize_decoder=True).to(device)
    elif cfg.sae_type == 'topk':
        sae = TopKSparseAutoencoder(d_in=ckpt["d_in"], d_latent=ckpt["d_latent"], k=cfg.k, normalize_decoder=True).to(device)
    else:
        raise ValueError(f"Invalid sae type: {cfg.sae_type}. Expected either 'relu' or 'qwen'.")
    sae.load_state_dict(ckpt["model_state"])
    sae.eval()

    return sae

def get_training_texts(cfg):
    n_train_docs = cfg.n_train_docs # 4000 # cfg.max_batches * cfg.collection_batch_size

    if cfg.dataset_name == 'openwebtext':
        ds = load_dataset("openwebtext", split="train", streaming=True)
    else:
        raise ValueError(f'Unsupported datasetname for LLM finetuning: {cfg.dataset_name}')

    train_texts = []
    for row in ds.take(n_train_docs * 2):  # 2× headroom for the length filter
        text = row["text"].strip()
        if len(text.split()) >= 50:
            train_texts.append(" ".join(text.split()[:350]))
        if len(train_texts) >= n_train_docs:
            break
    
    return train_texts

def finetune(cfg, args, device):
    save_mode = args.save_llm
    
    run = None
    if cfg.use_wandb:
        run = wandb.init(
            project=cfg.wandb_project,
            name=f"{cfg.model_name}_{cfg.sae_type}__{cfg.dataset_name}_layer{cfg.layer_idx}",
            config=vars(cfg),
            reinit=True,
        )

    llm, tokenizer, _ = get_model(cfg, device)
    sae = get_sae(cfg, device=device)

    train_texts = get_training_texts(cfg)

    llm, history = sae_regularized_finetune(
        llm, sae, tokenizer,
        texts=train_texts,
        layer_idx=cfg.layer_idx,
        target_features={42: 2.0, 107: 0.0},  # amplify 42, suppress 107
        lambda_sae=cfg.lambda_sae,
        n_epochs=cfg.num_epochs,
        batch_size=cfg.batch_size,
        eval_sae_drift_every=cfg.eval_sae_drift_every,
        max_length=cfg.max_length,
        lr=cfg.lr
    )

    if run:
        run.finish()

    if save_mode:
        save_filename = f"model_weights_files/fine_tuned_llms/{cfg.sae_type}_sae_{cfg.model_name}_{cfg.dataset_name}_layer{cfg.layer_idx}.pt"
        # ---- Save checkpoint ----
        ckpt = {
            "model_state": llm.state_dict(),
            "history": history,
        }
        torch.save(ckpt, save_filename)
        print(f"Saved: {save_filename}")

    return sae, history


if __name__ == '__main__':
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--save-llm", action="store_true", default=False)
    args = parser.parse_args()
    cfg = load_config(args.config)
    finetune(cfg, args, device)