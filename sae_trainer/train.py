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

import wandb

from .dataset_utils import make_collate_fn, TextDataset, get_data_loaders
from .sae_training_module import ReluSAETrainingModule, TopKSAETrainingModule
from .eval_utils import evaluate_sae, visualize_sae
from .model_utils import ReluSparseAutoencoder, TopKSparseAutoencoder
from .activation_collector_utils import GPT2ActivationCollector, QwenActivationCollector

def load_config(path: str) -> SimpleNamespace:
    with open(path) as f:
        data = yaml.safe_load(f)
    return SimpleNamespace(**data)

def training_wrapper(cfg, accum, layer_idx, device, save_mode=False, show_curves=False):

    run = None
    if cfg.use_wandb:
        run = wandb.init(
            project=cfg.wandb_project,
            name=f"{cfg.model_name}_{cfg.dataset_name}_layer{layer_idx}",
            config=vars(cfg),
            reinit=True,
        )

    train_loader, val_loader, d_in, act_scale = get_data_loaders(accum, layer_idx, cfg.sae_batch_size)
    if run:
        run.summary["act_scale"] = act_scale.item()
    # ---- Model + optimizer ----
    expansion = cfg.expansion_factor               # 4-16 are common starting points
    d_latent = d_in * expansion # d_in = 3584, so this should be in ~[14k, 60k] (14336-57344)

    if cfg.sae_type == "topk":
        sae_training_module = TopKSAETrainingModule(d_in=d_in, d_latent=d_latent, k=cfg.k, normalize_decoder=cfg.normalize_decoder, device=device)
    else:
        sae_training_module = ReluSAETrainingModule(d_in=d_in, d_latent=d_latent, normalize_decoder=cfg.normalize_decoder, device=device)
    opt = torch.optim.AdamW(sae_training_module.sae.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Optional: LR scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.num_epochs)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=5)

    sae, history = sae_training_module.train_sae(
        train_loader,
        val_loader,
        opt,
        scheduler,
        device,
        cfg=cfg,
        run=run,
        )

    if run:
        run.finish()

    if save_mode:
        save_filename = f"{cfg.sae_type}_sae_{cfg.model_name}_{cfg.dataset_name}_layer{layer_idx}.pt"
        # ---- Save checkpoint ----
        ckpt = {
            "model_state": sae.state_dict(),
            "d_in": d_in,
            "d_latent": d_latent,
            "act_scale": act_scale.item(),
            "history": history,
        }
        torch.save(ckpt, save_filename)
        print(f"Saved: {save_filename}")

    return sae, history, train_loader, val_loader

def get_model(cfg, device):
    if cfg.model_name == "gpt2":
        model_name = "gpt2"
        collector = GPT2ActivationCollector
    elif cfg.model_name == "qwen":
        model_name = "Qwen/Qwen2-0.5B-Instruct"
        collector = QwenActivationCollector
    else:
        raise ValueError(f"Invalid model name: {cfg.model_name}. Expected 'gpt2' or 'qwen'.")    

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    return model, tokenizer, collector

def get_dataloader(cfg, tokenizer):
    texts = []
    if cfg.dataset_name == "openwebtext":
        ds = load_dataset("openwebtext", split="train", streaming=True)
    elif cfg.dataset_name == "wikitext":
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    else:
        raise ValueError(f"Invalid dataset name: {cfg.dataset_name}. Expected 'openwebtext' or 'wikitext'.")

    # set max_texts to 1.2*(cfg.max_batches*collection_batch_size) to give us a little headroom
    max_texts = 1.2 * cfg.max_batches * cfg.collection_batch_size

    for row in ds:
        if len(texts) >= max_texts:
            break
        if row["text"].strip():
            texts.append(row["text"])
    dataset = TextDataset(texts)

    loader = DataLoader(
        dataset,
        batch_size=cfg.collection_batch_size,
        shuffle=False,
        collate_fn=make_collate_fn(tokenizer, max_length=128),
    )

    return loader

def collect_activations(loader, collector, target_layers, device, max_batches=50):

    # --- Dataloader loop to accumulate activations ---
    # Stores per-layer chunks, then concatenates at end.
    accum = {"residual_out": {i: [] for i in target_layers},
            "mlp_down_out": {i: [] for i in target_layers}}
    
    for step, batch in enumerate(loader):
        if step >= max_batches:
            break

        batch = {k: v.to(device) for k, v in batch.items()}
        acts = collector.collect_batch(batch)

        # Optional: remove pad tokens before append
        # (for now we keep all tokens; filtering can be done later)

        for name in ["residual_out", "mlp_down_out"]:
            for i, x in acts[name].items():
                accum[name][i].append(x)

    # concat
    for name in ["residual_out", "mlp_down_out"]:
        for i in target_layers:
            if len(accum[name][i]) > 0:
                accum[name][i] = torch.cat(accum[name][i], dim=0)
            else:
                accum[name][i] = None

    collector.remove()

    for i in target_layers:
        x = accum["residual_out"][i]
        print(f"Layer {i} residual:", None if x is None else tuple(x.shape))
    
    return accum



def train(cfg, args, device):
    save_mode = args.save_saes

    model, tokenizer, collector_class = get_model(cfg, device)

    loader = get_dataloader(cfg, tokenizer)

    collector = collector_class(model=model)
    target_layers = collector.get_layers()
    collector.register()

    accum = collect_activations(loader, collector, target_layers, device, max_batches=cfg.max_batches)
    
    saes = {}
    histories = {}
    train_loaders = {}
    val_loaders = {}

    for target_layer in target_layers:
        print(f"Training SAE for layer {target_layer}")
        sae, history, train_loader, val_loader = training_wrapper(cfg, accum, target_layer, device, save_mode=save_mode)
        saes[target_layer] = sae
        histories[target_layer] = history
        train_loaders[target_layer] = train_loader
        val_loaders[target_layer] = val_loader

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--save-saes", action="store_true", default=False)
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg, args, device)