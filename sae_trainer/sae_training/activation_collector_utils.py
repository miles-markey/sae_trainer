from pathlib import Path
import sys

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datasets import load_dataset

class ActivationCollector:
    """
    Collects activations from selected GPT-2 layers:
      - residual_out: output of decoder layer (hidden_states)
      - mlp_down_out: output of layer.mlp.c_proj (down projection)
    """
    def __init__(
        self,
        model: torch.nn.Module,
        layer_ids: List[int],
        capture_residual: bool = True,
        capture_mlp_down: bool = False,
        to_cpu: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        self.model = model
        self.layer_ids = layer_ids
        self.capture_residual = capture_residual
        self.capture_mlp_down = capture_mlp_down
        self.to_cpu = to_cpu
        self.dtype = dtype

        self.hooks = []
        self.buffer: Dict[str, Dict[int, torch.Tensor]] = {
            "residual_out": {},
            "mlp_down_out": {},
        }

    def _clear_batch_buffer(self):
        self.buffer["residual_out"].clear()
        self.buffer["mlp_down_out"].clear()

    @staticmethod
    def _hidden_from_output(output):
        # Some HF modules return tuple; hidden_states is first item
        return output[0] if isinstance(output, tuple) else output

    def _make_block_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            hs = self._hidden_from_output(output)
            t = hs.detach().to(self.dtype)
            if self.to_cpu:
                t = t.cpu()
            self.buffer["residual_out"][layer_idx] = t
        return hook

    def _make_mlp_down_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            t = output.detach().to(self.dtype)
            if self.to_cpu:
                t = t.cpu()
            self.buffer["mlp_down_out"][layer_idx] = t
        return hook

    def register(self):
        self.remove()
        for i in self.layer_ids:
            layer = self.model.transformer.h[i]  # GPT-2: transformer.h[i], QWEN: model.layers[i]
            if self.capture_residual:
                self.hooks.append(layer.register_forward_hook(self._make_block_hook(i)))
            if self.capture_mlp_down:
                self.hooks.append(layer.mlp.c_proj.register_forward_hook(self._make_mlp_down_hook(i))) # GPT-2: mlp.c_proj, QWEN: mlp.down_proj

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def collect_batch(self, model_inputs: Dict[str, torch.Tensor], model: Optional[torch.nn.Module] = None):
        """
        Runs one forward pass and returns flattened activations per layer:
          out[name][layer] -> [batch*seq, d_model]
        """
        if model is None:
            model = self.model

        self._clear_batch_buffer()
        with torch.no_grad():
            _ = model(**model_inputs)

        out = {"residual_out": {}, "mlp_down_out": {}}
        for name in out.keys():
            for layer_idx, x in self.buffer[name].items():
                # [B, S, D] -> [B*S, D]
                out[name][layer_idx] = x.reshape(-1, x.shape[-1])
        return out
    
    def get_layers(self):
        return self.layer_ids


class QwenActivationCollector(ActivationCollector):
    """
    Collects activations from selected Qwen layers:
      - residual_out: output of decoder layer (hidden_states)
      - mlp_down_out: output of layer.mlp.down_proj (down projection)
    """
    def __init__(
        self, 
        model: torch.nn.Module, 
        layer_ids: List[int] = [12, 16, 20, 23], 
        capture_residual: bool = True, 
        capture_mlp_down: bool = False, 
        to_cpu: bool = True,
        dtype: torch.dtype = torch.float32):
        super().__init__(model, layer_ids, capture_residual, capture_mlp_down, to_cpu, dtype)
    
    def register(self):
        self.remove()
        for i in self.layer_ids:
            layer = self.model.model.layers[i]  # GPT-2: transformer.h[i], QWEN: model.layers[i]
            if self.capture_residual:
                self.hooks.append(layer.register_forward_hook(self._make_block_hook(i)))
            if self.capture_mlp_down:
                self.hooks.append(layer.mlp.down_proj.register_forward_hook(self._make_mlp_down_hook(i))) # GPT-2: mlp.c_proj, QWEN: mlp.down_proj

class GPT2ActivationCollector(ActivationCollector):
    """
    Collects activations from selected GPT-2 layers:
      - residual_out: output of decoder layer (hidden_states)
      - mlp_down_out: output of layer.mlp.c_proj (down projection)
    """
    def __init__(
        self, 
        model: torch.nn.Module, 
        layer_ids: List[int] = [3, 6, 9, 11], 
        capture_residual: bool = True, 
        capture_mlp_down: bool = False, 
        to_cpu: bool = True, 
        dtype: torch.dtype = torch.float32):
        super().__init__(model, layer_ids, capture_residual, capture_mlp_down, to_cpu, dtype)
    
    def register(self):
        self.remove()
        for i in self.layer_ids:
            layer = self.model.transformer.h[i]  # GPT-2: transformer.h[i], QWEN: model.layers[i]
            if self.capture_residual:
                self.hooks.append(layer.register_forward_hook(self._make_block_hook(i)))
            if self.capture_mlp_down:
                self.hooks.append(layer.mlp.c_proj.register_forward_hook(self._make_mlp_down_hook(i))) # GPT-2: mlp.c_proj, QWEN: mlp.down_proj

