from dataclasses import dataclass
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import torch
import pandas as pd


@dataclass
class TraceConfig:
    layer_idx: int
    topk_per_token: int = 8
    min_activation: float = 0.0
    max_new_tokens: int = 80
    do_sample: bool = True
    temperature: float = 0.7
    context_window: int = 8


class FeatureTracer:
    """
    Trace SAE features during LLM generation.

    Assumes:
      - llm is a HF CausalLM with `.model.layers[...]`
      - sae forward returns (x_hat, z) or (x_hat, z, h)
      - SAE input dim matches hooked layer hidden dim
    """

    def __init__(self, llm, tokenizer, sae, device: str, config: TraceConfig):
        self.llm = llm
        self.tokenizer = tokenizer
        self.sae = sae
        self.device = device
        self.cfg = config

        self._hook_handle = None
        self._captured_hidden: List[torch.Tensor] = []
        self._rows: List[dict] = []

    # ---------- Hooking ----------
    def _hook_fn(self, module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        self._captured_hidden.append(h.detach())

    def _register_hook(self):
        self._remove_hook()
        layer = self.llm.model.layers[self.cfg.layer_idx]
        self._hook_handle = layer.register_forward_hook(self._hook_fn)

    def _remove_hook(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def reset(self):
        self._captured_hidden.clear()
        self._rows.clear()

    # ---------- Core tracing ----------
    @torch.no_grad()
    def trace_prompt(self, prompt: str, prompt_id: Optional[str] = None) -> Dict:
        self._captured_hidden = []
        self._register_hook()

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        out = self.llm.generate(
            **inputs,
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=self.cfg.do_sample,
            temperature=self.cfg.temperature,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        self._remove_hook()

        # Hidden states collected across generation calls; concatenate on seq axis
        H = torch.cat(self._captured_hidden, dim=1)  # [1, seq, d_model]
        flat = H.reshape(-1, H.shape[-1])

        sae_out = self.sae(flat)
        if isinstance(sae_out, tuple):
            z = sae_out[1]
        elif isinstance(sae_out, dict):
            z = sae_out.get("z", sae_out.get("latents", sae_out.get("codes")))
        else:
            raise ValueError("Unexpected SAE output format")

        Z = z.view(H.shape[0], H.shape[1], -1)[0]  # [seq, d_latent]

        token_ids = out[0].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)

        top_vals, top_idx = torch.topk(
            Z, k=min(self.cfg.topk_per_token, Z.shape[-1]), dim=-1
        )

        pid = prompt_id if prompt_id is not None else str(len(self._rows))

        for pos in range(min(len(tokens), Z.shape[0])):
            for j in range(top_idx.shape[1]):
                val = float(top_vals[pos, j])
                if val <= self.cfg.min_activation:
                    continue
                fid = int(top_idx[pos, j])
                self._rows.append(
                    {
                        "prompt_id": pid,
                        "prompt": prompt,
                        "generated_text": text,
                        "token_pos": pos,
                        "token": tokens[pos],
                        "feature_id": fid,
                        "activation": val,
                    }
                )

        return {
            "prompt_id": pid,
            "prompt": prompt,
            "generated_text": text,
            "num_tokens": len(tokens),
            "num_hits": sum(1 for r in self._rows if r["prompt_id"] == pid),
        }

    def trace_prompts(self, prompts: List[str], ids: Optional[List[str]] = None) -> List[Dict]:
        summaries = []
        for i, p in enumerate(prompts):
            pid = ids[i] if ids is not None else f"p{i}"
            summaries.append(self.trace_prompt(p, prompt_id=pid))
        return summaries

    # ---------- Analysis ----------
    def to_dataframe(self) -> pd.DataFrame:
        if not self._rows:
            return pd.DataFrame(
                columns=[
                    "prompt_id", "prompt", "generated_text",
                    "token_pos", "token", "feature_id", "activation"
                ]
            )
        return pd.DataFrame(self._rows)

    def top_features(self, n: int = 25) -> pd.DataFrame:
        df = self.to_dataframe()
        if df.empty:
            return df
        out = (
            df.groupby("feature_id")
              .agg(
                  hits=("feature_id", "count"),
                  mean_activation=("activation", "mean"),
                  max_activation=("activation", "max"),
              )
              .sort_values(["hits", "mean_activation"], ascending=False)
              .head(n)
              .reset_index()
        )
        return out

    def feature_contexts(
        self,
        feature_id: int,
        top_n: int = 15,
        window: Optional[int] = None
    ) -> pd.DataFrame:
        w = self.cfg.context_window if window is None else window
        df = self.to_dataframe()
        if df.empty:
            return pd.DataFrame()

        sub = df[df["feature_id"] == feature_id].sort_values("activation", ascending=False).head(top_n)
        rows = []
        for _, r in sub.iterrows():
            toks = self.tokenizer.convert_ids_to_tokens(
                self.tokenizer(r["generated_text"], return_tensors="pt")["input_ids"][0]
            )
            pos = int(r["token_pos"])
            lo, hi = max(0, pos - w), min(len(toks), pos + w + 1)
            ctx = self.tokenizer.convert_tokens_to_string(toks[lo:hi])
            rows.append(
                {
                    "prompt_id": r["prompt_id"],
                    "token_pos": pos,
                    "token": r["token"],
                    "activation": r["activation"],
                    "context": ctx,
                }
            )
        return pd.DataFrame(rows)

    def save_csv(self, path: str):
        self.to_dataframe().to_csv(path, index=False)