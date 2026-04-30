from dataclasses import dataclass
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Literal
import torch
import pandas as pd


@dataclass
class TraceConfig:
    layer_idx: int
    topk_per_token: int = 8
    min_activation: float = 0.0
    max_new_tokens: int = 20
    do_sample: bool = True
    temperature: float = 0.2
    repetition_penalty: float = 1.0   # >1 discourages repetition; 1.3-1.5 works well for GPT-2
    stop_strings: Optional[List[str]] = None  # e.g. ["\n"] to stop at first newline
    context_window: int = 8
    # "all"            — include both prompt and generated tokens
    # "generated_only" — exclude prompt tokens (default, previous behaviour)
    # "prompt_only"    — exclude generated tokens
    token_mode: Literal["all", "generated_only", "prompt_only"] = "generated_only"


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
        self._feature_embeddings = None

    # ---------- Hooking ----------
    def _hook_fn(self, module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        self._captured_hidden.append(h.detach())

    def _get_layer(self, idx: int):
        # GPT-2 style: model.transformer.h[i]
        if hasattr(self.llm, "transformer") and hasattr(self.llm.transformer, "h"):
            return self.llm.transformer.h[idx]
        # LLaMA / Qwen style: model.model.layers[i]
        if hasattr(self.llm, "model") and hasattr(self.llm.model, "layers"):
            return self.llm.model.layers[idx]
        raise AttributeError(
            f"Cannot find layer list on {type(self.llm).__name__}. "
            "Expected `model.transformer.h` (GPT-2) or `model.model.layers` (LLaMA/Qwen)."
        )

    def _register_hook(self):
        self._remove_hook()
        layer = self._get_layer(self.cfg.layer_idx)
        self._hook_handle = layer.register_forward_hook(self._hook_fn)

    def _remove_hook(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def reset(self):
        self._captured_hidden.clear()
        self._rows.clear()
        self._feature_embeddings = None

    # ---------- Input formatting ----------
    def _format_inputs(self, prompt: str, system_prompt: Optional[str]) -> Dict:
        if system_prompt is not None and getattr(self.tokenizer, "chat_template", None) is not None:
            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}]
            input_ids = self.tokenizer.apply_chat_template(
                messages, return_tensors="pt", add_generation_prompt=True
            ).to(self.device)
            return {"input_ids": input_ids}
        # Fallback for models without a chat template (e.g. GPT-2 base)
        text = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        return self.tokenizer(text, return_tensors="pt").to(self.device)

    # ---------- Core tracing ----------
    @torch.no_grad()
    def trace_prompt(self, prompt: str, prompt_id: Optional[str] = None, system_prompt: Optional[str] = None) -> Dict:
        self._captured_hidden = []
        self._register_hook()

        inputs = self._format_inputs(prompt, system_prompt)
        num_prompt_tokens = inputs["input_ids"].shape[1]
        generate_kwargs = dict(
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=self.cfg.do_sample,
            repetition_penalty=self.cfg.repetition_penalty,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if self.cfg.do_sample:
            generate_kwargs["temperature"] = self.cfg.temperature
        if self.cfg.stop_strings:
            generate_kwargs["stop_strings"] = self.cfg.stop_strings
            generate_kwargs["tokenizer"] = self.tokenizer
        torch.manual_seed(hash(prompt_id) % (2**32)) # Set seed for reproducability
        out = self.llm.generate(**inputs, **generate_kwargs)

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
            if self.cfg.token_mode == "generated_only" and pos < num_prompt_tokens:
                continue
            if self.cfg.token_mode == "prompt_only" and pos >= num_prompt_tokens:
                continue
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
            "num_prompt_tokens": num_prompt_tokens,
            "num_tokens": len(tokens),
            "num_hits": sum(1 for r in self._rows if r["prompt_id"] == pid),
        }

    def trace_prompts(self, prompts: List[str], ids: Optional[List[str]] = None, system_prompt: Optional[str] = None) -> List[Dict]:
        summaries = []
        for i, p in enumerate(prompts):
            pid = ids[i] if ids is not None else f"p{i}"
            summaries.append(self.trace_prompt(p, prompt_id=pid, system_prompt=system_prompt))
        return summaries

    def trace_prompts_from_iterable_dataset(
            self,
            ds,
            min_prompt_words: int = 50,
            truncation_limit: int = 150,
            max_prompts_to_trace: Optional[int] = None,
            system_prompt: Optional[str] = None,
        ) -> List[Dict]:
        summaries = []
        num_traced = 0
        for row in ds:
            text = row["text"].strip()
            if len(text.split()) >= min_prompt_words:
                summaries.append(self.trace_prompt(" ".join(text.split()[:truncation_limit]), prompt_id=num_traced, system_prompt=system_prompt))
                num_traced += 1
            if max_prompts_to_trace and num_traced >= max_prompts_to_trace:
                break
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

    def _expand_token_variants(self, tokens: List[str]) -> List[str]:
        """Add BPE prefix variants so callers don't need to know the tokenizer's prefix char."""
        expanded = set()
        for t in tokens:
            expanded.add(t)
            expanded.add("Ġ" + t)   # GPT-2 / RoBERTa word-initial prefix
            expanded.add("▁" + t)   # SentencePiece word-initial prefix
            expanded.add(t.lstrip("Ġ▁"))  # strip prefix if caller passed it
        return list(expanded)

    def token_positions(
        self,
        tokens: List[str],
        prompt_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """Return a DataFrame of (prompt_id, token_pos, token) for every occurrence
        of the requested token strings, useful for deciding what to filter on."""
        df = self.to_dataframe()
        if prompt_id is not None:
            df = df[df["prompt_id"] == prompt_id]
        mask = df["token"].isin(self._expand_token_variants(tokens))
        return (
            df[mask][["prompt_id", "token_pos", "token"]]
            .drop_duplicates()
            .sort_values(["prompt_id", "token_pos"])
            .reset_index(drop=True)
        )

    def _apply_token_filter(
        self,
        df: pd.DataFrame,
        tokens: Optional[List[str]],
        token_pos: Optional[List[int]],
    ) -> pd.DataFrame:
        if tokens is not None:
            df = df[df["token"].isin(self._expand_token_variants(tokens))]
        if token_pos is not None:
            df = df[df["token_pos"].isin(token_pos)]
        return df

    def top_features(
        self,
        n: int = 25,
        tokens: Optional[List[str]] = None,
        token_pos: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        df = self._apply_token_filter(self.to_dataframe(), tokens, token_pos)
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
        window: Optional[int] = None,
        tokens: Optional[List[str]] = None,
        token_pos: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        w = self.cfg.context_window if window is None else window
        df = self._apply_token_filter(self.to_dataframe(), tokens, token_pos)
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

    def compute_feature_embeddings(
        self,
        feature_ids: Optional[List[int]] = None,
        contexts_per_feature: int = 50,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 64,
    ) -> Dict[int, Dict]:
        """
        For each feature, embed its top activating context windows using a
        sentence-transformer model.

        Returns a dict keyed by feature_id:
          {
            "embeddings": np.ndarray [n, d],   # one row per context
            "activations": np.ndarray [n],     # corresponding activation values
            "contexts": List[str],             # decoded context strings
          }
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("pip install sentence-transformers")

        import numpy as np

        embed_model = SentenceTransformer(model_name)

        df = self.to_dataframe()
        if feature_ids is None:
            feature_ids = df["feature_id"].unique().tolist()

        results = {}
        for fid in feature_ids:
            ctx_df = self.feature_contexts(fid, top_n=contexts_per_feature)
            if ctx_df.empty:
                continue

            contexts = ctx_df["context"].tolist()
            activations = ctx_df["activation"].to_numpy()
            prompt_ids = ctx_df["prompt_id"].tolist()
            token_positions = ctx_df["token_pos"].to_numpy()

            embeddings = embed_model.encode(
                contexts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,  # unit-norm so cosine sim = dot product
            )

            results[fid] = {
                "embeddings": embeddings,
                "activations": activations,
                "contexts": contexts,
                "prompt_ids": prompt_ids,
                "token_positions": token_positions,
            }

        return results

    def feature_specificity_scores(
        self,
        #feature_embeddings: Optional[Dict] = None,
        feature_ids: Optional[List[int]] = None,
        contexts_per_feature: int = 50,
        model_name: str = "all-MiniLM-L6-v2",
        weighted: bool = True,
        mask_same_context: bool = True,
        min_prompts: int = 5,   # minimum distinct prompts a feature must appear in to be scored
    ) -> pd.DataFrame:
        """
        Compute a specificity score for each feature as the mean pairwise cosine
        similarity of its activating context embeddings, optionally weighted by
        activation strength.

        Also computes a baseline (mean similarity of randomly sampled context pairs
        across all features) so scores can be interpreted relatively.

        Returns a DataFrame with columns:
          feature_id, hits, mean_activation, specificity, specificity_vs_baseline
        """
        import numpy as np

        if self._feature_embeddings is None:
            self._feature_embeddings = self.compute_feature_embeddings(
                feature_ids=feature_ids,
                contexts_per_feature=contexts_per_feature,
                model_name=model_name,
            )

        def _weighted_mean_cosine(
            embeddings: "np.ndarray",
            weights: "np.ndarray",
            valid_pairs_mask: "np.ndarray",  # [n, n] bool — True = include this pair
        ) -> float:
            # embeddings are already unit-normed, so dot product == cosine similarity
            sim_matrix = embeddings @ embeddings.T  # [n, n]
            n = len(weights)
            if n < 2:
                return float("nan")
            w_outer = weights[:, None] * weights[None, :]
            # Exclude diagonal (self-similarity) and any masked pairs
            include = valid_pairs_mask & ~np.eye(n, dtype=bool)
            if include.sum() == 0:
                return float("nan")
            return float((sim_matrix[include] * w_outer[include]).sum() / w_outer[include].sum())

        # Compute per-feature scores
        rows = []
        for fid, data in self._feature_embeddings.items():
            emb = data["embeddings"]
            acts = data["activations"]
            n = len(acts)

            pids = np.array(data["prompt_ids"]) if "prompt_ids" in data else None
            n_distinct_prompts = len(set(pids)) if pids is not None else n
            if n_distinct_prompts < min_prompts:
                continue

            weights = acts / (acts.sum() + 1e-8) if weighted else np.ones(n) / n

            if mask_same_context and pids is not None:
                # Mask out all pairs from the same prompt — even distant token positions
                # share document-level similarity (topic, style, vocabulary) that would
                # inflate specificity scores independently of the feature's actual semantics
                same_prompt = pids[:, None] == pids[None, :]
                valid_pairs_mask = ~same_prompt
            else:
                valid_pairs_mask = np.ones((n, n), dtype=bool)

            score = _weighted_mean_cosine(emb, weights, valid_pairs_mask)
            rows.append({
                "feature_id": fid,
                "hits": len(acts),
                "n_prompts": n_distinct_prompts,
                "mean_activation": float(acts.mean()),
                "specificity": score,
            })

        result_df = pd.DataFrame(rows)
        if result_df.empty:
            return result_df

        # Baseline: mean cosine sim between random context pairs across all features
        all_embeddings = np.concatenate([d["embeddings"] for d in self._feature_embeddings.values()])
        rng = np.random.default_rng(42)
        n_baseline = min(1000, len(all_embeddings))
        idx = rng.choice(len(all_embeddings), size=(n_baseline, 2))
        baseline = float((all_embeddings[idx[:, 0]] * all_embeddings[idx[:, 1]]).sum(axis=1).mean())

        result_df["specificity_vs_baseline"] = result_df["specificity"] - baseline
        result_df = result_df.sort_values("specificity_vs_baseline", ascending=False).reset_index(drop=True)
        return result_df

    def get_feature_embeddings(self):
        return self._feature_embeddings
    
    def save_csv(self, path: str):
        self.to_dataframe().to_csv(path, index=False)