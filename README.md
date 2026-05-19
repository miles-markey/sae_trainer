# sae_trainer

See demo at: https://miles-markey.github.io/sae_trainer/

This package is divided into two useful use cases: training (in the “sae_training/” directory and) sae evaluation (in the “sae_semantic_eval/”) directory.

## sae_training

For SAE training, this package currently supports two LLM types: gpt-2 for smaller, proof-of-concept experiments, and Qwen-0.5 for production experiments. It also supports two SAE architectures: the traditional ReLU-SAE, which applies L1 and KL-divergence penalties to encourage sparse activations, and the more recent TopK-SAE, which enforces sparsity directly by hard-capping the number of active features per forward pass and employs dead neuron resampling to ensure those active features learn meaningful, non-redundant representations. For SAE training the package currently supports two datasets: “wikitext” and “openwebtext”, and a number of relevant training hyperparameters described in Table X. The package supports weights and biases integration for monitoring training curves. A full breakdown of the available SAE-training config parameters can be found in the following table.

Configurable parameters for SAE Training
| Parameter Name | Description |
|---|---|
| `model_name` | Name of the LLM the SAE will target. Either `gpt2` or `qwen`. |
| `sae_type` | SAE architecture. Either `relu` or `topk`. |
| `expansion_factor` | Expansion factor from the SAE input dimension to its latent space dimension. A larger value yields a wider latent space with more learnable features. |
| `normalize_decoder` | Whether to normalize the decoder weights during training, which can improve training stability. |
| `k` | Number of active features per forward pass. Only applicable when `sae_type: topk`. |
| `collection_batch_size` | Batch size used when collecting LLM activations from the dataset prior to SAE training. |
| `dataset_name` | Dataset used to generate LLM activations for SAE training. Either `wikitext` or `openwebtext`. |
| `max_batches` | Maximum number of activation batches to collect from the dataset. Controls the total volume of training data. |
| `use_wandb` | Whether to enable Weights & Biases integration for logging training curves and metrics. |
| `wandb_project` | Name of the Weights & Biases project to log runs to. Only applicable when `use_wandb: true`. |
| `num_epochs` | Number of training epochs over the collected activations. |
| `sae_batch_size` | Batch size used during SAE training over the collected activations. |
| `lr` | Learning rate for the SAE optimizer. |
| `weight_decay` | L2 weight decay regularization coefficient applied during optimization. |
| `lambda_l1` | Coefficient for the L1 sparsity penalty. Only applicable when `sae_type: relu`. |
| `lambda_l1_warmup_epochs` | Number of epochs over which the L1 penalty is linearly warmed up from zero. Only applicable when `sae_type: relu`. |
| `resample_interval_epochs` | Frequency (in epochs) at which dead neuron resampling is performed. Set to `0` to disable. Only applicable when `sae_type: topk`. |
| `lambda_kl` | Coefficient for the KL-divergence loss term, which encourages the SAE's activation distribution to match a target firing rate. Only applicable when `sae_type: relu`. |
| `target_firing_rate` | Target average firing rate per feature used in the KL-divergence loss. Only applicable when `sae_type: relu`. |
| `mass_frac_threshold` | Fraction of expected activation mass below which a neuron is considered dead and eligible for resampling. Only applicable when `sae_type: topk`. |

## sae_semantic_eval

The `sae_semantic_eval` module provides tools for qualitatively evaluating what an SAE has learned by tracing which features activate during LLM generation and measuring how semantically specific those features are.

### FeatureTracer

`FeatureTracer` is the central class. Given an LLM, tokenizer, and trained SAE, it hooks into a target layer during generation and records which SAE features fire at each token position. Tracing is configured via `TraceConfig`:

| Parameter | Description |
|---|---|
| `layer_idx` | Index of the LLM layer to hook. |
| `topk_per_token` | Number of top SAE features to record per token. |
| `min_activation` | Minimum activation value to record. |
| `max_new_tokens` | Number of tokens to generate per prompt. |
| `token_mode` | Which tokens to trace: `"all"`, `"generated_only"`, or `"prompt_only"`. |
| `min_prompts` | Minimum number of distinct prompts a feature must appear in to be scored. |

Prompts can be traced one at a time (`trace_prompt`), in a list (`trace_prompts`), or streamed from a HuggingFace `IterableDataset` (`trace_prompts_from_iterable_dataset`). Results are accessible as a flat DataFrame via `to_dataframe()`.

### Feature Specificity Scoring

After tracing, `feature_specificity_scores()` scores each feature by embedding its top activating context windows using a sentence-transformer model (`all-MiniLM-L6-v2` by default) and computing the mean pairwise cosine similarity across those embeddings. This yields two scores per feature:

- **`context_specificity`** — how consistently similar the surrounding context windows are
- **`token_specificity`** — how consistently similar the activating tokens themselves are

Both are reported relative to a random-pair baseline (`context_specificity_vs_baseline`, `token_specificity_vs_baseline`). A **`composite_score`** combines both axes via their harmonic mean, scaled by mean activation. **`firing_rate`** normalizes hit count over the total number of traced token positions.

### Visualization

`feature_tracer_eval_utils` provides a set of Jupyter-friendly plotting functions:

| Function | Description |
|---|---|
| `render_feature_card` | HTML card showing a feature's top activating contexts with a token highlight, activation distribution, and relative token position profile. |
| `plot_feature_specificity_scatter` | Interactive Plotly scatter of all features on context vs. token specificity axes, with quadrant labels and configurable size/color encoding. |
| `plot_feature_umap` | UMAP projection of feature context embeddings at the token level and centroid level. |
| `plot_feature_similarity_violins` | Violin plots of pairwise cosine similarity distributions for the top/bottom N features. |
| `plot_inter_feature_similarity` | Heatmap of cosine similarity between feature centroids to identify redundant features. |

### Prompt vs. Response Comparison

When the same prompts are traced twice — once with `token_mode="prompt_only"` and once with `token_mode="generated_only"` — the following functions support cross-tracer analysis:

| Function | Description |
|---|---|
| `plot_prompt_vs_response_features` | Interactive scatter comparing shared feature hit counts in prompt vs. response mode, colored by embedding similarity between the two tracers. |
| `render_feature_card_comparison` | Side-by-side HTML card showing how a single feature behaves in each tracer, with activation distributions, relative position profiles, and mean-embedding cosine similarity. |
| `plot_prompt_response_activation_scatter` | Per-prompt scatter of a single feature's mean activation in prompt tokens vs. response tokens, with a diagonal preservation reference line. |
| `render_prompt_response_token_card` | Full token sequence display for a single prompt, with blue/orange region backgrounds for prompt vs. response tokens and red highlights for feature activations. |

### Export

`export_feature_docs(tracer, docs_dir="docs")` writes two files for use with static HTML dashboards (e.g. a GitHub Pages wiki):

- **`feature_scores.csv`** — one row per feature with all specificity scores
- **`feature_cards.json`** — per-feature card data (top activating contexts, activation histogram values, relative position profile)

## Setup instructions

### 1. Python

This project targets **Python 3.12** (see `requires-python` in `pyproject.toml`). With [pyenv](https://github.com/pyenv/pyenv):

```bash
pyenv install 3.12.13   # once
cd /path/to/sae_trainer
pyenv local 3.12.13
```

Ensure your shell loads pyenv (for example `eval "$(pyenv init -)"` in `~/.zshrc`), then open a new terminal so `python3 --version` shows 3.12.x.

### 2. Install uv

[uv](https://docs.astral.sh/uv/) is the package and environment manager for this repo. Install it if `uv` is not on your `PATH`:

- **Homebrew (macOS):** `brew install uv`
- **Official installer:** see [Installing uv](https://docs.astral.sh/uv/getting-started/installation/).

### 3. Create the environment and install dependencies

From the repository root:

```bash
uv lock    # resolve deps and write uv.lock; commit uv.lock for reproducible installs
uv sync    # creates .venv and installs the default dependency group
```

Dependency groups (defined in `pyproject.toml`):

| Group   | Contents |
|---------|----------|
| `subset` | All pinned runtime libraries **except** `umap-learn` (lighter install). |
| `full`   | `subset` plus **`umap-learn`** (UMAP plots in notebooks / eval helpers). |

By default, **`uv sync` installs `full`**. Examples:

```bash
uv sync                              # default: full (subset + umap-learn)
uv sync --group subset               # subset only
uv sync --no-default-groups --group subset   # CI-style: no default groups, subset only
```

The project is installed in **editable** mode into `.venv`, so `import sae_trainer` works after `uv sync`.

### 4. Optional: pip and `requirements.txt`

You can still use `pip install -r requirements.txt` in a manually created virtual environment, but **uv + `pyproject.toml` + `uv.lock`** is the supported workflow. The top of `requirements.txt` summarizes the equivalent `uv` commands.

### 5. Jupyter

If you use JupyterLab from the `subset` / `full` groups, register this environment as a kernel (after `uv sync`):

```bash
.venv/bin/python -m ipykernel install --user --name sae-trainer --display-name "Python (sae-trainer)"
```

Then pick **Python (sae-trainer)** in the notebook kernel picker.
