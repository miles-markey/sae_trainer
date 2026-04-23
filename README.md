# sae_trainer

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
