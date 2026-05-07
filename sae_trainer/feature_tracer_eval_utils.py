import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import html as html_lib
from IPython.display import HTML, display as ipy_display

# Optional: UMAP plots (install `umap-learn`, e.g. `uv sync --group full`)
try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

from .feature_tracer_utils import FeatureTracer

def top_n_feats_by_hits_count(tracer: FeatureTracer, top_n=30):
    counts = (tracer.to_dataframe().groupby("feature_id").size().sort_values(ascending=False).head(top_n))
    plt.figure(figsize=(10,4))
    counts.plot(kind="bar")
    plt.title(f"Top {top_n} Features by Hit Count")
    plt.xlabel("feature_id")
    plt.ylabel("hits")
    plt.tight_layout()
    plt.show()

def top_n_feats_by_act_mass(tracer: FeatureTracer, top_n=30):
    # 2) Top features by activation mass
    mass = (tracer.to_dataframe().groupby("feature_id")["activation"].sum().sort_values(ascending=False).head(top_n))
    plt.figure(figsize=(10,4))
    mass.plot(kind="bar", color="tab:orange")
    plt.title(f"Top {top_n} Features by Activation Mass")
    plt.xlabel("feature_id")
    plt.ylabel("sum activation")
    plt.tight_layout()
    plt.show()

def prompt_x_feature_heatmap(tracer: FeatureTracer):
    # 3) Prompt x feature heatmap (counts)
    df = tracer.to_dataframe()
    top_features = df["feature_id"].value_counts().head(25).index
    heat = (df[df["feature_id"].isin(top_features)]
            .pivot_table(index="prompt_id", columns="feature_id", values="activation", aggfunc="count", fill_value=0))
    plt.figure(figsize=(12,4))
    sns.heatmap(heat, cmap="viridis")
    plt.title("Feature Hit Counts per Prompt")
    plt.xlabel("feature_id")
    plt.ylabel("prompt_id")
    plt.tight_layout()
    plt.show()

def feature_token_position_profile(tracer: FeatureTracer, feature_id=None):
    df = tracer.to_dataframe()    
    if not feature_id:
        # If no feature_id is specified, just pick the most frequent feature
        feature_id = int(df["feature_id"].value_counts().index[0])
    sub = df[df["feature_id"] == feature_id]
    pos_counts = sub["token_pos"].value_counts().sort_index()

    plt.figure(figsize=(10,3))
    plt.plot(pos_counts.index, pos_counts.values, marker="o")
    plt.title(f"Token Position Hit Profile (feature {feature_id})")
    plt.xlabel("token_pos")
    plt.ylabel("hits")
    plt.tight_layout()
    plt.show()

def top_m_feature_coactivation(tracer: FeatureTracer, top_m=20):
    df = tracer.to_dataframe()
    top_feats = df["feature_id"].value_counts().head(top_m).index.tolist()

    pairs = df[df["feature_id"].isin(top_feats)][["prompt_id","token_pos","feature_id"]].drop_duplicates()
    mat = pd.crosstab(index=[pairs["prompt_id"], pairs["token_pos"]], columns=pairs["feature_id"]).astype(float)
    co = mat.T @ mat  # co-occurrence counts
    np.fill_diagonal(co.values, 0)

    plt.figure(figsize=(8,6))
    sns.heatmap(co, cmap="magma")
    plt.title(f"Feature Co-activation (top {top_m})")
    plt.xlabel("feature_id")
    plt.ylabel("feature_id")
    plt.tight_layout()
    plt.show()

def render_feature_card(
    feature_id: int,
    tracer,
    top_n: int = 10,
    window: int = 8,
    tokens=None,       # e.g. ["Miles"] — filter to rows where this token was generated
    token_pos=None,    # e.g. [4, 5, 6] — filter to specific sequence positions
):
    # --- Usage examples ---

    # Render cards for top 5 features across all tokens (no filter)
    # top = tracer.top_features(5)
    # for fid in top["feature_id"]:
    #     render_feature_card(int(fid), tracer)

    # Explore which positions "Miles" appears at
    # tracer.token_positions(["Miles"])

    # Top features active only when "Miles" is generated
    # tracer.top_features(20, tokens=["Miles"])

    # Render cards scoped to "Miles" token
    # top = tracer.top_features(5, tokens=["Miles"])
    # for fid in top["feature_id"]:
    #     render_feature_card(int(fid), tracer, tokens=["Miles"])

    # Render cards scoped to a position range (e.g. answer tokens only)
    # top = tracer.top_features(5, token_pos=list(range(8, 16)))
    # for fid in top["feature_id"]:
    #     render_feature_card(int(fid), tracer, token_pos=list(range(8, 16)))
    df = tracer.to_dataframe()
    df = tracer._apply_token_filter(df, tokens, token_pos)
    sub = df[df["feature_id"] == feature_id].copy()

    if sub.empty:
        scope = f"tokens={tokens}" if tokens else f"token_pos={token_pos}" if token_pos else "all tokens"
        print(f"Feature {feature_id} not found for {scope}.")
        return

    hits = len(sub)
    mean_act = sub["activation"].mean()
    max_act = sub["activation"].max()
    min_act = sub["activation"].min()

    scope_label = ""
    if tokens:
        scope_label = f" | tokens={tokens}"
    elif token_pos:
        scope_label = f" | pos={token_pos}"

    # Reconstruct token-split contexts for precise highlighting
    top_rows = (
        sub.sort_values("activation", ascending=False)
        .drop_duplicates(subset="prompt_id", keep="first")
        .head(top_n)
    )
    contexts = []
    for _, r in top_rows.iterrows():
        ids = tracer.tokenizer(r["generated_text"], return_tensors="pt")["input_ids"][0]
        toks = tracer.tokenizer.convert_ids_to_tokens(ids)
        pos = int(r["token_pos"])
        lo = max(0, pos - window)
        hi = min(len(toks), pos + window + 1)

        pre = tracer.tokenizer.convert_tokens_to_string(toks[lo:pos])
        mid = tracer.tokenizer.convert_tokens_to_string([toks[pos]]) if pos < len(toks) else ""
        post = tracer.tokenizer.convert_tokens_to_string(toks[pos + 1:hi])

        contexts.append({
            "prompt_id": r["prompt_id"],
            "token_pos": pos,
            "activation": r["activation"],
            "pre": pre,
            "mid": mid,
            "post": post,
        })

    # --- HTML card ---
    html_parts = [f"""
    <div style="border:2px solid #4a90d9;border-radius:8px;padding:16px;margin:12px 0;
                font-family:monospace;background:#f8f9fa;max-width:900px;">
      <h3 style="margin:0 0 8px 0;color:#1a1a2e;">Feature #{feature_id}<span style="font-size:0.7em;color:#888;">{scope_label}</span></h3>
      <div style="display:flex;gap:24px;margin-bottom:10px;font-size:0.9em;color:#555;">
        <span><b>Hits:</b> {hits}</span>
        <span><b>Mean act:</b> {mean_act:.3f}</span>
        <span><b>Max act:</b> {max_act:.3f}</span>
        <span><b>Prompts:</b> {sub['prompt_id'].nunique()}</span>
      </div>
      <hr style="margin:8px 0;border-color:#ddd;">
      <b>Top Activating Contexts</b><br><br>
    """]

    for ctx in contexts:
        act = ctx["activation"]
        opacity = 0.15 + 0.75 * (act - min_act) / (max_act - min_act + 1e-8)
        pre = html_lib.escape(ctx["pre"])
        mid = html_lib.escape(ctx["mid"])
        post = html_lib.escape(ctx["post"])
        html_parts.append(f"""
        <div style="margin:5px 0;padding:6px 10px;background:white;
                    border-left:4px solid rgba(220,50,50,{opacity:.2f});
                    border-radius:0 4px 4px 0;font-size:0.85em;line-height:1.5;">
          <span style="color:#999;font-size:0.78em;">[pos={ctx['token_pos']}, act={act:.3f}]</span>
          <span style="margin-left:8px;">...{pre}<span style="background:rgba(220,50,50,{opacity:.2f});
            font-weight:bold;padding:1px 3px;border-radius:2px;">{mid}</span>{post}...</span>
        </div>""")

    html_parts.append("</div>")
    ipy_display(HTML("".join(html_parts)))

    # --- Plots: activation distribution + token position profile ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 2.5))

    axes[0].hist(sub["activation"], bins=20, color="#4a90d9", edgecolor="white")
    axes[0].set_title("Activation Distribution")
    axes[0].set_xlabel("activation")
    axes[0].set_ylabel("count")

    pos_counts = sub["token_pos"].value_counts().sort_index()
    axes[1].plot(pos_counts.index, pos_counts.values, marker="o", color="#e67e22")
    axes[1].set_title("Token Position Profile")
    axes[1].set_xlabel("token_pos")
    axes[1].set_ylabel("hits")

    fig.suptitle(f"Feature #{feature_id}{scope_label}", fontsize=10, color="#888")
    plt.tight_layout()
    plt.show()

def plot_feature_umap(
    tracer,
    top_n: int = 20,           # restrict to top-N features by specificity to keep plot readable
    random_state: int = 42,
    figsize=(10, 7),
    include_token_level_plot=False,
    specificity_type: str = 'context'
):
    """
    Two separate UMAP figures (shown one after the other), if include_token_level_plot is True:

    1) Token-level: each point is one context window embedding, colored by feature_id.
       Shows whether each feature's activating contexts cluster tightly. Controlled by include_token_level_plot

    2) Centroid-level: each point is one feature (mean embedding).
       Point size = hit count, color = f'{specificity_type}_specificity_vs_baseline' (if scores provided).
       Shows the broader feature landscape and how features relate to each other.

    figsize applies to each figure independently.
    """
    if not HAS_UMAP:
        print(
            "UMAP is not installed. Use the `full` dependency group (includes umap-learn), "
            "e.g. `uv sync --group full`, or run `pip install umap-learn`."
        )
        return

    # --- Usage ---
    # scores = tracer.feature_specificity_scores(feature_embeddings=embeddings)
    # plot_feature_umap(embeddings, specificity_scores=scores, top_n=20)
    if specificity_type =='context':
        specificity_scores = tracer.get_feature_context_specificity_scores()
    elif specificity_type == 'token':
        specificity_scores = tracer.get_feature_token_specificity_scores()
    else:
        raise ValueError(f"specificity_type must be either 'context' or 'token', received {specificity_type}")
    feature_embeddings = tracer.get_feature_embeddings()


    # Select features to plot
    if specificity_scores is not None and not specificity_scores.empty:
        top_fids = specificity_scores.head(top_n)["feature_id"].tolist()
    else:
        # Fall back to top-N by hit count
        top_fids = sorted(
            feature_embeddings.keys(),
            key=lambda fid: len(feature_embeddings[fid]["activations"]),
            reverse=True,
        )[:top_n]

    selected = {fid: feature_embeddings[fid] for fid in top_fids if fid in feature_embeddings}
    if not selected:
        print("No features to plot.")
        return

    # --- Build token-level arrays ---
    all_embs, all_labels = [], []
    for fid, data in selected.items():
        all_embs.append(data[f"{specificity_type}_embeddings"])
        all_labels.extend([fid] * len(data[f"{specificity_type}_embeddings"]))

    all_embs = np.concatenate(all_embs, axis=0)
    all_labels = np.array(all_labels)

    # --- Fit a single UMAP on all token embeddings ---
    reducer = umap.UMAP(n_components=2, random_state=random_state, metric="cosine")
    token_2d = reducer.fit_transform(all_embs)

    # Compute centroids in 2D for the centroid plot
    centroids_2d = np.array([
        token_2d[all_labels == fid].mean(axis=0) for fid in top_fids
    ])

    # Color palette — one color per feature
    palette = plt.get_cmap("tab20", len(top_fids))
    color_map = {fid: palette(i) for i, fid in enumerate(top_fids)}
    colors = [color_map[fid] for fid in all_labels]

    hit_counts = np.array([len(selected[fid]["activations"]) for fid in top_fids])
    marker_sizes = 50 + 200 * (hit_counts / hit_counts.max())  # scale dot size by hits

    if include_token_level_plot:
        # --- Figure 1: token-level ---
        fig1, ax1 = plt.subplots(figsize=figsize)
        ax1.scatter(token_2d[:, 0], token_2d[:, 1], c=colors, s=18, alpha=0.7, linewidths=0)

        # Clip axis limits to 1st–99th percentile to remove outlier-driven whitespace
        x_lo, x_hi = np.percentile(token_2d[:, 0], [1, 99])
        y_lo, y_hi = np.percentile(token_2d[:, 1], [1, 99])
        margin = 0.08
        ax1.set_xlim(x_lo - margin * (x_hi - x_lo), x_hi + margin * (x_hi - x_lo))
        ax1.set_ylim(y_lo - margin * (y_hi - y_lo), y_hi + margin * (y_hi - y_lo))

        # Legend — one patch per feature
        legend_handles = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map[fid],
                       markersize=7, label=str(fid))
            for fid in top_fids
        ]
        ax1.legend(handles=legend_handles, title="feature_id", fontsize=7,
                   title_fontsize=7, loc="best", framealpha=0.7,
                   ncol=max(1, len(top_fids) // 10))

        ax1.set_title(f"Token-level embeddings (top {len(top_fids)} features)", fontsize=11)
        ax1.set_xlabel("UMAP-1")
        ax1.set_ylabel("UMAP-2")
        ax1.axis("off")
        fig1.suptitle("Feature Embedding Landscape — token level", fontsize=13, y=1.02)
        fig1.tight_layout()
        plt.show()

    # --- Figure 2: centroid-level ---
    fig2, ax2 = plt.subplots(figsize=figsize)
    if specificity_scores is not None and f"{specificity_type}_specificity_vs_baseline" in specificity_scores.columns:
        score_map = specificity_scores.set_index("feature_id")[f"{specificity_type}_specificity_vs_baseline"]
        spec_values = np.array([score_map.get(fid, 0.0) for fid in top_fids])
        full_col = specificity_scores[f"{specificity_type}_specificity_vs_baseline"].dropna()
        sc = ax2.scatter(
            centroids_2d[:, 0], centroids_2d[:, 1],
            s=marker_sizes, c=spec_values, cmap="RdYlGn",
            vmin=full_col.min(), vmax=full_col.max(),
            alpha=0.85, linewidths=0.5, edgecolors="grey",
        )
        fig2.colorbar(sc, ax=ax2, label=f"{specificity_type}_specificity_vs_baseline")
    else:
        centroid_colors = [color_map[fid] for fid in top_fids]
        ax2.scatter(
            centroids_2d[:, 0], centroids_2d[:, 1],
            s=marker_sizes, c=centroid_colors, alpha=0.85,
            linewidths=0.5, edgecolors="grey",
        )

    for i, fid in enumerate(top_fids):
        ax2.annotate(str(fid), centroids_2d[i], fontsize=7, ha="center", va="bottom")

    ax2.set_title(f"Feature centroids (size = hits, color = specificity)", fontsize=11)
    ax2.set_xlabel("UMAP-1")
    ax2.set_ylabel("UMAP-2")
    ax2.axis("off")
    fig2.suptitle("Feature Embedding Landscape — centroids", fontsize=13, y=1.02)
    fig2.tight_layout()
    plt.show()


def plot_feature_similarity_violins(
    tracer: FeatureTracer,
    top_n: int | None = 20,
    bottom_n: int | None = None,
    mask_same_context: bool = True,
    figsize: tuple = (14, 5),
    specificity_type: str = 'context'
):
    """
    Violin plot of pairwise cosine similarities for each feature's context embeddings.

    Each violin shows the distribution of all cross-prompt pairwise similarities for
    one feature. Features are ordered left-to-right by specificity_vs_baseline (highest
    first). A dashed baseline shows the mean random-pair similarity across all features.
    """
    if top_n is not None and bottom_n is not None:
        raise ValueError('Must provide either top_n or bottom_n, not both')
    
    if specificity_type =='context':
        scores = tracer.get_feature_context_specificity_scores()
    elif specificity_type == 'token':
        scores = tracer.get_feature_token_specificity_scores()
    else:
        raise ValueError(f"specificity_type must be either 'context' or 'token', received {specificity_type}")
    feature_embeddings = tracer.get_feature_embeddings()

    if scores is None or scores.empty or feature_embeddings is None:
        print("No specificity scores available. Run tracer.feature_specificity_scores() first.")
        return

    ranked = scores.dropna(subset=[f"{specificity_type}_specificity_vs_baseline"])
    if bottom_n is not None:
        selected_fids = ranked.tail(bottom_n)["feature_id"].tolist()
    else:
        selected_fids = ranked.head(top_n)["feature_id"].tolist()
    top_fids = selected_fids

    # Build per-feature pairwise similarity distributions
    records = []
    for fid in top_fids:
        data = feature_embeddings.get(fid)
        if data is None:
            continue
        emb = data[f"{specificity_type}_embeddings"]           # [n, d], unit-normed
        pids = np.array(data["prompt_ids"]) if "prompt_ids" in data else None
        n = len(emb)
        if n < 2:
            continue

        sim_matrix = emb @ emb.T           # cosine sim = dot product (unit-normed)

        # Exclude diagonal; optionally exclude same-prompt pairs
        include = ~np.eye(n, dtype=bool)
        if mask_same_context and pids is not None:
            include &= (pids[:, None] != pids[None, :])

        for s in sim_matrix[include]:
            records.append({"feature_id": str(fid), "cosine_similarity": float(s)})

    if not records:
        print("No pairwise similarities to plot.")
        return

    plot_df = pd.DataFrame(records)
    ordered_labels = [str(fid) for fid in top_fids if str(fid) in plot_df["feature_id"].values]

    # Baseline: mean random-pair similarity across all features
    all_emb = np.concatenate([feature_embeddings[fid][f"{specificity_type}_embeddings"] for fid in feature_embeddings])
    rng = np.random.default_rng(42)
    idx = rng.choice(len(all_emb), size=(min(1000, len(all_emb)), 2))
    baseline = float((all_emb[idx[:, 0]] * all_emb[idx[:, 1]]).sum(axis=1).mean())

    fig, ax = plt.subplots(figsize=figsize)
    sns.violinplot(
        data=plot_df,
        x="feature_id", y="cosine_similarity",
        order=ordered_labels,
        cut=0,        # don't extend violins beyond the data range
        inner="box",  # show IQR box inside each violin
        linewidth=0.8,
        ax=ax,
    )
    ax.axhline(baseline, color="red", linestyle="--", linewidth=1.2, label=f"baseline ({baseline:.3f})")
    level = specificity_type.capitalize()
    if bottom_n is not None:
        xlabel = "feature_id  (ordered by specificity_vs_baseline, low → high)"
        title  = f"{level} similarity distributions — bottom {len(ordered_labels)} features"
    else:
        xlabel = "feature_id  (ordered by specificity_vs_baseline, high → low)"
        title  = f"{level} similarity distributions — top {len(ordered_labels)} features"
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("pairwise cosine similarity", fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.tick_params(axis="x", labelsize=7, rotation=45)
    ax.legend(fontsize=8)
    fig.tight_layout()
    plt.show()


def compute_inter_feature_similarity(feature_embeddings: dict, specificity_type: str = 'context') -> tuple[np.ndarray, list]:
    """
    Compute pairwise cosine similarity between feature centroids.

    Each feature's centroid is the mean of its context window embeddings
    (already unit-normed, so mean then re-normalize gives the centroid direction).

    Returns:
        sim_matrix: np.ndarray [n_features, n_features] of cosine similarities
        feature_ids: list of feature IDs corresponding to matrix rows/cols
    """
    feature_ids = list(feature_embeddings.keys())

    centroids = []
    for fid in feature_ids:
        emb = feature_embeddings[fid][f"{specificity_type}_embeddings"]  # [n, d], unit-normed
        centroid = emb.mean(axis=0)
        # Re-normalize so centroid is also unit-length for valid cosine sim
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        centroids.append(centroid)

    centroids = np.stack(centroids)           # [n_features, d]
    sim_matrix = centroids @ centroids.T      # [n_features, n_features]

    return sim_matrix, feature_ids


def plot_inter_feature_similarity(
    tracer,
    top_n: int = 30,            # restrict to top-N features by specificity
    figsize=(12, 10),
    annot_threshold: float = 0.5,  # annotate cells above this similarity value
    specificity_type: str = 'context'
):
    """
    Heatmap of pairwise cosine similarity between feature centroids.

    Features are ordered by specificity_vs_baseline (if provided) so that the
    most interpretable features appear top-left.  Off-diagonal cells with high
    similarity indicate potentially redundant features.
    """

    # --- Usage ---
    # scores = tracer.feature_specificity_scores()
    # embeddings = tracer.get_feature_embeddings()
    #
    # sim_matrix, fids = plot_inter_feature_similarity(
    #     embeddings,
    #     specificity_scores=scores,
    #     top_n=30,
    #     annot_threshold=0.5,  
    # )
    # Select and order features

    if specificity_type =='context':
        specificity_scores = tracer.get_feature_context_specificity_scores()
    elif specificity_type == 'token':
        specificity_scores = tracer.get_feature_token_specificity_scores()
    else:
        raise ValueError(f"specificity_type must be either 'context' or 'token', received {specificity_type}")
    feature_embeddings = tracer.get_feature_embeddings()


    if specificity_scores is not None and not specificity_scores.empty:
        ordered_fids = specificity_scores.head(top_n)["feature_id"].tolist()
        ordered_fids = [fid for fid in ordered_fids if fid in feature_embeddings]
    else:
        ordered_fids = sorted(
            feature_embeddings.keys(),
            key=lambda fid: len(feature_embeddings[fid]["activations"]),
            reverse=True,
        )[:top_n]

    subset = {fid: feature_embeddings[fid] for fid in ordered_fids}
    sim_matrix, feature_ids = compute_inter_feature_similarity(subset, specificity_type)

    # Build labels — feature_id plus hit count for context
    labels = [
        f"{fid}\n(n={len(feature_embeddings[fid]['activations'])})"
        for fid in feature_ids
    ]

    sim_df = pd.DataFrame(sim_matrix, index=labels, columns=labels)

    # Annotation mask — only annotate high-similarity off-diagonal cells
    annot = np.where(
        (sim_matrix >= annot_threshold) & ~np.eye(len(feature_ids), dtype=bool),
        np.round(sim_matrix, 2).astype(str),
        "",
    )

    _, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        sim_df,
        ax=ax,
        cmap="RdYlGn",
        vmin=-1, vmax=1,
        center=0,
        annot=annot,
        fmt="",
        linewidths=0.3,
        linecolor="#ddd",
        square=True,
        cbar_kws={"label": "cosine similarity", "shrink": 0.7},
    )
    ax.set_title(
        f"Inter-feature centroid similarity (top {len(feature_ids)} by specificity)",
        fontsize=12, pad=12,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)
    plt.tight_layout()
    plt.show()

    # Print pairs above threshold for easy inspection
    redundant_pairs = []
    for i in range(len(feature_ids)):
        for j in range(i + 1, len(feature_ids)):
            if sim_matrix[i, j] >= annot_threshold:
                redundant_pairs.append({
                    "feature_a": feature_ids[i],
                    "feature_b": feature_ids[j],
                    "cosine_sim": round(float(sim_matrix[i, j]), 4),
                })

    if redundant_pairs:
        print(f"\nPotentially redundant pairs (cosine_sim >= {annot_threshold}):")
        ipy_display(pd.DataFrame(redundant_pairs).sort_values("cosine_sim", ascending=False))
    else:
        print(f"\nNo pairs with cosine_sim >= {annot_threshold} — features appear well-separated.")

    return sim_matrix, feature_ids


def plot_feature_specificity_scatter(
    tracer: FeatureTracer,
    top_n: int | None = None,
    figsize: tuple = (9, 7),
    annotate_n: int = 10,
):
    """
    Scatterplot with context_specificity_vs_baseline on the x-axis and
    token_specificity_vs_baseline on the y-axis.

    Each point is one feature.  Dashed lines at x=0 / y=0 divide the plot
    into four interpretable quadrants:

        High context + High token  → concept-specific features
        Low  context + High token  → token-specific, context-agnostic (e.g. "the")
        High context + Low  token  → context-specific, token-diverse
        Low  context + Low  token  → noisy or near-dead

    Point size  = hit count (larger = fires more often)
    Point color = composite_score (green = genuinely active + specific)

    Parameters
    ----------
    top_n : restrict to the top-N features by composite_score (None = all features)
    annotate_n : label the top-N features by composite_score with their feature_id
    """
    scores_df = tracer.get_feature_specificity_scores_df()
    if scores_df is None or scores_df.empty:
        print("No specificity scores available. Run tracer.feature_specificity_scores() first.")
        return

    df = scores_df.dropna(subset=["context_specificity_vs_baseline", "token_specificity_vs_baseline"]).copy()
    if df.empty:
        print("Scores DataFrame has no rows with both context and token specificity.")
        return

    if top_n is not None:
        df = df.nlargest(top_n, "composite_score")

    x = df["context_specificity_vs_baseline"].values
    y = df["token_specificity_vs_baseline"].values
    sizes = 30 + 200 * (df["hits"].values / df["hits"].values.max())
    colors = df["composite_score"].values

    fig, ax = plt.subplots(figsize=figsize)

    sc = ax.scatter(
        x, y,
        s=sizes,
        c=colors,
        cmap="RdYlGn",
        alpha=0.85,
        linewidths=0.4,
        edgecolors="grey",
    )
    fig.colorbar(sc, ax=ax, label="composite_score")

    # Quadrant dividers
    ax.axvline(0, color="#888", linestyle="--", linewidth=0.9, zorder=0)
    ax.axhline(0, color="#888", linestyle="--", linewidth=0.9, zorder=0)

    # Quadrant labels
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    pad = 0.03
    quad_kw = dict(fontsize=7.5, color="#555", ha="center", va="center", style="italic")
    ax.text(xmin + (0 - xmin) * 0.5, ymax - (ymax - 0) * pad * 2, "context-specific\ntoken-diverse",  **quad_kw)
    ax.text(xmax - (xmax - 0) * 0.5, ymax - (ymax - 0) * pad * 2, "concept-specific",               **quad_kw)
    ax.text(xmin + (0 - xmin) * 0.5, ymin + (0 - ymin) * pad * 2, "noisy / near-dead",              **quad_kw)
    ax.text(xmax - (xmax - 0) * 0.5, ymin + (0 - ymin) * pad * 2, "token-specific\ncontext-agnostic", **quad_kw)

    # Annotate top features by composite_score
    if annotate_n and annotate_n > 0:
        top_rows = df.nlargest(annotate_n, "composite_score")
        for _, row in top_rows.iterrows():
            ax.annotate(
                str(int(row["feature_id"])),
                (row["context_specificity_vs_baseline"], row["token_specificity_vs_baseline"]),
                fontsize=6.5,
                xytext=(4, 4),
                textcoords="offset points",
                color="#222",
            )

    ax.set_xlabel("context_specificity_vs_baseline", fontsize=10)
    ax.set_ylabel("token_specificity_vs_baseline", fontsize=10)
    n_label = f"top {len(df)}" if top_n is not None else f"all {len(df)}"
    ax.set_title(f"Feature specificity landscape ({n_label} features)\nsize = hits, color = composite_score", fontsize=11)
    fig.tight_layout()
    plt.show()
