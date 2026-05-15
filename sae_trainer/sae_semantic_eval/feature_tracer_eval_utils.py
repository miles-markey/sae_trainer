import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import html as html_lib
from IPython.display import HTML, display as ipy_display
import plotly.graph_objects as go
import plotly.express as px

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

    rel_pos = (sub["token_pos_relative"] / sub["num_tokens_relative"]).round(2)
    rel_pos_counts = rel_pos.value_counts().sort_index()
    axes[1].plot(rel_pos_counts.index, rel_pos_counts.values, marker="o", color="#e67e22")
    axes[1].set_title("Relative Token Position Profile")
    axes[1].set_xlabel("relative position (token_pos / num_tokens)")
    axes[1].set_ylabel("hits")
    axes[1].set_xlim(0, 1)

    fig.suptitle(f"Feature #{feature_id}{scope_label}", fontsize=10, color="#888")
    plt.tight_layout()
    plt.show()

def render_feature_card_comparison(
    feature_id: int,
    prompt_tracer: FeatureTracer,
    response_tracer: FeatureTracer,
    top_n: int = 8,
    window: int = 8,
):
    """
    Side-by-side feature card comparing how a feature behaves in prompt_tracer vs response_tracer.

    Shows per-tracer stats, top activating contexts, activation distributions, and
    relative token position profiles. If compute_feature_embeddings() has been called on
    both tracers, also shows the cosine similarity between the mean context embeddings.
    """

    def _extract_sub(tracer):
        df = tracer.to_dataframe()
        sub = df[df["feature_id"] == feature_id].copy()
        return sub if not sub.empty else None

    def _extract_contexts(sub, tracer):
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
            lo, hi = max(0, pos - window), min(len(toks), pos + window + 1)
            contexts.append({
                "token_pos": pos,
                "activation": r["activation"],
                "pre":  tracer.tokenizer.convert_tokens_to_string(toks[lo:pos]),
                "mid":  tracer.tokenizer.convert_tokens_to_string([toks[pos]]) if pos < len(toks) else "",
                "post": tracer.tokenizer.convert_tokens_to_string(toks[pos + 1:hi]),
            })
        return contexts

    def _column_html(label, color, sub, contexts):
        if sub is None:
            return f"""
            <div style="flex:1;padding:0 12px;">
              <h4 style="margin:0 0 8px 0;color:{color};">{label}</h4>
              <p style="color:#aaa;font-size:0.85em;">Feature not found in this tracer.</p>
            </div>"""

        hits = len(sub)
        mean_act = sub["activation"].mean()
        max_act = sub["activation"].max()
        min_act = sub["activation"].min()

        ctx_html = ""
        for ctx in contexts:
            act = ctx["activation"]
            opacity = 0.15 + 0.75 * (act - min_act) / (max_act - min_act + 1e-8)
            pre  = html_lib.escape(ctx["pre"])
            mid  = html_lib.escape(ctx["mid"])
            post = html_lib.escape(ctx["post"])
            ctx_html += f"""
            <div style="margin:4px 0;padding:5px 8px;background:white;
                        border-left:4px solid rgba(220,50,50,{opacity:.2f});
                        border-radius:0 4px 4px 0;font-size:0.82em;line-height:1.5;">
              <span style="color:#999;font-size:0.78em;">[pos={ctx['token_pos']}, act={act:.3f}]</span>
              <span style="margin-left:6px;">...{pre}<span style="background:rgba(220,50,50,{opacity:.2f});
                font-weight:bold;padding:1px 3px;border-radius:2px;">{mid}</span>{post}...</span>
            </div>"""

        return f"""
        <div style="flex:1;padding:0 12px;min-width:0;">
          <h4 style="margin:0 0 6px 0;color:{color};">{label}</h4>
          <div style="display:flex;gap:16px;margin-bottom:8px;font-size:0.85em;color:#555;">
            <span><b>Hits:</b> {hits}</span>
            <span><b>Mean act:</b> {mean_act:.3f}</span>
            <span><b>Max act:</b> {max_act:.3f}</span>
            <span><b>Prompts:</b> {sub['prompt_id'].nunique()}</span>
          </div>
          <div style="font-size:0.82em;font-weight:bold;margin-bottom:4px;color:#444;">
            Top Activating Contexts
          </div>
          {ctx_html}
        </div>"""

    p_sub = _extract_sub(prompt_tracer)
    r_sub = _extract_sub(response_tracer)
    p_ctx = _extract_contexts(p_sub, prompt_tracer) if p_sub is not None else []
    r_ctx = _extract_contexts(r_sub, response_tracer) if r_sub is not None else []

    # Embedding similarity (if available)
    emb_sim_html = ""
    p_embs = prompt_tracer.get_feature_embeddings()
    r_embs = response_tracer.get_feature_embeddings()
    if p_embs and r_embs and feature_id in p_embs and feature_id in r_embs:
        p_mean = p_embs[feature_id]["context_embeddings"].mean(axis=0)
        p_mean /= np.linalg.norm(p_mean) + 1e-8
        r_mean = r_embs[feature_id]["context_embeddings"].mean(axis=0)
        r_mean /= np.linalg.norm(r_mean) + 1e-8
        sim = float(p_mean @ r_mean)
        sim_color = "#27ae60" if sim >= 0.7 else "#e67e22" if sim >= 0.4 else "#c0392b"
        emb_sim_html = (
            f'<span style="font-size:0.85em;color:{sim_color};margin-left:16px;">'
            f'embedding similarity: <b>{sim:.3f}</b></span>'
        )

    html = f"""
    <div style="border:2px solid #999;border-radius:8px;padding:16px;margin:12px 0;
                font-family:monospace;background:#f8f9fa;max-width:1200px;">
      <div style="display:flex;align-items:baseline;margin-bottom:12px;">
        <h3 style="margin:0;color:#1a1a2e;">Feature #{feature_id}</h3>
        {emb_sim_html}
      </div>
      <hr style="margin:0 0 12px 0;border-color:#ddd;">
      <div style="display:flex;gap:0;">
        {_column_html("Prompt mode", "#4a90d9", p_sub, p_ctx)}
        <div style="width:1px;background:#ddd;flex-shrink:0;"></div>
        {_column_html("Response mode", "#e67e22", r_sub, r_ctx)}
      </div>
    </div>"""
    ipy_display(HTML(html))

    # --- Side-by-side plots ---
    has_p = p_sub is not None
    has_r = r_sub is not None
    if not has_p and not has_r:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 4))

    def _fill_col(ax_dist, ax_pos, sub, color, label):
        if sub is None:
            ax_dist.text(0.5, 0.5, "not found", ha="center", va="center",
                         transform=ax_dist.transAxes, color="#aaa")
            ax_pos.text(0.5, 0.5, "not found", ha="center", va="center",
                        transform=ax_pos.transAxes, color="#aaa")
            return
        ax_dist.hist(sub["activation"], bins=20, color=color, edgecolor="white")
        ax_dist.set_title(f"{label} — Activation Distribution", fontsize=9)
        ax_dist.set_xlabel("activation")
        ax_dist.set_ylabel("count")

        rel_pos = (sub["token_pos_relative"] / sub["num_tokens_relative"]).round(2)
        counts = rel_pos.value_counts().sort_index()
        ax_pos.plot(counts.index, counts.values, marker="o", color=color)
        ax_pos.set_title(f"{label} — Relative Token Position", fontsize=9)
        ax_pos.set_xlabel("relative position")
        ax_pos.set_ylabel("hits")
        ax_pos.set_xlim(0, 1)

    _fill_col(axes[0, 0], axes[1, 0], p_sub, "#4a90d9", "Prompt")
    _fill_col(axes[0, 1], axes[1, 1], r_sub, "#e67e22", "Response")

    fig.suptitle(f"Feature #{feature_id}", fontsize=10, color="#888")
    plt.tight_layout()
    plt.show()


def plot_prompt_response_activation_scatter(
    feature_id: int,
    prompt_tracer: FeatureTracer,
    response_tracer: FeatureTracer,
    figsize: tuple = (750, 700),
):
    """
    Per-prompt scatter showing whether a feature's activation is preserved from
    prompt tokens to response tokens.

    Each point is one prompt (matched by prompt_id). Points are only included if
    both tracers processed the same prompt — unmatched prompt_ids are dropped.

      x-axis = mean activation over prompt tokens (0 if feature didn't fire)
      y-axis = mean activation over response tokens (0 if feature didn't fire)

    Point color encodes firing category:
      both          (green)  : feature fired in both prompt and response portions
      prompt-only   (blue)   : feature fired in prompt but not response
      response-only (orange) : feature fired in response but not prompt

    Prompts where the feature didn't fire in either portion are excluded.
    A diagonal reference line (y = x) marks perfect activation preservation.
    """
    p_df = prompt_tracer.to_dataframe()
    r_df = response_tracer.to_dataframe()

    # Mean activation per prompt for this feature (only prompts where it fired)
    p_feat = (
        p_df[p_df["feature_id"] == feature_id]
        .groupby("prompt_id")["activation"].mean()
    )
    r_feat = (
        r_df[r_df["feature_id"] == feature_id]
        .groupby("prompt_id")["activation"].mean()
    )

    # Only compare prompts that both tracers processed
    shared_pids = sorted(
        set(p_df["prompt_id"].unique()) & set(r_df["prompt_id"].unique())
    )
    if not shared_pids:
        print("No shared prompt_ids between the two tracers.")
        return

    p_feat = p_feat.reindex(shared_pids, fill_value=0.0)
    r_feat = r_feat.reindex(shared_pids, fill_value=0.0)

    combined = pd.DataFrame({
        "prompt_id":          shared_pids,
        "prompt_activation":  p_feat.values,
        "response_activation": r_feat.values,
    })

    # Drop prompts where the feature didn't fire in either portion
    combined = combined[(combined["prompt_activation"] > 0) | (combined["response_activation"] > 0)]
    if combined.empty:
        print(f"Feature {feature_id} did not fire in any shared prompt.")
        return

    def _category(row):
        if row.prompt_activation > 0 and row.response_activation > 0:
            return "both"
        elif row.prompt_activation > 0:
            return "prompt-only"
        return "response-only"

    combined["category"] = [_category(r) for r in combined.itertuples()]

    color_map  = {"both": "#27ae60", "prompt-only": "#4a90d9", "response-only": "#e67e22"}
    traces = []
    for cat, grp in combined.groupby("category"):
        hover = [
            f"<b>prompt {row.prompt_id}</b><br>"
            f"prompt_activation: {row.prompt_activation:.4f}<br>"
            f"response_activation: {row.response_activation:.4f}"
            for row in grp.itertuples()
        ]
        traces.append(go.Scatter(
            x=grp["prompt_activation"],
            y=grp["response_activation"],
            mode="markers",
            name=cat,
            marker=dict(color=color_map[cat], size=9, opacity=0.85,
                        line=dict(width=0.5, color="grey")),
            text=hover,
            hoverinfo="text",
        ))

    # Diagonal reference line (y = x)
    ax_max = max(combined["prompt_activation"].max(), combined["response_activation"].max()) * 1.05
    diag = go.Scatter(
        x=[0, ax_max], y=[0, ax_max],
        mode="lines",
        name="y = x (perfect preservation)",
        line=dict(color="#aaa", dash="dash", width=1.2),
        hoverinfo="skip",
    )
    traces.append(diag)

    # Correlation across prompts where both fired
    both = combined[combined["category"] == "both"]
    corr_str = ""
    if len(both) >= 2:
        corr = float(np.corrcoef(both["prompt_activation"], both["response_activation"])[0, 1])
        corr_str = f" · r={corr:.3f} (prompts where both fired)"

    n_both     = (combined["category"] == "both").sum()
    n_p_only   = (combined["category"] == "prompt-only").sum()
    n_r_only   = (combined["category"] == "response-only").sum()

    layout = go.Layout(
        title=dict(
            text=(
                f"Feature #{feature_id} — prompt vs. response activation per prompt<br>"
                f"<sup>both: {n_both} · prompt-only: {n_p_only} · response-only: {n_r_only}"
                f"{corr_str}</sup>"
            ),
            font=dict(size=13),
        ),
        xaxis=dict(title="mean activation — prompt tokens", rangemode="tozero"),
        yaxis=dict(title="mean activation — response tokens", rangemode="tozero"),
        width=figsize[0],
        height=figsize[1],
        hovermode="closest",
        legend=dict(title="category"),
    )

    go.Figure(data=traces, layout=layout).show()


def render_prompt_response_token_card(
    feature_id: int,
    prompt_id: int,
    prompt_tracer: FeatureTracer,
    response_tracer: FeatureTracer,
):
    """
    Renders the full token sequence of a single prompt, highlighting where a given
    feature fired in the prompt portion (blue region) vs the response portion (orange region).

    Token background encodes region (prompt=blue, response=orange). Tokens where the
    feature fired are highlighted in red with intensity proportional to activation strength;
    hovering over a token shows its position and activation value.

    Requires both tracers to have been run on the same prompt_id.
    Uses token_pos_relative and num_tokens_relative stored in _rows to recover the
    prompt/response boundary without re-tokenizing.
    """
    prompt_id = str(prompt_id)
    
    tokenizer = prompt_tracer.tokenizer

    p_df = prompt_tracer.to_dataframe()
    r_df = response_tracer.to_dataframe()

    p_pid = p_df[p_df["prompt_id"] == prompt_id]
    r_pid = r_df[r_df["prompt_id"] == prompt_id]

    if p_pid.empty and r_pid.empty:
        print(f"prompt_id {prompt_id} not found in either tracer.")
        return

    # Recover prompt/response boundary from stored relative position fields
    if not r_pid.empty:
        row0 = r_pid.iloc[0]
        num_prompt_tokens = int(row0["token_pos"] - row0["token_pos_relative"])
    else:
        row0 = p_pid.iloc[0]
        num_prompt_tokens = int(row0["num_tokens_relative"])

    generated_text = row0["generated_text"]
    token_ids = tokenizer(generated_text, return_tensors="pt")["input_ids"][0]
    tokens = tokenizer.convert_ids_to_tokens(token_ids.tolist())

    # Activation lookup for this feature at each token position
    p_acts = (
        p_pid[p_pid["feature_id"] == feature_id]
        .set_index("token_pos")["activation"]
        .to_dict()
    )
    r_acts = (
        r_pid[r_pid["feature_id"] == feature_id]
        .set_index("token_pos")["activation"]
        .to_dict()
    )
    all_acts = {**p_acts, **r_acts}
    max_act = max(all_acts.values()) if all_acts else 1.0

    # Per-region stats
    p_hits, r_hits = len(p_acts), len(r_acts)
    p_mean = float(np.mean(list(p_acts.values()))) if p_acts else 0.0
    r_mean = float(np.mean(list(r_acts.values()))) if r_acts else 0.0

    # Build token spans
    spans = []
    for pos, tok in enumerate(tokens):
        tok_str = html_lib.escape(tokenizer.convert_tokens_to_string([tok]))
        is_prompt = pos < num_prompt_tokens
        region_bg = "#ddeeff" if is_prompt else "#ffeedd"

        act = all_acts.get(pos, 0.0)
        if act > 0:
            intensity = act / max_act
            alpha = 0.2 + 0.7 * intensity
            weight = "bold" if intensity > 0.5 else "normal"
            spans.append(
                f'<span title="pos={pos}, act={act:.3f}" '
                f'style="background:rgba(220,50,50,{alpha:.2f});font-weight:{weight};'
                f'border-radius:3px;padding:1px 2px;cursor:default;">{tok_str}</span>'
            )
        else:
            spans.append(
                f'<span style="background:{region_bg};border-radius:2px;'
                f'padding:1px 2px;">{tok_str}</span>'
            )

    # Inject [RESPONSE] boundary marker
    spans.insert(
        num_prompt_tokens,
        '<span style="display:inline-block;margin:0 5px;padding:1px 7px;'
        'background:#ccc;border-radius:4px;font-size:0.72em;color:#444;'
        'vertical-align:middle;">[RESPONSE]</span>',
    )

    html = f"""
    <div style="border:2px solid #999;border-radius:8px;padding:16px;margin:12px 0;
                font-family:monospace;background:#f8f9fa;max-width:1100px;">
      <div style="display:flex;align-items:baseline;gap:12px;margin-bottom:10px;">
        <h3 style="margin:0;color:#1a1a2e;">Feature #{feature_id}</h3>
        <span style="font-size:0.8em;color:#888;">prompt {prompt_id}</span>
      </div>
      <div style="display:flex;gap:16px;margin-bottom:12px;font-size:0.85em;">
        <div style="background:#ddeeff;border-radius:6px;padding:7px 14px;color:#555;">
          <b style="color:#4a90d9;">Prompt portion</b>&nbsp;&nbsp;
          hits: {p_hits} &nbsp;·&nbsp; mean act: {p_mean:.3f}
        </div>
        <div style="background:#ffeedd;border-radius:6px;padding:7px 14px;color:#555;">
          <b style="color:#e67e22;">Response portion</b>&nbsp;&nbsp;
          hits: {r_hits} &nbsp;·&nbsp; mean act: {r_mean:.3f}
        </div>
        <div style="background:rgba(220,50,50,0.15);border-radius:6px;padding:7px 14px;color:#555;">
          <b style="color:#c0392b;">&#9632; highlighted</b>&nbsp;&nbsp;
          feature activation (hover for value)
        </div>
      </div>
      <hr style="margin:0 0 12px 0;border-color:#ddd;">
      <div style="line-height:2.2;font-size:0.88em;word-wrap:break-word;">
        {"".join(spans)}
      </div>
    </div>"""

    ipy_display(HTML(html))


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
    figsize: tuple = (900, 700),
    context_threshold: float = 0.025,
    token_threshold: float = 0.04,
    size_by: str = "composite_score",
):
    """
    Interactive scatterplot with context_specificity_vs_baseline on the x-axis and
    token_specificity_vs_baseline on the y-axis.

    Hover over any point to see its feature_id, both specificity scores, composite
    score, hit count, and mean activation.

    Dashed lines at context_threshold / token_threshold divide the plot into four quadrants:

        High context + High token  → concept-specific features
        Low  context + High token  → token-specific, context-agnostic (e.g. "the")
        High context + Low  token  → context-specific, token-diverse
        Low  context + Low  token  → noisy or near-dead

    Parameters
    ----------
    top_n               : restrict to the top-N features by composite_score (None = all features)
    figsize             : (width_px, height_px) for the plotly figure
    context_threshold   : x-axis divider for the quadrant lines
    token_threshold     : y-axis divider for the quadrant lines
    size_by             : "composite_score" (size=composite, color=firing_rate)
                          or "firing_rate"         (size=firing_rate, color=composite_score)
    """
    if size_by not in ("composite_score", "firing_rate"):
        raise ValueError("size_by must be 'composite_score' or 'firing_rate'")

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

    def _normalize(series):
        lo, hi = series.min(), series.max()
        return (series - lo) / (hi - lo + 1e-8)

    if size_by == "composite_score":
        marker_sizes = (6 + 18 * _normalize(df["composite_score"])).tolist()
        marker_color = df["firing_rate"]
        colorscale = "Blues"
        colorbar_title = "firing_rate"
    else:
        marker_sizes = (6 + 18 * _normalize(df["firing_rate"])).tolist()
        marker_color = df["composite_score"]
        colorscale = "RdYlGn"
        colorbar_title = "composite_score"

    hover_text = [
        (
            f"<b>feature {int(row.feature_id)}</b><br>"
            f"context_vs_baseline: {row.context_specificity_vs_baseline:.4f}<br>"
            f"token_vs_baseline: {row.token_specificity_vs_baseline:.4f}<br>"
            f"composite_score: {row.composite_score:.4f}<br>"
            f"firing_rate: {row.firing_rate:.4f}<br>"
            f"mean_activation: {row.mean_activation:.4f}"
        )
        for row in df.itertuples()
    ]

    scatter = go.Scatter(
        x=df["context_specificity_vs_baseline"],
        y=df["token_specificity_vs_baseline"],
        mode="markers",
        marker=dict(
            size=marker_sizes,
            color=marker_color,
            colorscale=colorscale,
            colorbar=dict(title=colorbar_title),
            line=dict(width=0.5, color="grey"),
            opacity=0.85,
        ),
        text=hover_text,
        hoverinfo="text",
        showlegend=False,
    )

    x_range = [df["context_specificity_vs_baseline"].min(), df["context_specificity_vs_baseline"].max()]
    y_range = [df["token_specificity_vs_baseline"].min(), df["token_specificity_vs_baseline"].max()]
    x_pad = (x_range[1] - x_range[0]) * 0.08
    y_pad = (y_range[1] - y_range[0]) * 0.08
    xl = [x_range[0] - x_pad, x_range[1] + x_pad]
    yl = [y_range[0] - y_pad, y_range[1] + y_pad]

    cx = context_threshold
    ty = token_threshold
    quadrant_labels = [
        dict(x=(xl[0] + cx) / 2, y=(ty + yl[1]) / 2, text="context-specific<br>token-diverse"),
        dict(x=(cx + xl[1]) / 2, y=(ty + yl[1]) / 2, text="concept-specific"),
        dict(x=(xl[0] + cx) / 2, y=(yl[0] + ty) / 2, text="noisy / near-dead"),
        dict(x=(cx + xl[1]) / 2, y=(yl[0] + ty) / 2, text="token-specific<br>context-agnostic"),
    ]
    annotations = [
        dict(
            x=q["x"], y=q["y"],
            text=q["text"],
            showarrow=False,
            font=dict(size=10, color="#888"),
            xref="x", yref="y",
        )
        for q in quadrant_labels
    ]

    n_label = f"top {len(df)}" if top_n is not None else f"all {len(df)}"
    size_color_label = "size = composite_score · color = firing_rate" if size_by == "composite_score" else "size = firing_rate · color = composite_score"
    layout = go.Layout(
        title=dict(
            text=f"Feature specificity landscape ({n_label} features)<br>"
                 f"<sup>{size_color_label}</sup>",
            font=dict(size=14),
        ),
        xaxis=dict(title="context_specificity_vs_baseline", zeroline=False, range=xl),
        yaxis=dict(title="token_specificity_vs_baseline", zeroline=False, range=yl),
        width=figsize[0],
        height=figsize[1],
        hovermode="closest",
        annotations=annotations,
        shapes=[
            dict(type="line", x0=cx, x1=cx, y0=yl[0], y1=yl[1],
                 line=dict(color="#aaa", dash="dash", width=1.2)),
            dict(type="line", x0=xl[0], x1=xl[1], y0=ty, y1=ty,
                 line=dict(color="#aaa", dash="dash", width=1.2)),
        ],
    )

    go.Figure(data=[scatter], layout=layout).show()


def plot_prompt_vs_response_features(
    prompt_tracer: FeatureTracer,
    response_tracer: FeatureTracer,
    min_hits: int = 5,
    min_context_specificity_vs_baseline: float | None = None,
    max_token_specificity_vs_baseline: float | None = None,
    figsize: tuple = (900, 750),
):
    """
    Compare features that appear in both a prompt-mode tracer and a response-mode tracer.

    Only shared features are plotted (those that fired in both tracers):
      x-axis = hit count in prompt_tracer
      y-axis = hit count in response_tracer
      color  = cosine similarity between mean context embeddings across the two tracers
               (green = same semantic context in both modes, red = different)

    Specificity filters (applied per-tracer, feature excluded if it fails in either):
      min_context_specificity_vs_baseline : exclude features below this context specificity
      max_token_specificity_vs_baseline   : exclude features above this token specificity
                                            (useful for removing token-specific features
                                            that aren't capturing context)

    Requires compute_feature_embeddings() on both tracers.
    Filters require feature_specificity_scores() on both tracers.
    """
    prompt_embs = prompt_tracer.get_feature_embeddings()
    response_embs = response_tracer.get_feature_embeddings()

    if prompt_embs is None or response_embs is None:
        print("Call compute_feature_embeddings() on both tracers first.")
        return

    # Build specificity lookup tables (None if scores not yet computed)
    def _scores_lookup(tracer, col):
        try:
            return tracer.get_feature_specificity_scores_df().set_index("feature_id")[col].to_dict()
        except RuntimeError:
            return None

    p_ctx  = _scores_lookup(prompt_tracer,   "context_specificity_vs_baseline")
    r_ctx  = _scores_lookup(response_tracer, "context_specificity_vs_baseline")
    p_tok  = _scores_lookup(prompt_tracer,   "token_specificity_vs_baseline")
    r_tok  = _scores_lookup(response_tracer, "token_specificity_vs_baseline")

    shared_fids = set(prompt_embs.keys()) & set(response_embs.keys())

    rows = []
    for fid in shared_fids:
        p_hits = len(prompt_embs[fid]["activations"])
        r_hits = len(response_embs[fid]["activations"])

        if max(p_hits, r_hits) < min_hits:
            continue

        # Specificity filters — skip if either tracer fails the threshold
        if min_context_specificity_vs_baseline is not None and p_ctx and r_ctx:
            if p_ctx.get(fid, -np.inf) < min_context_specificity_vs_baseline:
                continue
            if r_ctx.get(fid, -np.inf) < min_context_specificity_vs_baseline:
                continue

        if max_token_specificity_vs_baseline is not None and p_tok and r_tok:
            if p_tok.get(fid, np.inf) > max_token_specificity_vs_baseline:
                continue
            if r_tok.get(fid, np.inf) > max_token_specificity_vs_baseline:
                continue

        p_mean = prompt_embs[fid]["context_embeddings"].mean(axis=0)
        p_mean /= np.linalg.norm(p_mean) + 1e-8
        r_mean = response_embs[fid]["context_embeddings"].mean(axis=0)
        r_mean /= np.linalg.norm(r_mean) + 1e-8
        emb_sim = float(p_mean @ r_mean)

        rows.append({
            "feature_id": fid,
            "prompt_hits": p_hits,
            "response_hits": r_hits,
            "emb_sim": emb_sim,
        })

    if not rows:
        print(f"No shared features passed all filters (min_hits={min_hits}).")
        return

    df = pd.DataFrame(rows)

    hover_text = [
        (
            f"<b>feature {int(r.feature_id)}</b><br>"
            f"prompt_hits: {int(r.prompt_hits)}<br>"
            f"response_hits: {int(r.response_hits)}<br>"
            f"embedding_sim: {r.emb_sim:.4f}"
        )
        for r in df.itertuples()
    ]

    filter_parts = []
    if min_context_specificity_vs_baseline is not None:
        filter_parts.append(f"min_context_spec={min_context_specificity_vs_baseline}")
    if max_token_specificity_vs_baseline is not None:
        filter_parts.append(f"max_token_spec={max_token_specificity_vs_baseline}")
    filter_str = " · ".join(filter_parts) + " · " if filter_parts else ""

    layout = go.Layout(
        title=dict(
            text=(
                f"Prompt vs. response feature comparison — {len(df)} shared features<br>"
                f"<sup>{filter_str}min_hits={min_hits}</sup>"
            ),
            font=dict(size=14),
        ),
        xaxis=dict(title="prompt_tracer hits"),
        yaxis=dict(title="response_tracer hits"),
        width=figsize[0],
        height=figsize[1],
        hovermode="closest",
    )

    go.Figure(
        data=[go.Scatter(
            x=df["prompt_hits"], y=df["response_hits"],
            mode="markers",
            marker=dict(
                color=df["emb_sim"],
                colorscale="RdYlGn",
                cmin=-1, cmax=1,
                colorbar=dict(title="embedding<br>similarity", x=1.02),
                size=10,
                opacity=0.85,
                line=dict(width=0.5, color="grey"),
            ),
            text=hover_text,
            hoverinfo="text",
            showlegend=False,
        )],
        layout=layout,
    ).show()
