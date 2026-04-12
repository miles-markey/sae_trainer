import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import html as html_lib
from IPython.display import HTML, display as ipy_display
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
    top_rows = sub.sort_values("activation", ascending=False).head(top_n)
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