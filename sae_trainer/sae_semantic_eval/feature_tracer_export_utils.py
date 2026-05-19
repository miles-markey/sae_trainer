import json
from pathlib import Path

from .feature_tracer_utils import FeatureTracer


def export_feature_scores_csv(tracer: FeatureTracer, path: str | Path):
    """
    Export feature specificity scores to CSV.

    Columns: feature_id, hits, n_prompts, mean_activation, firing_rate,
             context_specificity, context_specificity_vs_baseline,
             token_specificity, token_specificity_vs_baseline, composite_score
    """
    df = tracer.get_feature_specificity_scores_df()
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} features → {path}")


def export_feature_cards_json(
    tracer: FeatureTracer,
    path: str | Path,
    top_n_contexts: int = 5,
    window: int = 8,
):
    """
    Export per-feature card data to JSON.

    Each entry contains:
      contexts        — top_n_contexts windows (pre/mid/post split, activation,
                        token_pos, prompt_id), one per unique prompt, sorted by
                        activation descending
      activations     — full list of activation values for the histogram
      position_profile — [{rel_pos, hits}, ...] for the relative position line chart
    """
    df = tracer.to_dataframe()

    # Export only features that have specificity scores so the JSON keys
    # match the CSV rows exactly.
    try:
        scores_df = tracer.get_feature_specificity_scores_df()
        feature_ids = scores_df["feature_id"].tolist()
    except RuntimeError:
        feature_ids = sorted(df["feature_id"].unique().tolist())

    cards = {}
    for fid in feature_ids:
        sub = df[df["feature_id"] == fid].sort_values("activation", ascending=False)
        if sub.empty:
            continue

        # --- Top-N contexts (one per prompt) ---
        top_rows = (
            sub.drop_duplicates(subset="prompt_id", keep="first")
            .head(top_n_contexts)
        )
        contexts = []
        for _, r in top_rows.iterrows():
            ids = tracer.tokenizer(
                r["generated_text"], return_tensors="pt"
            )["input_ids"][0]
            toks = tracer.tokenizer.convert_ids_to_tokens(ids.tolist())
            pos = int(r["token_pos"])
            lo = max(0, pos - window)
            hi = min(len(toks), pos + window + 1)
            contexts.append({
                "prompt_id":  r["prompt_id"],
                "token_pos":  pos,
                "activation": round(float(r["activation"]), 4),
                "pre":  tracer.tokenizer.convert_tokens_to_string(toks[lo:pos]),
                "mid":  tracer.tokenizer.convert_tokens_to_string([toks[pos]])
                        if pos < len(toks) else "",
                "post": tracer.tokenizer.convert_tokens_to_string(toks[pos + 1:hi]),
            })

        # --- Activation histogram data ---
        activations = [round(float(v), 4) for v in sub["activation"].tolist()]

        # --- Relative position profile ---
        rel_pos = (sub["token_pos_relative"] / sub["num_tokens_relative"]).round(2)
        pos_counts = rel_pos.value_counts().sort_index()
        position_profile = [
            {"rel_pos": round(float(k), 2), "hits": int(v)}
            for k, v in pos_counts.items()
        ]

        cards[str(int(fid))] = {
            "feature_id":       int(fid),
            "contexts":         contexts,
            "activations":      activations,
            "position_profile": position_profile,
        }

    with open(path, "w") as f:
        json.dump(cards, f, indent=2)
    print(f"Saved {len(cards)} feature cards → {path}")


def export_feature_docs(
    tracer: FeatureTracer,
    docs_dir: str | Path = "docs",
    top_n_contexts: int = 5,
    window: int = 8,
):
    """
    Export both files to docs_dir/.

    Requires:
      tracer.feature_specificity_scores() — to have been called (populates CSV columns
                                            and determines which features to export)
      tracer.compute_feature_embeddings() — called internally by feature_specificity_scores()

    Usage:
      export_feature_docs(tracer, docs_dir="docs")
    """
    docs_dir = Path(docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)

    export_feature_scores_csv(tracer, docs_dir / "feature_scores.csv")
    export_feature_cards_json(
        tracer, docs_dir / "feature_cards.json",
        top_n_contexts=top_n_contexts,
        window=window,
    )

    print(f"\nDocs exported to {docs_dir}/")
    print("  feature_scores.csv  — scatter plot data (one row per feature)")
    print("  feature_cards.json  — card data keyed by feature_id (str)")
