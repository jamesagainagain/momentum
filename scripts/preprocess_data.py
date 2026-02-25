"""
Preprocess raw data before clustering.

Actions:
  1. Deduplicate exact (text, Author) pairs — keep highest engagement.
  2. Cap single-author dominance at max_share of total posts.
  3. Save cleaned CSV and preprocessing report JSON.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import yaml


def _engagement_score(df, engagement_cols):
    score = pd.Series(0.0, index=df.index)
    for col in engagement_cols:
        if col in df.columns:
            score += pd.to_numeric(df[col], errors="coerce").fillna(0)
    return score


def deduplicate_exact(df, text_col="text", author_col="Author", engagement_cols=None):
    """Remove exact (text, author) duplicates, keeping highest-engagement row.

    Preserves the original DataFrame index so callers can track which rows survived.
    """
    engagement_cols = engagement_cols or ["X Likes", "X Reposts", "likes", "shares"]
    df = df.copy()
    df["_eng"] = _engagement_score(df, engagement_cols)
    deduped = (
        df.sort_values("_eng", ascending=False)
        .drop_duplicates(subset=[text_col, author_col], keep="first")
    )
    return deduped.drop(columns=["_eng"])


def cap_author_dominance(df, author_col="Author", max_share=0.05, engagement_cols=None):
    """Cap any single author to at most max_share * total posts.

    Preserves the original DataFrame index so callers can track which rows survived.
    """
    engagement_cols = engagement_cols or ["X Likes", "X Reposts", "likes", "shares"]
    df = df.copy()
    df["_eng"] = _engagement_score(df, engagement_cols)
    cap = max(1, int(len(df) * max_share))
    parts = []
    for _, group in df.groupby(author_col, sort=False):
        if len(group) > cap:
            group = group.nlargest(cap, "_eng", keep="first")
        parts.append(group)
    return pd.concat(parts).drop(columns=["_eng"])


def load_and_clean(df, text_col="text", author_col="Author",
                   timestamp_col="timestamp", max_author_share=0.05,
                   engagement_cols=None):
    """Run full cleaning pipeline. Returns (cleaned_df, report_dict).

    cleaned_df retains its original integer index (positions in the input df)
    so callers can use it to slice a positionally-aligned array (e.g. embeddings).
    Call .reset_index(drop=True) on the result if a clean 0-based index is needed.
    """
    engagement_cols = engagement_cols or ["X Likes", "X Reposts", "likes", "shares"]
    rows_input = len(df)
    deduped = deduplicate_exact(df, text_col=text_col, author_col=author_col,
                                engagement_cols=engagement_cols)
    exact_removed = rows_input - len(deduped)
    capped = cap_author_dominance(deduped, author_col=author_col,
                                  max_share=max_author_share,
                                  engagement_cols=engagement_cols)
    cap_removed = len(deduped) - len(capped)
    report = {
        "rows_input": rows_input,
        "rows_output": len(capped),
        "exact_duplicates_removed": int(exact_removed),
        "author_cap_removed": int(cap_removed),
        "total_removed": int(rows_input - len(capped)),
        "removal_pct": round(100 * (rows_input - len(capped)) / max(rows_input, 1), 2),
        "author_cap_share": max_author_share,
    }
    return capped, report


def main():
    parser = argparse.ArgumentParser(description="Preprocess raw data before clustering")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config_dir = os.path.dirname(config_path)

    def resolve(p):
        return p if os.path.isabs(p) else os.path.join(config_dir, p)

    input_csv = resolve(config["input"].get("raw_data") or config["input"]["data"])
    text_col = config["input"]["text_column"]
    ts_col = config["input"]["timestamp_column"]
    output_dir = resolve(config["output"]["dir"])
    out_csv = os.path.join(output_dir, "data_clean.csv")
    out_indices = os.path.join(output_dir, "data_clean_indices.npy")
    out_report = os.path.join(output_dir, "preprocessing_report.json")

    if os.path.exists(out_csv) and not args.force:
        print(f"Output exists: {out_csv} (use --force to rerun)")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading {input_csv}...")
    df = pd.read_csv(input_csv, low_memory=False)
    print(f"Loaded {len(df):,} rows")

    author_col = next((c for c in ["Author", "author"] if c in df.columns), text_col)
    if author_col == text_col:
        print("WARN: No Author column found; deduplicating by text only")

    cleaned, report = load_and_clean(df, text_col=text_col, author_col=author_col,
                                     timestamp_col=ts_col, max_author_share=0.05)

    kept_indices = cleaned.index.to_numpy()
    np.save(out_indices, kept_indices)

    cleaned.reset_index(drop=True).to_csv(out_csv, index=False)
    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Input:   {report['rows_input']:,}")
    print(f"Output:  {report['rows_output']:,}")
    print(f"  Exact dupes removed: {report['exact_duplicates_removed']:,}")
    print(f"  Author cap removed:  {report['author_cap_removed']:,}")
    print(f"  Total removed:       {report['total_removed']:,} ({report['removal_pct']}%)")
    print(f"Saved to {out_csv}")
    print(f"Kept indices saved to {out_indices} (used by script 1 to slice embeddings)")


if __name__ == "__main__":
    main()
