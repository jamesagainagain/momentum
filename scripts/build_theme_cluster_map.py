"""
Script 4b: Build theme→cluster mapping from theme activation scores.

For each theme in theme_lexicon.yaml, finds the HDBSCAN cluster with the
highest average TF-IDF activation score. Saves output/theme_cluster_map.json.

Usage:
  python scripts/4b_build_theme_cluster_map.py --config config.yaml [--force]
"""
import argparse
import json
import os

import pandas as pd
import yaml


def build_theme_cluster_map(activations_df, theme_ids):
    """
    For each theme_id, find the cluster with the highest mean activation score.
    Returns {theme_id (int): {"cluster_id", "cluster_label", "avg_score"}}
    """
    mapping = {}
    for tid in theme_ids:
        score_col = f"theme_{tid}_score"
        if score_col not in activations_df.columns:
            continue
        grouped = (
            activations_df.groupby(["cluster", "cluster_label"])[score_col]
            .mean()
            .reset_index()
            .rename(columns={score_col: "avg_score"})
        )
        grouped = grouped[grouped["avg_score"] > 0].sort_values("avg_score", ascending=False)
        if grouped.empty:
            continue
        best = grouped.iloc[0]
        mapping[int(tid)] = {
            "cluster_id": int(best["cluster"]),
            "cluster_label": str(best["cluster_label"]),
            "avg_score": float(best["avg_score"]),
        }
    return mapping


def validate_map(mapping, expected_theme_count):
    return [
        f"Theme {i} has no cluster match"
        for i in range(expected_theme_count)
        if i not in mapping
    ]


def main():
    parser = argparse.ArgumentParser(description="Build theme→cluster mapping")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config_dir = os.path.dirname(config_path)

    def resolve(p):
        return p if os.path.isabs(p) else os.path.join(config_dir, p)

    ta_cfg = config.get("theme_activation", {})
    lexicon_path = resolve(ta_cfg.get("theme_lexicon", "theme_lexicon.yaml"))
    ta_windows_path = resolve(
        ta_cfg.get("output", "output/theme_activations.parquet")
    ).replace(".parquet", "_windows.parquet")
    out_path = resolve("output/theme_cluster_map.json")

    if os.path.exists(out_path) and not args.force:
        print(f"Output exists: {out_path} (use --force to rerun)")
        return

    with open(lexicon_path, encoding="utf-8") as f:
        themes = yaml.safe_load(f).get("themes", [])
    theme_ids = [int(t["id"]) for t in themes]
    theme_names = {int(t["id"]): t.get("name", str(t["id"])) for t in themes}

    if not os.path.exists(ta_windows_path):
        raise FileNotFoundError(
            f"Theme activations not found: {ta_windows_path}\n"
            "Run branch 1 first: python scripts/4_theme_activation_tfidf.py --force"
        )

    print(f"Loading {ta_windows_path}...")
    df = pd.read_parquet(ta_windows_path)
    mapping = build_theme_cluster_map(df, theme_ids)
    warnings = validate_map(mapping, expected_theme_count=len(themes))

    output = {}
    for tid in theme_ids:
        if tid in mapping:
            output[str(tid)] = {"theme_name": theme_names.get(tid, str(tid)), **mapping[tid]}
        else:
            output[str(tid)] = {
                "theme_name": theme_names.get(tid, str(tid)),
                "cluster_id": None, "cluster_label": None, "avg_score": 0.0,
            }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=True)

    print(f"\n{'ID':>4} | {'Theme Name':<40} | {'Cluster':>8} | {'Label':<35} | Score")
    print("-" * 100)
    for tid in sorted(theme_ids):
        e = output[str(tid)]
        print(f"{tid:>4} | {e['theme_name']:<40} | {str(e['cluster_id'] or '—'):>8} | "
              f"{str(e['cluster_label'] or '—'):<35} | {e['avg_score']:.4f}")

    if warnings:
        print(f"\nWarnings: {len(warnings)} themes unmatched")
        for w in warnings:
            print(f"  ⚠ {w}")

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
