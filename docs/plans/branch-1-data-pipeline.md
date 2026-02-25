# Branch 1: Data Pipeline — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Branch:** `feature/data-pipeline`
**Merge into:** `main` first — branches 2 and 3 depend on the clean `output/clusters.parquet` this produces.

**Goal:** Clean the raw input data of spam/bots, re-run HDBSCAN clustering on clean embeddings with all 12 cores, fix cluster labels (strip URL tracking tokens), and regenerate all temporal stats and theme activations.

**Architecture:** New `scripts/0_preprocess_data.py` deduplicates and caps the raw CSV. New `scripts/cluster_label_utils.py` filters noise tokens from TF-IDF labels. Script 1 re-clusters with clean data and all cores. Scripts 2 and 4 regenerate downstream outputs.

**Tech Stack:** Python 3.13, pandas, numpy, hdbscan, scikit-learn (PCA, TF-IDF), UMAP, PyYAML.

**M2 Pro note:** 12 cores. `NUMBA_NUM_THREADS=12` and `n_jobs=-1` throughout. Script 1 (HDBSCAN + UMAP) **must be run from your Mac Terminal, not Cursor**, due to Numba cache file restrictions. Everything else runs fine in Cursor.

**Outputs produced (consumed by branches 2 and 3):**
- `output/data_clean.csv` — deduplicated, author-capped raw data
- `output/clusters.parquet` — new HDBSCAN cluster assignments with clean labels
- `output/temporal_stats.parquet` — monthly per-cluster metrics
- `output/temporal_events.parquet` — detected event spikes
- `output/cluster_trend_snapshots_1M.parquet` / `.csv` — monthly frontend snapshots
- `output/cluster_trend_snapshots_1W_recent.parquet` / `.csv` — weekly recency snapshots
- `output/theme_activations_windows.parquet` — all 28 themes scored against all clusters

---

## Task 1: Preprocessing script (`scripts/0_preprocess_data.py`)

**Files:**
- Create: `scripts/0_preprocess_data.py`
- Create: `tests/test_preprocess.py`

**What it does:**
1. Removes exact-duplicate `(text, Author)` pairs — keeps highest-engagement copy
2. Caps any single author at 5% of total posts — keeps their most-engaged posts

**Step 1: Write the failing tests**

```python
# tests/test_preprocess.py
import pandas as pd
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.preprocess_data import (
    deduplicate_exact,
    cap_author_dominance,
    load_and_clean,
)

def _make_df(rows):
    return pd.DataFrame(rows, columns=["text", "Author", "timestamp", "X Likes", "X Reposts"])

def test_deduplicate_exact_removes_same_author_same_text():
    df = _make_df([
        ("hello world", "alice", "2024-01-01", 5, 1),
        ("hello world", "alice", "2024-01-02", 2, 0),  # duplicate — remove
        ("hello world", "bob",   "2024-01-01", 3, 0),  # different author — keep
        ("different",   "alice", "2024-01-01", 1, 0),
    ])
    out = deduplicate_exact(df, text_col="text", author_col="Author")
    assert len(out) == 3
    alice_rows = out[out["Author"] == "alice"]
    assert len(alice_rows) == 1
    assert int(alice_rows["X Likes"].iloc[0]) == 5

def test_cap_author_dominance():
    # 100 posts, one author has 20 (20%) → capped to 5%=5 posts
    rows = [("post", "spammer", f"2024-01-{i:02d}", i, 0) for i in range(1, 21)]
    rows += [("post", f"user{i}", "2024-01-01", 1, 0) for i in range(80)]
    df = _make_df(rows)
    out = cap_author_dominance(df, author_col="Author", max_share=0.05,
                               engagement_cols=["X Likes", "X Reposts"])
    spammer_count = (out["Author"] == "spammer").sum()
    assert spammer_count == 5
    spammer_likes = sorted(out[out["Author"] == "spammer"]["X Likes"].tolist(), reverse=True)
    assert spammer_likes == [20, 19, 18, 17, 16]

def test_load_and_clean_returns_report():
    df = _make_df([
        ("same text", "bot", "2024-01-01", 1, 0),
        ("same text", "bot", "2024-01-01", 1, 0),
        ("other",     "human", "2024-01-01", 5, 2),
    ])
    cleaned, report = load_and_clean(df, text_col="text", author_col="Author",
                                     timestamp_col="timestamp")
    assert report["rows_input"] == 3
    assert report["rows_output"] == 2
    assert report["exact_duplicates_removed"] == 1
```

**Step 2: Run tests to verify they fail**

```bash
cd /Users/james/momentum
python -m pytest tests/test_preprocess.py -v
```
Expected: `ImportError: cannot import name 'deduplicate_exact'`

**Step 3: Implement `scripts/0_preprocess_data.py`**

```python
"""
Script 0: Preprocess raw data before clustering.

Actions:
  1. Deduplicate exact (text, Author) pairs — keep highest engagement.
  2. Cap single-author dominance at max_share of total posts.
  3. Save cleaned CSV and preprocessing report JSON.

Usage:
  python scripts/0_preprocess_data.py --config config.yaml [--force]
"""
import argparse
import json
import os

import pandas as pd
import yaml


def _engagement_score(df, engagement_cols):
    score = pd.Series(0.0, index=df.index)
    for col in engagement_cols:
        if col in df.columns:
            score += pd.to_numeric(df[col], errors="coerce").fillna(0)
    return score


def deduplicate_exact(df, text_col="text", author_col="Author", engagement_cols=None):
    """Remove exact (text, author) duplicates, keeping highest-engagement row."""
    engagement_cols = engagement_cols or ["X Likes", "X Reposts", "likes", "shares"]
    df = df.copy()
    df["_eng"] = _engagement_score(df, engagement_cols)
    deduped = (
        df.sort_values("_eng", ascending=False)
        .drop_duplicates(subset=[text_col, author_col], keep="first")
    )
    return deduped.drop(columns=["_eng"]).reset_index(drop=True)


def cap_author_dominance(df, author_col="Author", max_share=0.05, engagement_cols=None):
    """Cap any single author to at most max_share * total posts."""
    engagement_cols = engagement_cols or ["X Likes", "X Reposts", "likes", "shares"]
    df = df.copy()
    df["_eng"] = _engagement_score(df, engagement_cols)
    cap = max(1, int(len(df) * max_share))
    parts = []
    for _, group in df.groupby(author_col, sort=False):
        if len(group) > cap:
            group = group.nlargest(cap, "_eng", keep="first")
        parts.append(group)
    return pd.concat(parts).drop(columns=["_eng"]).reset_index(drop=True)


def load_and_clean(df, text_col="text", author_col="Author",
                   timestamp_col="timestamp", max_author_share=0.05,
                   engagement_cols=None):
    """Run full cleaning pipeline. Returns (cleaned_df, report_dict)."""
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

    input_csv = resolve(config["input"]["data"])
    text_col = config["input"]["text_column"]
    ts_col = config["input"]["timestamp_column"]
    output_dir = resolve(config["output"]["dir"])
    out_csv = os.path.join(output_dir, "data_clean.csv")
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
    cleaned.to_csv(out_csv, index=False)
    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Input:   {report['rows_input']:,}")
    print(f"Output:  {report['rows_output']:,}")
    print(f"  Exact dupes removed: {report['exact_duplicates_removed']:,}")
    print(f"  Author cap removed:  {report['author_cap_removed']:,}")
    print(f"  Total removed:       {report['total_removed']:,} ({report['removal_pct']}%)")
    print(f"Saved to {out_csv}")


if __name__ == "__main__":
    main()
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_preprocess.py -v
```
Expected: 3 PASSED

**Step 5: Run on real data**

```bash
python scripts/0_preprocess_data.py --config config.yaml --force
```
Check `output/preprocessing_report.json` — expect 10–30% rows removed.

**Step 6: Update `config.yaml` — point input at clean data**

Change `input.data`:
```yaml
input:
  data: "output/data_clean.csv"
```

**Step 7: Commit**

```bash
git add scripts/0_preprocess_data.py tests/test_preprocess.py config.yaml
git commit -m "feat: add preprocessing script — dedup spam posts, cap single-author dominance"
```

---

## Task 2: Clean cluster labels (`scripts/cluster_label_utils.py`)

**Files:**
- Create: `scripts/cluster_label_utils.py`
- Create: `tests/test_cluster_labels.py`
- Modify: `scripts/1_cluster_embeddings.py`

**Step 1: Write the failing tests**

```python
# tests/test_cluster_labels.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.cluster_label_utils import strip_url_tokens, clean_label_text

def test_strip_url_tokens_removes_hex_like_strings():
    text = "deepmind owq9fagevl 5tjwz1kzcf artificial intelligence london"
    cleaned = strip_url_tokens(text)
    assert "owq9fagevl" not in cleaned
    assert "5tjwz1kzcf" not in cleaned
    assert "deepmind" in cleaned
    assert "london" in cleaned

def test_strip_url_tokens_keeps_real_words():
    text = "alphafold protein structure deepmind"
    assert strip_url_tokens(text) == text

def test_clean_label_text_produces_readable_label():
    tokens = ["deepmind", "owq9fagevl", "protein", "lrvqaf3vww", "ai"]
    label = clean_label_text(tokens, n_top=3)
    assert "owq9fagevl" not in label
    assert "lrvqaf3vww" not in label
    assert label
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_cluster_labels.py -v
```
Expected: `ImportError: No module named 'scripts.cluster_label_utils'`

**Step 3: Create `scripts/cluster_label_utils.py`**

```python
"""
Clean, human-readable cluster labels.
Filters URL tracking tokens (8+ chars, <2 vowels) before TF-IDF labelling.
"""
import re

_VOWELS = set("aeiou")
_HEX_RE = re.compile(r"^[0-9a-f]{8,}$")
_MIXED_DIGIT_START = re.compile(r"^\d[a-z0-9]{7,}$")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _is_noise_token(token: str) -> bool:
    if len(token) < 8:
        return False
    if _HEX_RE.match(token):
        return True
    if _MIXED_DIGIT_START.match(token):
        return True
    if sum(1 for c in token if c in _VOWELS) < 2:
        return True
    return False


def strip_url_tokens(text: str) -> str:
    tokens = _TOKEN_RE.findall(text.lower())
    return " ".join(t for t in tokens if not _is_noise_token(t))


def clean_label_text(tokens: list, n_top: int = 3) -> str:
    clean = [t for t in tokens if not _is_noise_token(t)]
    return ", ".join(clean[:n_top]) if clean else "unlabelled"
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_cluster_labels.py -v
```
Expected: 3 PASSED

**Step 5: Modify `scripts/1_cluster_embeddings.py`**

In `generate_cluster_labels()`, find this line:
```python
top_terms = [feature_names[i] for i in top_indices]
labels[cluster_id] = ", ".join(top_terms)
```

Replace with:
```python
top_terms = [feature_names[i] for i in top_indices]
from scripts.cluster_label_utils import clean_label_text
labels[cluster_id] = clean_label_text(top_terms, n_top=3)
```

Also verify `main()` sets Numba threads. Find the line near the top:
```python
os.environ.setdefault("NUMBA_NUM_THREADS", str(n))
```
Change `setdefault` to a hard set so it always uses all cores:
```python
os.environ["NUMBA_NUM_THREADS"] = str(n)
```

**Step 6: Commit code changes**

```bash
git add scripts/cluster_label_utils.py scripts/1_cluster_embeddings.py tests/test_cluster_labels.py
git commit -m "feat: filter URL tracking tokens from cluster labels, ensure all cores used for HDBSCAN+UMAP"
```

**Step 7: Run Script 1 — FROM YOUR MAC TERMINAL (not Cursor)**

```bash
cd /Users/james/momentum
python scripts/1_cluster_embeddings.py --force
```

Expected:
```
Cores: 12 available, using 12 | HDBSCAN: 12 job(s) | UMAP (Numba): 12 thread(s)
Saved 4XXXX rows to output/clusters.parquet
```
Takes ~15–25 minutes. Go make coffee.

**Step 8: Verify clean labels**

```bash
python3 -c "
import pandas as pd
c = pd.read_parquet('output/clusters.parquet')
for cid, label in sorted(c.groupby('cluster')['cluster_label'].first().items()):
    if cid != -1:
        print(f'{cid}: {label}')
"
```
No `owq9fagevl`, `wgnfqclyr3`, or similar noise tokens.

**Step 9: Commit output**

```bash
git add output/clusters.parquet
git commit -m "data: re-cluster on clean data with noise-filtered labels"
```

---

## Task 3: Temporal analysis + theme activation (Scripts 2 and 4)

**Files:**
- Run: `scripts/2_temporal_analysis.py` (already handles weekly recency — see Task 9 additions from main plan)
- Run: `scripts/4_theme_activation_tfidf.py`

**Step 1: Add recency config to `config.yaml`**

Under `temporal:`:
```yaml
temporal:
  default_window: "1M"
  recency_window: "1W"
  recency_lookback_months: 12
  recency_min_posts_per_window: 3
  recency_min_density: 0.5
```

Under `output:`:
```yaml
  cluster_trend_snapshots_1w_recent: "output/cluster_trend_snapshots_1W_recent.parquet"
  cluster_trend_snapshots_1w_recent_csv: "output/cluster_trend_snapshots_1W_recent.csv"
```

**Step 2: Create `scripts/temporal_recency.py`**

```python
"""
Temporal recency utilities: weekly snapshots for the most recent N months.
Only clusters passing a density gate are included.
"""
import numpy as np
import pandas as pd


def is_cluster_weekly_dense_enough(df, cluster_id, min_posts=3, min_density=0.5):
    cdf = df[df["cluster"] == cluster_id].copy()
    if cdf.empty:
        return False
    cdf["timestamp"] = pd.to_datetime(cdf["timestamp"], errors="coerce")
    cdf = cdf.dropna(subset=["timestamp"])
    weekly = cdf.set_index("timestamp").resample("1W")["cluster"].count()
    active = weekly[weekly > 0]
    if active.empty:
        return False
    first, last = active.index.min(), active.index.max()
    active_range = weekly[(weekly.index >= first) & (weekly.index <= last)]
    return bool((active_range >= min_posts).sum() / max(len(active_range), 1) >= min_density)


def build_weekly_recency_snapshots(df, lookback_months=12, min_posts=3, min_density=0.5):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    cutoff = df["timestamp"].max() - pd.DateOffset(months=lookback_months)
    recent = df[df["timestamp"] >= cutoff].copy()

    dense_ids = [
        cid for cid in recent["cluster"].unique()
        if cid != -1 and is_cluster_weekly_dense_enough(
            recent, cid, min_posts=min_posts, min_density=min_density
        )
    ]

    _empty_cols = [
        "time_window", "cluster", "cluster_label", "post_count",
        "market_share", "volume_pct_change", "volume_volatility",
        "momentum", "lifecycle_state", "anomaly_score", "window_type",
    ]
    if not dense_ids:
        return pd.DataFrame(columns=_empty_cols)

    records = []
    for cid in dense_ids:
        cdf = recent[recent["cluster"] == cid].copy()
        label = cdf["cluster_label"].iloc[0] if "cluster_label" in cdf.columns else str(cid)
        r = cdf.set_index("timestamp").resample("1W").agg(post_count=("cluster", "count")).fillna(0)
        r["cluster"] = cid
        r["cluster_label"] = label
        r["volume_pct_change"] = r["post_count"].pct_change().fillna(0).clip(-10, 10)
        r["volume_volatility"] = r["post_count"].rolling(3, min_periods=1).std().fillna(0)
        r["momentum"] = r["post_count"] - r["post_count"].rolling(3, min_periods=1).mean()
        mean_v, std_v = r["post_count"].mean(), r["post_count"].std()
        r["anomaly_score"] = (r["post_count"] - mean_v) / (std_v if std_v > 0 else 1)
        recent_g = r["volume_pct_change"].tail(3).mean()
        recent_v = r["post_count"].tail(3).mean()
        overall_v = r["post_count"].mean()
        if overall_v == 0 or recent_v < overall_v * 0.2:
            lc = "dormant"
        elif recent_g > 0.5:
            lc = "emerging"
        elif recent_g > 0.1:
            lc = "trending"
        elif recent_g < -0.3:
            lc = "declining"
        else:
            lc = "stable"
        r["lifecycle_state"] = lc
        records.append(r)

    result = pd.concat(records).reset_index().rename(columns={"timestamp": "time_window"})
    total_pw = result.groupby("time_window")["post_count"].transform("sum")
    result["market_share"] = result["post_count"] / total_pw.replace(0, np.nan)
    result["window_type"] = "1W"
    result["time_window"] = pd.to_datetime(result["time_window"])
    return result[_empty_cols].sort_values(["time_window", "cluster"]).reset_index(drop=True)
```

**Step 3: Write the failing tests for temporal_recency**

```python
# tests/test_weekly_recency.py
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.temporal_recency import build_weekly_recency_snapshots, is_cluster_weekly_dense_enough

def _make_cluster_df():
    dates = pd.date_range("2025-01-06", periods=52, freq="W-MON")
    rows = []
    for d in dates:
        for _ in range(5):
            rows.append({"timestamp": d, "cluster": 1, "cluster_label": "dense",
                         "likes": 2, "reach": 10, "shares": 1, "comments": 0, "Author": "a"})
    for d in dates[:10]:
        rows.append({"timestamp": d, "cluster": 2, "cluster_label": "sparse",
                     "likes": 1, "reach": 5, "shares": 0, "comments": 0, "Author": "b"})
    return pd.DataFrame(rows)

def test_dense_cluster_qualifies():
    df = _make_cluster_df()
    assert is_cluster_weekly_dense_enough(df, 1, min_posts=3, min_density=0.5) is True

def test_sparse_cluster_does_not_qualify():
    df = _make_cluster_df()
    assert is_cluster_weekly_dense_enough(df, 2, min_posts=3, min_density=0.5) is False

def test_build_weekly_recency_snapshots_only_includes_dense_clusters():
    df = _make_cluster_df()
    snaps = build_weekly_recency_snapshots(df, lookback_months=12, min_posts=3, min_density=0.5)
    assert 1 in snaps["cluster"].unique()
    assert 2 not in snaps["cluster"].unique()

def test_build_weekly_recency_snapshots_has_required_columns():
    df = _make_cluster_df()
    snaps = build_weekly_recency_snapshots(df, lookback_months=12, min_posts=3, min_density=0.5)
    for col in ["time_window", "cluster", "post_count", "market_share",
                "volume_pct_change", "volume_volatility", "momentum",
                "lifecycle_state", "anomaly_score", "window_type"]:
        assert col in snaps.columns
    assert (snaps["window_type"] == "1W").all()
```

**Step 4: Run tests**

```bash
python -m pytest tests/test_weekly_recency.py -v
```
Expected: 4 PASSED

**Step 5: Extend `scripts/2_temporal_analysis.py` — add weekly recency at end of `main()`**

After the existing monthly snapshot save block, add:

```python
# --- Weekly recency snapshot ---
from scripts.temporal_recency import build_weekly_recency_snapshots
recency_cfg = config.get("temporal", {})
weekly_recency = build_weekly_recency_snapshots(
    df,
    lookback_months=int(recency_cfg.get("recency_lookback_months", 12)),
    min_posts=int(recency_cfg.get("recency_min_posts_per_window", 3)),
    min_density=float(recency_cfg.get("recency_min_density", 0.5)),
)
weekly_recency_parquet = resolve(
    config["output"].get("cluster_trend_snapshots_1w_recent",
                         "output/cluster_trend_snapshots_1W_recent.parquet")
)
weekly_recency_csv = resolve(
    config["output"].get("cluster_trend_snapshots_1w_recent_csv",
                         "output/cluster_trend_snapshots_1W_recent.csv")
)
weekly_recency.to_parquet(weekly_recency_parquet, index=False)
weekly_recency.to_csv(weekly_recency_csv, index=False)
print(f"Saved weekly recency snapshots ({len(weekly_recency)} rows, "
      f"{weekly_recency['cluster'].nunique()} dense clusters) to {weekly_recency_parquet}")
```

**Step 6: Run Script 2**

```bash
python scripts/2_temporal_analysis.py --config config.yaml --force
```

Expected: monthly outputs + new weekly recency files.

**Step 7: Run Script 4 (all 28 themes, 12 cores)**

```bash
python scripts/4_theme_activation_tfidf.py --config config.yaml --mode windows --n-jobs 12 --force
```

Verify 28 theme columns:
```bash
python3 -c "
import pandas as pd
ta = pd.read_parquet('output/theme_activations_windows.parquet')
cols = [c for c in ta.columns if c.endswith('_score')]
print(f'{len(cols)} theme score columns:', cols)
"
```
Expected: 28 columns.

**Step 8: Run all tests**

```bash
python -m pytest tests/ -v
```
Expected: all PASS

**Step 9: Commit everything**

```bash
git add scripts/temporal_recency.py scripts/2_temporal_analysis.py \
        scripts/4_theme_activation_tfidf.py \
        tests/test_weekly_recency.py config.yaml \
        output/temporal_stats.parquet output/temporal_events.parquet \
        output/cluster_trend_snapshots_1M.parquet output/cluster_trend_snapshots_1M.csv \
        output/cluster_trend_snapshots_1W_recent.parquet output/cluster_trend_snapshots_1W_recent.csv \
        output/theme_activations_windows.parquet
git commit -m "feat: regenerate temporal stats + theme activations; add weekly recency snapshots"
```

---

## Merge Signal

When all tests pass and `output/theme_activations_windows.parquet` contains 28 theme columns, this branch is ready to merge into `main`. Branches 2 and 3 can then be rebased onto main and executed in parallel.

```bash
git checkout main
git merge feature/data-pipeline
```
