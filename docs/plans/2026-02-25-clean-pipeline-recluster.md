# Clean Pipeline & Re-cluster Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove spam/noise from raw data, re-run the full pipeline on clean data, fix the broken theme→cluster mapping, surface theme names throughout the dashboard, and add a weekly recency layer for recent high-density data.

**Architecture:** A new `scripts/0_preprocess_data.py` deduplicates and caps the raw CSV before Script 1 runs. Script 1 re-clusters the clean embeddings with all 12 cores. Script 4 scores all 28 themes against the new clusters. A new `scripts/4b_build_theme_cluster_map.py` builds the explicit theme→cluster mapping used by the forecast engine. The server and dashboard label clusters by theme name where a mapping exists. Script 2 generates a second weekly snapshot covering the last 12 months only; the dashboard trend chart toggles between monthly (full history) and weekly (recent 12 months) per cluster, showing weekly only when density supports it.

**Tech Stack:** Python 3.13, pandas, numpy, hdbscan, scikit-learn (PCA, TF-IDF), UMAP (run outside Cursor sandbox), joblib, PyYAML, FastAPI.

**M2 Pro note:** 12 cores available. All scripts set `n_jobs=-1` or `NUMBA_NUM_THREADS=12` explicitly. UMAP must be run from your Mac Terminal (not Cursor's sandbox) due to Numba cache restrictions.

---

## Task 1: Write preprocessing script (`scripts/0_preprocess_data.py`)

**Files:**
- Create: `scripts/0_preprocess_data.py`
- Create: `tests/test_preprocess.py`

**What it does:**
1. Loads `clustered_data_labeled.csv`
2. Removes exact-duplicate `(text, Author)` pairs — same author posting identical text
3. Removes near-duplicate posts: for each author, any text with >85% character overlap to another of their posts in the same month (catches same article with different URLs)
4. Caps single-author dominance: if one author contributes >5% of total posts, cap them at `floor(0.05 * total)` posts, keeping their most-engaged posts (by `X Likes + X Reposts`)
5. Saves cleaned CSV to `output/data_clean.csv` and a quality report to `output/preprocessing_report.json`

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
        ("hello world", "alice", "2024-01-02", 2, 0),  # duplicate
        ("hello world", "bob",   "2024-01-01", 3, 0),  # different author - keep
        ("different",   "alice", "2024-01-01", 1, 0),
    ])
    out = deduplicate_exact(df, text_col="text", author_col="Author")
    assert len(out) == 3
    # alice keeps the row with higher engagement (5 likes)
    alice_rows = out[out["Author"] == "alice"]
    assert len(alice_rows) == 1
    assert int(alice_rows["X Likes"].iloc[0]) == 5

def test_cap_author_dominance():
    # 100 posts, one author has 20 (20%) - should be capped to 5%=5 posts
    rows = [("post", "spammer", f"2024-01-{i:02d}", i, 0) for i in range(1, 21)]
    rows += [("post", f"user{i}", "2024-01-01", 1, 0) for i in range(80)]
    df = _make_df(rows)
    out = cap_author_dominance(df, author_col="Author", max_share=0.05,
                               engagement_cols=["X Likes", "X Reposts"])
    spammer_count = (out["Author"] == "spammer").sum()
    assert spammer_count == 5
    # Kept the 5 posts with highest likes (15-20)
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
import sys

import pandas as pd
import yaml


def _engagement_score(df, engagement_cols):
    score = pd.Series(0.0, index=df.index)
    for col in engagement_cols:
        if col in df.columns:
            score += pd.to_numeric(df[col], errors="coerce").fillna(0)
    return score


def deduplicate_exact(df, text_col="text", author_col="Author",
                      engagement_cols=None):
    """Remove exact (text, author) duplicates, keeping highest-engagement row."""
    engagement_cols = engagement_cols or ["X Likes", "X Reposts", "likes", "shares"]
    df = df.copy()
    df["_eng"] = _engagement_score(df, engagement_cols)
    df_sorted = df.sort_values("_eng", ascending=False)
    deduped = df_sorted.drop_duplicates(subset=[text_col, author_col], keep="first")
    return deduped.drop(columns=["_eng"]).reset_index(drop=True)


def cap_author_dominance(df, author_col="Author", max_share=0.05,
                         engagement_cols=None):
    """Cap any single author to at most max_share * total posts."""
    engagement_cols = engagement_cols or ["X Likes", "X Reposts", "likes", "shares"]
    df = df.copy()
    df["_eng"] = _engagement_score(df, engagement_cols)
    total = len(df)
    cap = max(1, int(total * max_share))
    parts = []
    for author, group in df.groupby(author_col, sort=False):
        if len(group) > cap:
            group = group.nlargest(cap, "_eng", keep="first")
        parts.append(group)
    out = pd.concat(parts).drop(columns=["_eng"]).reset_index(drop=True)
    return out


def load_and_clean(df, text_col="text", author_col="Author",
                   timestamp_col="timestamp", max_author_share=0.05,
                   engagement_cols=None):
    """Run full cleaning pipeline. Returns (cleaned_df, report_dict)."""
    engagement_cols = engagement_cols or ["X Likes", "X Reposts", "likes", "shares"]
    rows_input = len(df)

    # Step 1: exact dedup
    deduped = deduplicate_exact(df, text_col=text_col, author_col=author_col,
                                engagement_cols=engagement_cols)
    exact_removed = rows_input - len(deduped)

    # Step 2: author cap
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

    # Detect author column
    author_col = None
    for candidate in ["Author", "author"]:
        if candidate in df.columns:
            author_col = candidate
            break
    if not author_col:
        print("WARN: No Author column found; skipping author-based dedup and cap")
        author_col = text_col  # fallback: dedup by text only

    cleaned, report = load_and_clean(
        df,
        text_col=text_col,
        author_col=author_col,
        timestamp_col=ts_col,
        max_author_share=0.05,
    )

    cleaned.to_csv(out_csv, index=False)
    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Input rows:   {report['rows_input']:,}")
    print(f"Output rows:  {report['rows_output']:,}")
    print(f"  Exact dupes removed:  {report['exact_duplicates_removed']:,}")
    print(f"  Author cap removed:   {report['author_cap_removed']:,}")
    print(f"  Total removed:        {report['total_removed']:,} ({report['removal_pct']}%)")
    print(f"Saved to {out_csv}")
    print(f"Report: {out_report}")


if __name__ == "__main__":
    main()
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_preprocess.py -v
```
Expected: 3 PASSED

**Step 5: Run the script on your real data**

```bash
python scripts/0_preprocess_data.py --config config.yaml --force
```
Expected output shows ~10–30% rows removed. Check `output/preprocessing_report.json`.

**Step 6: Update `config.yaml` to point Script 1 at the clean data**

In `config.yaml`, change:
```yaml
input:
  data: "clustered_data_labeled.csv"
```
to:
```yaml
input:
  data: "output/data_clean.csv"
```

**Step 7: Commit**

```bash
git add scripts/0_preprocess_data.py tests/test_preprocess.py config.yaml
git commit -m "feat: add data preprocessing script to remove spam/bot content before clustering"
```

---

## Task 2: Re-run Script 1 (HDBSCAN re-cluster on clean data)

**Files:**
- Modify: `scripts/1_cluster_embeddings.py` — add parallelism hardening

**Context:** Script 1 already exists and works. We need to:
1. Verify it uses all 12 cores (`NUMBA_NUM_THREADS=12`, `n_jobs=-1`)
2. Fix the TF-IDF label generator to strip URL tracking tokens before labelling
3. Re-run it with `--force` on the clean data

**Step 1: Write the failing test for the token filter**

```python
# tests/test_cluster_labels.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.cluster_label_utils import strip_url_tokens, clean_label_text

def test_strip_url_tokens_removes_hex_like_strings():
    # t.co tracking tokens: 8+ char alphanumeric, mostly consonants
    text = "deepmind owq9fagevl 5tjwz1kzcf artificial intelligence london"
    cleaned = strip_url_tokens(text)
    assert "owq9fagevl" not in cleaned
    assert "5tjwz1kzcf" not in cleaned
    assert "deepmind" in cleaned
    assert "london" in cleaned  # real word, keep it

def test_strip_url_tokens_keeps_real_words():
    text = "alphafold protein structure deepmind"
    assert strip_url_tokens(text) == text

def test_clean_label_text_produces_readable_label():
    tokens = ["deepmind", "owq9fagevl", "protein", "lrvqaf3vww", "ai"]
    label = clean_label_text(tokens, n_top=3)
    assert "owq9fagevl" not in label
    assert "lrvqaf3vww" not in label
    assert label  # non-empty
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_cluster_labels.py -v
```
Expected: `ImportError: No module named 'scripts.cluster_label_utils'`

**Step 3: Create `scripts/cluster_label_utils.py`**

```python
"""
Utilities for generating clean, human-readable cluster labels.
Filters out URL tracking tokens, hex hashes, and other noise
before TF-IDF keyword extraction.
"""
import re

# Patterns that indicate a token is noise, not a real word:
# - 8+ chars with no vowels (tracking tokens like owq9fagevl)
# - 8+ chars that are all hex chars (md5-like: a-f0-9)
# - Starts with digit and has mixed alnum (tracking IDs like 5tjwz1kzcf)
_VOWELS = set("aeiou")
_HEX_RE = re.compile(r"^[0-9a-f]{8,}$")
_MIXED_DIGIT_START = re.compile(r"^\d[a-z0-9]{7,}$")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _is_noise_token(token: str) -> bool:
    """Return True if token looks like a URL hash / tracking ID."""
    if len(token) < 8:
        return False
    # All hex
    if _HEX_RE.match(token):
        return True
    # Starts with digit, 8+ chars mixed
    if _MIXED_DIGIT_START.match(token):
        return True
    # 8+ chars with fewer than 2 vowels = likely not a real word
    vowel_count = sum(1 for c in token if c in _VOWELS)
    if len(token) >= 8 and vowel_count < 2:
        return True
    return False


def strip_url_tokens(text: str) -> str:
    """Remove noise tokens from text, preserving real words."""
    tokens = _TOKEN_RE.findall(text.lower())
    clean = [t for t in tokens if not _is_noise_token(t)]
    return " ".join(clean)


def clean_label_text(tokens: list, n_top: int = 3) -> str:
    """
    Given a list of candidate label tokens (e.g. from TF-IDF top terms),
    filter out noise and return the top n_top as a comma-separated label.
    """
    clean = [t for t in tokens if not _is_noise_token(t)]
    return ", ".join(clean[:n_top]) if clean else "unlabelled"
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_cluster_labels.py -v
```
Expected: 3 PASSED

**Step 5: Modify `scripts/1_cluster_embeddings.py` to use the token filter**

In `generate_cluster_labels()`, after extracting `top_terms`, add:

```python
# existing line:
top_terms = [feature_names[i] for i in top_indices]
# ADD: filter noise tokens before building label
from scripts.cluster_label_utils import clean_label_text
labels[cluster_id] = clean_label_text(top_terms, n_top=3)
```

Also verify the top of `main()` sets Numba threads explicitly (already present — just confirm):

```python
# Should already be there in the script. If not, add before any umap import:
import os
os.environ["NUMBA_NUM_THREADS"] = str(os.cpu_count() or 12)
```

And confirm HDBSCAN uses `core_dist_n_jobs=-1` (already set in config as `n_jobs: -1`).

**Step 6: Open your Mac Terminal (NOT Cursor) and run**

```bash
cd /Users/james/momentum
python scripts/1_cluster_embeddings.py --force
```

Expected output:
```
Cores: 12 available, using 12 | HDBSCAN: 12 job(s) | UMAP (Numba): 12 thread(s)
[Pipeline progress bar]
Saved 4XXXX rows to output/clusters.parquet
```

This takes ~15–25 minutes. The UMAP step (~8–20 min) is the longest.

**Step 7: Verify the output has clean labels**

```bash
python3 -c "
import pandas as pd
c = pd.read_parquet('output/clusters.parquet')
label_map = c.groupby('cluster')['cluster_label'].first()
for cid, label in sorted(label_map.items()):
    if cid != -1:
        print(f'{cid}: {label}')
"
```

Verify: no labels like `owq9fagevl` or `wgnfqclyr3`. All labels should be real English words.

**Step 8: Commit**

```bash
git add scripts/1_cluster_embeddings.py scripts/cluster_label_utils.py tests/test_cluster_labels.py
git commit -m "feat: filter URL tracking tokens from cluster labels, use all cores for HDBSCAN+UMAP"
```

---

## Task 3: Re-run Scripts 2, 3, 4 (temporal analysis + theme activation)

**Files:**
- Modify: `scripts/4_theme_activation_tfidf.py` — ensure `n_jobs` uses all cores
- Run Scripts 2, 3, 4 in sequence

**Context:** These scripts already work. After re-clustering, the cluster IDs have changed, so all downstream outputs need regenerating. Script 4 must score all 28 themes (currently the saved output only has themes 1–3, which was a partial run).

**Step 1: Verify Script 4 will score all 28 themes**

```bash
python3 -c "
import yaml
with open('theme_lexicon.yaml') as f:
    d = yaml.safe_load(f)
print('Themes in lexicon:', len(d['themes']))
print('IDs:', [t['id'] for t in d['themes']])
"
```
Expected: 28 themes, IDs 0–27.

The script reads the lexicon and scores all themes in it — so this will work automatically once we have the correct lexicon loaded. The previous run only produced 3 theme columns because the old activations file was from a partial run with an older lexicon version.

**Step 2: Run Script 2 (temporal analysis)**

```bash
python scripts/2_temporal_analysis.py --config config.yaml --force
```

Expected:
```
Loaded XXXXX rows from output/clusters.parquet
Analysing XX clusters with window='1ME'...
Saved temporal stats (XXXX rows) to output/temporal_stats.parquet
Saved XX events to output/temporal_events.parquet
Saved monthly frontend snapshots (XXXX rows) to output/cluster_trend_snapshots_1M.parquet
```

**Step 3: Run Script 4 (theme activation — all 28 themes, all cores)**

```bash
python scripts/4_theme_activation_tfidf.py --config config.yaml --mode windows --n-jobs 12 --force
```

Expected:
```
Loaded 28 themes from theme_lexicon.yaml
Scoring XXXX units...
Saved XXXX rows to output/theme_activations_windows.parquet
Avg activated themes per unit: X.XX
Done in XXs
```

Verify output has all 28 theme columns:

```bash
python3 -c "
import pandas as pd
ta = pd.read_parquet('output/theme_activations_windows.parquet')
score_cols = [c for c in ta.columns if c.endswith('_score')]
print(f'Theme score columns ({len(score_cols)}):', score_cols)
"
```
Expected: 28 columns (`theme_0_score` through `theme_27_score`).

**Step 4: Commit**

```bash
git add output/temporal_stats.parquet output/temporal_events.parquet \
        output/cluster_trend_snapshots_1M.parquet output/cluster_trend_snapshots_1M.csv \
        output/theme_activations_windows.parquet
git commit -m "data: regenerate temporal stats and theme activations on clean re-clustered data"
```

---

## Task 4: Build theme→cluster mapping (`scripts/4b_build_theme_cluster_map.py`)

**Files:**
- Create: `scripts/4b_build_theme_cluster_map.py`
- Create: `tests/test_theme_cluster_map.py`

**What it does:** For each of the 28 themes, find the HDBSCAN cluster that best represents it by computing the average theme activation score per cluster across all time windows. Outputs `output/theme_cluster_map.json`.

Format:
```json
{
  "0": {"cluster_id": 31, "cluster_label": "deepmind, google, ai", "avg_score": 0.412},
  "4": {"cluster_id": 33, "cluster_label": "alphafold, deepmind, protein", "avg_score": 0.891},
  ...
}
```

**Step 1: Write the failing tests**

```python
# tests/test_theme_cluster_map.py
import pandas as pd
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.build_theme_cluster_map import build_theme_cluster_map, validate_map

def _make_activations():
    # 3 themes, 3 clusters, 2 time windows each
    rows = []
    for tw in ["2024-01-31", "2024-02-29"]:
        rows += [
            {"cluster": 0, "cluster_label": "deepmind ai", "time_window": tw,
             "theme_0_score": 0.8, "theme_1_score": 0.1, "theme_2_score": 0.05},
            {"cluster": 1, "cluster_label": "protein folding", "time_window": tw,
             "theme_0_score": 0.1, "theme_1_score": 0.9, "theme_2_score": 0.05},
            {"cluster": 2, "cluster_label": "nhs data", "time_window": tw,
             "theme_0_score": 0.05, "theme_1_score": 0.05, "theme_2_score": 0.85},
        ]
    return pd.DataFrame(rows)

def test_build_theme_cluster_map_assigns_best_cluster():
    df = _make_activations()
    result = build_theme_cluster_map(df, theme_ids=[0, 1, 2])
    assert result[0]["cluster_id"] == 0
    assert result[1]["cluster_id"] == 1
    assert result[2]["cluster_id"] == 2

def test_build_theme_cluster_map_includes_avg_score():
    df = _make_activations()
    result = build_theme_cluster_map(df, theme_ids=[0])
    assert abs(result[0]["avg_score"] - 0.8) < 0.001

def test_validate_map_detects_unmatched_themes():
    mapping = {0: {"cluster_id": 5, "cluster_label": "x", "avg_score": 0.1}}
    # only 1 of 3 themes mapped — should warn but not raise
    warnings = validate_map(mapping, expected_theme_count=3)
    assert len(warnings) == 2  # themes 1 and 2 missing
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_theme_cluster_map.py -v
```
Expected: `ImportError`

**Step 3: Implement `scripts/4b_build_theme_cluster_map.py`**

```python
"""
Script 4b: Build theme→cluster mapping from theme activation scores.

For each theme in theme_lexicon.yaml, finds the HDBSCAN cluster
with the highest average activation score. Saves a JSON mapping
used by event_forecast.py to replace the broken theme_id==cluster_id assumption.

Usage:
  python scripts/4b_build_theme_cluster_map.py --config config.yaml [--force]
"""
import argparse
import json
import os
import sys

import pandas as pd
import yaml


def build_theme_cluster_map(activations_df, theme_ids):
    """
    For each theme_id, find the cluster with the highest mean score.
    Returns dict: {theme_id (int): {"cluster_id", "cluster_label", "avg_score"}}
    """
    mapping = {}
    for tid in theme_ids:
        score_col = f"theme_{tid}_score"
        if score_col not in activations_df.columns:
            continue
        # Mean score per cluster
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
    """Return list of warning strings for unmapped themes."""
    warnings = []
    for i in range(expected_theme_count):
        if i not in mapping:
            warnings.append(f"Theme {i} has no cluster match (no activation data)")
    return warnings


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
    ta_base = resolve(ta_cfg.get("output", "output/theme_activations.parquet"))
    ta_windows_path = ta_base.replace(".parquet", "_windows.parquet")
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
            "Run script 4 first: python scripts/4_theme_activation_tfidf.py --force"
        )

    print(f"Loading theme activations from {ta_windows_path}...")
    df = pd.read_parquet(ta_windows_path)

    mapping = build_theme_cluster_map(df, theme_ids)
    warnings = validate_map(mapping, expected_theme_count=len(themes))

    # Enrich with theme names for readability
    output = {}
    for tid in theme_ids:
        if tid in mapping:
            output[str(tid)] = {
                "theme_name": theme_names.get(tid, str(tid)),
                **mapping[tid],
            }
        else:
            output[str(tid)] = {
                "theme_name": theme_names.get(tid, str(tid)),
                "cluster_id": None,
                "cluster_label": None,
                "avg_score": 0.0,
            }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=True)

    print(f"\nTheme → Cluster Mapping ({len(mapping)}/{len(themes)} themes matched):")
    print(f"{'ID':>4} | {'Theme Name':<40} | {'Cluster ID':>10} | {'Cluster Label':<35} | Score")
    print("-" * 110)
    for tid in sorted(theme_ids):
        entry = output.get(str(tid), {})
        cid = entry.get("cluster_id", "—")
        clabel = entry.get("cluster_label", "—") or "—"
        score = entry.get("avg_score", 0.0)
        tname = entry.get("theme_name", "")
        print(f"{tid:>4} | {tname:<40} | {str(cid):>10} | {str(clabel):<35} | {score:.4f}")

    if warnings:
        print(f"\nWarnings ({len(warnings)}):")
        for w in warnings:
            print(f"  ⚠  {w}")

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_theme_cluster_map.py -v
```
Expected: 3 PASSED

**Step 5: Run the script**

```bash
python scripts/4b_build_theme_cluster_map.py --config config.yaml --force
```

Review the printed table. Verify that themes like "Protein Folding", "NHS Data", "AlphaFold", "Game-Playing AI" map to clusters with matching labels.

**Step 6: Commit**

```bash
git add scripts/4b_build_theme_cluster_map.py tests/test_theme_cluster_map.py output/theme_cluster_map.json
git commit -m "feat: build explicit theme→cluster mapping to fix broken theme_id==cluster_id assumption"
```

---

## Task 5: Update `event_forecast.py` to use the mapping

**Files:**
- Modify: `scripts/event_forecast.py`
- Create: `tests/test_event_forecast_mapping.py`

**What changes:** `estimate_event_direction()` currently looks up `profile_by_cluster.get(tid)` where `tid` is the theme ID. Replace with `profile_by_cluster.get(theme_cluster_map[tid])`.

**Step 1: Write the failing test**

```python
# tests/test_event_forecast_mapping.py
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.event_forecast import estimate_event_direction, build_cluster_profiles

def _make_profiles():
    # Theme 4 (Protein Folding) maps to cluster 33 in this test
    return pd.DataFrame([
        {"cluster": 33, "n_windows": 10, "avg_post_count": 100,
         "avg_volatility": 0.2, "avg_market_share": 0.05,
         "recent_growth": 0.3, "recent_momentum": 10.0,
         "recent_growth_lag1": 0.25, "recent_growth_lag2": 0.2,
         "recent_momentum_lag1": 8.0, "recent_momentum_lag2": 7.0,
         "trend_slope": 2.0, "direction_score": 0.3, "direction_label": "up"},
    ])

def test_estimate_uses_cluster_map_not_theme_id():
    """Without cluster map, theme 4 would look up cluster 4 (wrong).
    With map, it correctly uses cluster 33."""
    themes = [{
        "id": 4,
        "name": "Protein Folding",
        "keywords": ["alphafold", "protein", "folding"],
        "synonyms": [],
        "phrases": [],
    }]
    profiles = _make_profiles()
    theme_cluster_map = {"4": {"cluster_id": 33, "cluster_label": "alphafold", "avg_score": 0.8}}

    result = estimate_event_direction(
        event_text="AlphaFold predicts new protein structure",
        themes=themes,
        cluster_profiles_df=profiles,
        activation_threshold=1.0,
        theme_cluster_map=theme_cluster_map,
    )
    assert result["activated_theme_count"] == 1
    assert result["activated_themes"][0]["theme_id"] == 4
    assert result["activated_themes"][0]["mapped_cluster_id"] == 33
```

**Step 2: Run to verify it fails**

```bash
python -m pytest tests/test_event_forecast_mapping.py -v
```
Expected: FAIL — `estimate_event_direction` doesn't accept `theme_cluster_map` parameter yet.

**Step 3: Modify `scripts/event_forecast.py`**

In `estimate_event_direction()`, add `theme_cluster_map=None` parameter:

```python
def estimate_event_direction(
    event_text,
    themes,
    cluster_profiles_df,
    activation_threshold=0.03,
    per_theme_thresholds=None,
    llm_config=None,
    theme_cluster_map=None,   # ADD THIS
):
```

Then change the profile lookup block (currently `profile = profile_by_cluster.get(tid)`):

```python
    for theme in themes:
        tid = int(theme["id"])
        score = float(scores.get(tid, 0.0))
        threshold = float(
            thresholds.get(str(tid), thresholds.get(tid, activation_threshold))
        )
        if score < threshold:
            continue

        # Resolve which cluster this theme maps to
        if theme_cluster_map and str(tid) in theme_cluster_map:
            mapped = theme_cluster_map[str(tid)]
            mapped_cluster_id = mapped.get("cluster_id")
        else:
            mapped_cluster_id = tid  # fallback to old behaviour

        profile = profile_by_cluster.get(mapped_cluster_id)
        if not profile:
            continue

        activated.append({
            "theme_id": tid,
            "mapped_cluster_id": mapped_cluster_id,   # ADD for transparency
            ...
        })
```

**Step 4: Update `server.py` to load and pass `theme_cluster_map`**

In `server.py`, after loading THEMES, add:

```python
THEME_CLUSTER_MAP = {}
theme_cluster_map_path = BASE / "output" / "theme_cluster_map.json"
if theme_cluster_map_path.exists():
    with open(theme_cluster_map_path, encoding="utf-8") as f:
        THEME_CLUSTER_MAP = json.load(f)
```

Then in the `/api/predict` endpoint, pass it:

```python
prediction = estimate_event_direction(
    ...
    theme_cluster_map=THEME_CLUSTER_MAP,   # ADD
)
```

**Step 5: Run tests**

```bash
python -m pytest tests/test_event_forecast_mapping.py tests/test_event_forecast.py -v
```
Expected: all PASS

**Step 6: Commit**

```bash
git add scripts/event_forecast.py server.py tests/test_event_forecast_mapping.py
git commit -m "feat: use explicit theme→cluster map in event forecast, remove theme_id==cluster_id assumption"
```

---

## Task 6: Re-train the direction model (Script 6)

**Files:**
- Modify: `scripts/6_train_direction_model.py` — enable all cores on parallelisable models
- Run Script 6 with `--force`

**Context:** The model was trained on dirty temporal stats. Now that we have clean data and a new clustering, we retrain from scratch.

**Step 1: Verify and enable parallelism in model definitions**

In `scripts/6_train_direction_model.py`, in `model_candidates()`, ensure all tree models use `n_jobs=-1`:

```python
# RandomForest — add n_jobs=-1
out["random_forest"] = RandomForestClassifier(
    n_estimators=300,
    n_jobs=-1,          # ADD
    random_state=random_state,
    class_weight="balanced_subsample",
    min_samples_leaf=1,
)

# ExtraTrees — already has n_jobs=-1, confirm it's there
out["extra_trees"] = ExtraTreesClassifier(
    n_estimators=400,
    n_jobs=-1,          # already present
    ...
)

# HistGBM — add n_jobs=-1 (sklearn 1.4+ supports it)
out["hist_gbm"] = HistGradientBoostingClassifier(
    max_depth=8,
    learning_rate=0.05,
    max_iter=400,
    n_jobs=-1,          # ADD
    random_state=random_state,
)
```

**Step 2: Install xgboost and lightgbm for maximum model diversity**

```bash
pip install xgboost lightgbm
```

Verify:
```bash
python3 -c "import xgboost, lightgbm; print('xgb:', xgboost.__version__, 'lgbm:', lightgbm.__version__)"
```

**Step 3: Run Script 5 first (rebuild cluster state database)**

```bash
python scripts/5_event_direction_forecast.py --config config.yaml --force
```

**Step 4: Run Script 6 (train model)**

```bash
python scripts/6_train_direction_model.py --config config.yaml --force
```

Expected output ends with:
```
Best model: soft_voting_ensemble  (or the best single model)
Heuristic vs best: accuracy X.XXXX -> X.XXXX, macro_f1 X.XXXX -> X.XXXX
Saved model artifact to output/models/direction_model.pkl
```

**Step 5: Verify model metrics**

```bash
python3 -c "
import json
with open('output/models/direction_model_metrics.json') as f:
    m = json.load(f)
print('Best model:', m['best_model'])
print('Test accuracy:', m['models'][m['best_model']]['accuracy'])
print('Test macro_f1:', m['models'][m['best_model']]['macro_f1'])
print('vs heuristic accuracy:', m['models']['heuristic']['accuracy'])
"
```

**Step 6: Commit**

```bash
git add scripts/6_train_direction_model.py output/models/direction_model.pkl \
        output/models/direction_model_metrics.json
git commit -m "feat: retrain direction model on clean data with all cores, add xgboost/lightgbm"
```

---

## Task 7: Surface theme names in the dashboard

**Files:**
- Modify: `server.py` — enrich cluster labels with theme names from `theme_cluster_map.json`
- Modify: `frontend/src/lib/api.ts` — no changes needed (data already flows through)

**What changes:** The `/api/clusters` and `/api/snapshots` endpoints return `cluster_label` as raw TF-IDF keywords (e.g. "alphafold, deepmind, protein"). We enrich this with the theme name from the mapping (e.g. "AlphaFold Scientific Impact") when one exists.

**Step 1: Add a helper to `server.py`**

After loading `THEME_CLUSTER_MAP`, build a reverse lookup — cluster_id → theme_name:

```python
# Build reverse: cluster_id -> theme_name (for display enrichment)
CLUSTER_THEME_LABELS: dict[int, str] = {}
for theme_id_str, entry in THEME_CLUSTER_MAP.items():
    cid = entry.get("cluster_id")
    if cid is not None:
        theme_name = entry.get("theme_name", "")
        if theme_name:
            CLUSTER_THEME_LABELS[int(cid)] = theme_name
```

**Step 2: Enrich the `/api/clusters` response**

In `get_clusters()`, in the cluster dict construction, add `theme_name`:

```python
cid = int(row["cluster"])
clusters.append({
    "id": f"cluster-{cid}",
    "cluster": cid,
    "cluster_label": str(row.get("cluster_label", f"Cluster {cid}")),
    "theme_name": CLUSTER_THEME_LABELS.get(cid, ""),   # ADD: empty string if no mapping
    ...
})
```

**Step 3: Update the frontend cluster pill display**

In `frontend/src/pages/Index.tsx`, the monthly trends cluster picker currently shows `c.label` (raw TF-IDF label). Change it to prefer `theme_name` when available:

```typescript
// In clusterOptions useMemo, add theme_name alongside label
return displaySnapshots
  .filter(...)
  .map((s) => ({
    id: s.cluster_id,
    label: s.cluster_label,
    theme_name: (s as any).theme_name || "",   // ADD
  }));

// In the button render, show theme_name if available:
{c.theme_name || c.label}
```

Also update the `ClusterInfo` type in `frontend/src/lib/api.ts`:

```typescript
export interface ClusterInfo {
  id: string;
  cluster: number;
  cluster_label: string;
  theme_name: string;    // ADD
  size: number;
  // ... rest unchanged
}
```

**Step 4: Restart the server and verify**

```bash
uvicorn server:app --reload --port 8000
```

Open dashboard. The monthly trends cluster pills should now show theme names like "AlphaFold Scientific Impact" and "NHS Data Governance" instead of "alphafold, deepmind, protein".

**Step 5: Commit**

```bash
git add server.py frontend/src/lib/api.ts frontend/src/pages/Index.tsx
git commit -m "feat: surface theme names in dashboard cluster labels using theme→cluster mapping"
```

---

## Task 8: Run full pipeline validation

**Step 1: Run all tests**

```bash
python -m pytest tests/ -v
```
Expected: all PASS

**Step 2: Smoke test the API**

```bash
# Start server
uvicorn server:app --port 8000 &

# Check health
curl -s http://localhost:8000/api/health | python3 -m json.tool

# Check clusters
curl -s http://localhost:8000/api/clusters | python3 -c "
import json, sys
d = json.load(sys.stdin)
print('Total clusters:', d['total'])
for c in d['clusters'][:5]:
    print(f'  {c[\"cluster\"]}: theme={c.get(\"theme_name\",\"-\")} | label={c[\"cluster_label\"]}')
"

# Test a prediction
curl -s -X POST http://localhost:8000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"event_text": "AlphaFold discovers new protein structure for cancer treatment"}' \
  | python3 -c "
import json, sys
d = json.load(sys.stdin)
print('Direction:', d['predicted_direction'])
print('Confidence:', d['confidence'])
print('Activated themes:', [t['theme_name'] for t in d.get('activated_themes', [])])
"
```

**Step 3: Final commit**

```bash
git add -A
git commit -m "feat: complete clean-pipeline phase — spam removal, re-cluster, theme mapping, model retrain"
```

---

## Full Run Order (Summary)

```bash
# From your Mac Terminal (not Cursor)
cd /Users/james/momentum

# 0. Preprocess
python scripts/0_preprocess_data.py --force

# 1. Re-cluster (15-25 min, uses all 12 cores)
python scripts/1_cluster_embeddings.py --force

# 2. Temporal analysis
python scripts/2_temporal_analysis.py --force

# 4. Theme activation (all 28 themes, 12 cores)
python scripts/4_theme_activation_tfidf.py --mode windows --n-jobs 12 --force

# 4b. Build theme→cluster map
python scripts/4b_build_theme_cluster_map.py --force

# 5. Rebuild cluster state database
python scripts/5_event_direction_forecast.py --force

# 6. Retrain model (uses all cores)
python scripts/6_train_direction_model.py --force

# Start server
uvicorn server:app --reload --port 8000
```

Total time: ~20–30 minutes (dominated by UMAP in Script 1).

---

## Task 9: Weekly recency layer — Script 2 + API + dashboard toggle

**Files:**
- Modify: `scripts/2_temporal_analysis.py` — generate weekly snapshot for last 12 months
- Modify: `config.yaml` — add `recency_window` and `recency_lookback_months`
- Modify: `server.py` — add `/api/snapshots/weekly-recent` endpoint + density gating
- Modify: `frontend/src/lib/api.ts` — add `WeeklySnapshot` type + `fetchWeeklySnapshots()`
- Modify: `frontend/src/pages/Index.tsx` — add monthly/weekly toggle to trend chart
- Create: `tests/test_weekly_recency.py`

**Background — why weekly only for recent data:**
- Full dataset (2016–2026): median 0 posts/cluster/week — 86% of windows are zero. Useless for stats.
- Recent 12 months (2025–2026): ~980 posts/month total, ~11 posts/cluster/week for active clusters. Borderline but workable.
- Gate: only show weekly data for a cluster when ≥50% of its weekly windows in the recency period have ≥3 posts. Otherwise fall back to monthly silently.

---

### Step 1: Write the failing tests

```python
# tests/test_weekly_recency.py
import pandas as pd
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.temporal_recency import (
    build_weekly_recency_snapshots,
    is_cluster_weekly_dense_enough,
)

def _make_cluster_df():
    dates = pd.date_range("2025-01-06", periods=52, freq="W-MON")
    rows = []
    # cluster 1: posts every week (dense)
    for d in dates:
        for _ in range(5):
            rows.append({"timestamp": d, "cluster": 1, "cluster_label": "dense topic",
                         "likes": 2, "reach": 10, "shares": 1, "comments": 0,
                         "Author": "user_a"})
    # cluster 2: posts only 10 weeks out of 52 (sparse)
    for d in dates[:10]:
        rows.append({"timestamp": d, "cluster": 2, "cluster_label": "sparse topic",
                     "likes": 1, "reach": 5, "shares": 0, "comments": 0,
                     "Author": "user_b"})
    return pd.DataFrame(rows)

def test_dense_cluster_qualifies():
    df = _make_cluster_df()
    assert is_cluster_weekly_dense_enough(df, cluster_id=1, min_posts=3, min_density=0.5) is True

def test_sparse_cluster_does_not_qualify():
    df = _make_cluster_df()
    assert is_cluster_weekly_dense_enough(df, cluster_id=2, min_posts=3, min_density=0.5) is False

def test_build_weekly_recency_snapshots_only_includes_dense_clusters():
    df = _make_cluster_df()
    snaps = build_weekly_recency_snapshots(df, lookback_months=12, min_posts=3, min_density=0.5)
    cluster_ids = snaps["cluster"].unique()
    assert 1 in cluster_ids
    assert 2 not in cluster_ids

def test_build_weekly_recency_snapshots_has_required_columns():
    df = _make_cluster_df()
    snaps = build_weekly_recency_snapshots(df, lookback_months=12, min_posts=3, min_density=0.5)
    for col in ["time_window", "cluster", "cluster_label", "post_count",
                "market_share", "volume_pct_change", "volume_volatility",
                "momentum", "lifecycle_state", "anomaly_score", "window_type"]:
        assert col in snaps.columns, f"Missing column: {col}"
    assert (snaps["window_type"] == "1W").all()
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_weekly_recency.py -v
```
Expected: `ImportError: No module named 'scripts.temporal_recency'`

---

### Step 3: Create `scripts/temporal_recency.py`

```python
"""
Temporal recency utilities: weekly snapshots for the most recent N months.

Only clusters that pass a density gate are included — prevents the chart
showing noisy 0/0/spike/0 weekly patterns for thin clusters.
"""
import numpy as np
import pandas as pd


def is_cluster_weekly_dense_enough(
    df: pd.DataFrame,
    cluster_id: int,
    min_posts: int = 3,
    min_density: float = 0.5,
) -> bool:
    """
    Return True if cluster has enough weekly posts to be worth showing.

    Requires that at least `min_density` fraction of weeks (within the
    cluster's active period) contain >= `min_posts` posts.
    """
    cdf = df[df["cluster"] == cluster_id].copy()
    if cdf.empty:
        return False
    cdf["timestamp"] = pd.to_datetime(cdf["timestamp"], errors="coerce")
    cdf = cdf.dropna(subset=["timestamp"])
    weekly = cdf.set_index("timestamp").resample("1W")["cluster"].count()
    # Trim to active period
    active = weekly[weekly > 0]
    if active.empty:
        return False
    first, last = active.index.min(), active.index.max()
    active_range = weekly[(weekly.index >= first) & (weekly.index <= last)]
    dense_weeks = (active_range >= min_posts).sum()
    density = dense_weeks / max(len(active_range), 1)
    return bool(density >= min_density)


def build_weekly_recency_snapshots(
    df: pd.DataFrame,
    lookback_months: int = 12,
    min_posts: int = 3,
    min_density: float = 0.5,
) -> pd.DataFrame:
    """
    Build weekly snapshots for the most recent `lookback_months` of data,
    only for clusters that pass the density gate.

    Returns a DataFrame with the same schema as cluster_trend_snapshots_1M.csv
    plus a `window_type` column set to "1W".
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    cutoff = df["timestamp"].max() - pd.DateOffset(months=lookback_months)
    recent = df[df["timestamp"] >= cutoff].copy()

    # Identify dense-enough clusters using the recent slice
    dense_cluster_ids = [
        cid for cid in recent["cluster"].unique()
        if cid != -1
        and is_cluster_weekly_dense_enough(recent, cid, min_posts=min_posts, min_density=min_density)
    ]

    if not dense_cluster_ids:
        return pd.DataFrame(columns=[
            "time_window", "cluster", "cluster_label", "post_count",
            "market_share", "volume_pct_change", "volume_volatility",
            "momentum", "lifecycle_state", "anomaly_score", "window_type",
        ])

    records = []
    for cid in dense_cluster_ids:
        cdf = recent[recent["cluster"] == cid].copy()
        label = cdf["cluster_label"].iloc[0] if "cluster_label" in cdf.columns else str(cid)

        resampled = cdf.set_index("timestamp").resample("1W").agg(
            post_count=("cluster", "count"),
        ).fillna(0)

        resampled["cluster"] = cid
        resampled["cluster_label"] = label
        resampled["volume_pct_change"] = resampled["post_count"].pct_change().fillna(0).clip(-10, 10)
        resampled["volume_volatility"] = (
            resampled["post_count"].rolling(3, min_periods=1).std().fillna(0)
        )
        rolling_mean = resampled["post_count"].rolling(3, min_periods=1).mean()
        resampled["momentum"] = resampled["post_count"] - rolling_mean

        mean_vol = resampled["post_count"].mean()
        std_vol = resampled["post_count"].std()
        resampled["anomaly_score"] = (
            (resampled["post_count"] - mean_vol) / (std_vol if std_vol > 0 else 1)
        )

        # Lifecycle based on last 3 weeks
        recent_growth = resampled["volume_pct_change"].tail(3).mean()
        recent_vol = resampled["post_count"].tail(3).mean()
        overall_vol = resampled["post_count"].mean()
        if overall_vol == 0 or recent_vol < overall_vol * 0.2:
            lifecycle = "dormant"
        elif recent_growth > 0.5:
            lifecycle = "emerging"
        elif recent_growth > 0.1:
            lifecycle = "trending"
        elif recent_growth < -0.3:
            lifecycle = "declining"
        else:
            lifecycle = "stable"
        resampled["lifecycle_state"] = lifecycle

        records.append(resampled)

    result = pd.concat(records).reset_index()
    result = result.rename(columns={"timestamp": "time_window"})

    # Market share across all dense clusters per window
    total_per_window = result.groupby("time_window")["post_count"].transform("sum")
    result["market_share"] = result["post_count"] / total_per_window.replace(0, np.nan)

    result["window_type"] = "1W"
    result["time_window"] = pd.to_datetime(result["time_window"])

    columns = [
        "time_window", "cluster", "cluster_label", "post_count",
        "market_share", "volume_pct_change", "volume_volatility",
        "momentum", "lifecycle_state", "anomaly_score", "window_type",
    ]
    return result[columns].sort_values(["time_window", "cluster"]).reset_index(drop=True)
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_weekly_recency.py -v
```
Expected: 4 PASSED

---

### Step 5: Add recency config to `config.yaml`

Add under the existing `temporal:` block:

```yaml
temporal:
  default_window: "1M"
  recency_window: "1W"
  recency_lookback_months: 12
  recency_min_posts_per_window: 3
  recency_min_density: 0.5
```

Add to `output:` block:

```yaml
output:
  # ... existing outputs ...
  cluster_trend_snapshots_1w_recent: "output/cluster_trend_snapshots_1W_recent.parquet"
  cluster_trend_snapshots_1w_recent_csv: "output/cluster_trend_snapshots_1W_recent.csv"
```

---

### Step 6: Modify `scripts/2_temporal_analysis.py` to generate weekly snapshot

At the end of `main()`, after the existing monthly snapshot save, add:

```python
# --- Weekly recency snapshot ---
from scripts.temporal_recency import build_weekly_recency_snapshots

recency_cfg = config.get("temporal", {})
lookback = int(recency_cfg.get("recency_lookback_months", 12))
min_posts = int(recency_cfg.get("recency_min_posts_per_window", 3))
min_density = float(recency_cfg.get("recency_min_density", 0.5))

weekly_recency = build_weekly_recency_snapshots(
    df,
    lookback_months=lookback,
    min_posts=min_posts,
    min_density=min_density,
)

weekly_recency_parquet = resolve(
    config["output"].get(
        "cluster_trend_snapshots_1w_recent",
        "output/cluster_trend_snapshots_1W_recent.parquet",
    )
)
weekly_recency_csv = resolve(
    config["output"].get(
        "cluster_trend_snapshots_1w_recent_csv",
        "output/cluster_trend_snapshots_1W_recent.csv",
    )
)
weekly_recency.to_parquet(weekly_recency_parquet, index=False)
weekly_recency.to_csv(weekly_recency_csv, index=False)

dense_clusters = weekly_recency["cluster"].nunique()
print(f"Saved weekly recency snapshots ({len(weekly_recency)} rows, "
      f"{dense_clusters} dense clusters) to {weekly_recency_parquet}")
```

**Run Script 2 again to regenerate with the weekly output:**

```bash
python scripts/2_temporal_analysis.py --config config.yaml --force
```

Expected: existing monthly outputs regenerate + new weekly recency files appear.

Verify:
```bash
python3 -c "
import pandas as pd
w = pd.read_csv('output/cluster_trend_snapshots_1W_recent.csv')
print('Weekly recency shape:', w.shape)
print('Dense clusters:', w['cluster'].nunique())
print('Date range:', w['time_window'].min(), '->', w['time_window'].max())
print('Clusters:', w.groupby('cluster')['cluster_label'].first().to_dict())
"
```

---

### Step 7: Add `/api/snapshots/weekly-recent` endpoint to `server.py`

After loading `SNAPSHOTS_CSV_PATH`, add:

```python
WEEKLY_RECENT_CSV_PATH = _resolve(
    CONFIG["output"].get(
        "cluster_trend_snapshots_1w_recent_csv",
        "output/cluster_trend_snapshots_1W_recent.csv",
    )
)
```

Add the new endpoint after the existing `/api/snapshots`:

```python
@app.get("/api/snapshots/weekly-recent")
def get_weekly_recent_snapshots():
    """Weekly snapshots for the last 12 months, dense clusters only."""
    if not WEEKLY_RECENT_CSV_PATH.exists():
        return []
    df = pd.read_csv(str(WEEKLY_RECENT_CSV_PATH), low_memory=False)

    snapshots = []
    for _, row in df.iterrows():
        snapshots.append({
            "cluster_id": f"cluster-{int(row['cluster'])}",
            "cluster_label": str(row.get("cluster_label", f"Cluster {int(row['cluster'])}")),
            "time_window": str(row.get("time_window", "")),
            "post_count": int(row.get("post_count", 0)),
            "market_share": float(row.get("market_share", 0) * 100),
            "momentum": float(row.get("momentum", 0)),
            "volatility": float(row.get("volume_volatility", 0)),
            "window_type": "1W",
        })
    return _sanitize(snapshots)
```

Also add a `weekly_dense_clusters` field to the existing `/api/health` response so the frontend knows upfront which clusters have weekly data:

```python
@app.get("/api/health")
def health():
    weekly_clusters = 0
    if WEEKLY_RECENT_CSV_PATH.exists():
        wdf = pd.read_csv(str(WEEKLY_RECENT_CSV_PATH), low_memory=False)
        weekly_clusters = int(wdf["cluster"].nunique()) if "cluster" in wdf.columns else 0
    return {
        "status": "ok",
        "model_loaded": model_bundle is not None,
        "themes": len(THEMES),
        "clusters": len(profiles_df),
        "weekly_dense_clusters": weekly_clusters,
    }
```

---

### Step 8: Add types and fetch function to `frontend/src/lib/api.ts`

Add `theme_name` to `ClusterInfo` and `Snapshot`, and add the new fetch:

```typescript
export interface ClusterInfo {
  id: string;
  cluster: number;
  cluster_label: string;
  theme_name: string;        // ADD
  size: number;
  market_share: number;
  volume_pct_change: number;
  volume_volatility: number;
  momentum: number;
  lifecycle: string;
  anomaly_score: number;
}

export interface Snapshot {
  cluster_id: string;
  cluster_label: string;
  time_window: string;
  post_count: number;
  market_share: number;
  momentum: number;
  volatility: number;
  window_type?: "1M" | "1W";  // ADD
}

export async function fetchWeeklySnapshots(): Promise<Snapshot[]> {
  return request("/snapshots/weekly-recent");
}
```

---

### Step 9: Update `frontend/src/pages/Index.tsx` — add monthly/weekly toggle

**What changes:**
- Add a `granularity` state: `"monthly" | "weekly"`
- When `"weekly"` selected, use `fetchWeeklySnapshots()` data instead of `displaySnapshots`
- Show the toggle only when weekly data is available (non-empty)
- For clusters that have no weekly data, the toggle stays on monthly and the button is greyed out

```typescript
// Add state
const [granularity, setGranularity] = useState<"monthly" | "weekly">("monthly");
const [weeklySnapshots, setWeeklySnapshots] = useState<Snapshot[]>([]);

// Add to the existing useEffect fetch block:
fetchWeeklySnapshots().then(setWeeklySnapshots).catch(() => {}),

// Derived: which clusters have weekly data
const weeklyClusterIds = useMemo(
  () => new Set(weeklySnapshots.map((s) => s.cluster_id)),
  [weeklySnapshots]
);
const hasWeeklyData = weeklySnapshots.length > 0;

// Replace displaySnapshots with granularity-aware source in trendData:
const activeSnapshots = granularity === "weekly" && hasWeeklyData
  ? weeklySnapshots
  : displaySnapshots;

// trendData useMemo uses activeSnapshots instead of displaySnapshots
const trendData = useMemo(() => {
  const ids = selectedClusterIds.length
    ? selectedClusterIds
    : clusterOptions.slice(0, 3).map((c) => c.id);
  const timeWindows = [...new Set(activeSnapshots.map((s) => s.time_window))].sort();
  return timeWindows.map((tw) => {
    const row: Record<string, string | number> = {
      time_window: granularity === "weekly" ? tw.slice(0, 10) : tw.slice(0, 7),
    };
    ids.forEach((id) => {
      const snap = activeSnapshots.find((s) => s.cluster_id === id && s.time_window === tw);
      if (snap) row[id] = snap[selectedMetric] ?? 0;
    });
    return row;
  });
}, [selectedClusterIds, selectedMetric, clusterOptions, activeSnapshots, granularity]);
```

**Add the toggle UI** in the Monthly Trends card header, next to the metric buttons:

```tsx
{/* Granularity toggle — only shown when weekly data exists */}
{hasWeeklyData && (
  <div className="flex items-center gap-1 border border-border rounded p-0.5">
    <button
      onClick={() => setGranularity("monthly")}
      className={cn(
        "px-2.5 py-1 rounded text-[10px] font-medium transition-colors",
        granularity === "monthly"
          ? "bg-foreground text-background"
          : "text-muted-foreground hover:text-foreground"
      )}
    >
      Monthly
    </button>
    <button
      onClick={() => setGranularity("weekly")}
      className={cn(
        "px-2.5 py-1 rounded text-[10px] font-medium transition-colors",
        granularity === "weekly"
          ? "bg-foreground text-background"
          : "text-muted-foreground hover:text-foreground"
      )}
    >
      Weekly (recent)
    </button>
  </div>
)}
```

**Add a subtitle update** to show context:

```tsx
<p className="mt-1 text-[11px] text-muted-foreground">
  {granularity === "weekly"
    ? `Weekly · last 12 months · ${weeklyClusterIds.size} dense clusters`
    : `Monthly · full history${useRealData ? " (live)" : ""}`}
</p>
```

**Filter cluster pills** when in weekly mode to only show clusters that have weekly data:

```tsx
const activeClusterOptions = useMemo(() => {
  if (granularity === "weekly" && hasWeeklyData) {
    return clusterOptions.filter((c) => weeklyClusterIds.has(c.id));
  }
  return clusterOptions;
}, [clusterOptions, granularity, hasWeeklyData, weeklyClusterIds]);
```

Then replace all `clusterOptions` references in the pill map and `activeClusterIds` derivation with `activeClusterOptions`.

---

### Step 10: Run all tests + verify end-to-end

```bash
python -m pytest tests/ -v
```
Expected: all PASS including the 4 new weekly recency tests.

Start server and open dashboard:

```bash
uvicorn server:app --reload --port 8000
```

Verify:
1. Monthly Trends shows monthly data by default — full 2016–2026 range
2. "Weekly (recent)" toggle appears when `output/cluster_trend_snapshots_1W_recent.csv` exists
3. Switching to weekly shows only the last 12 months, at weekly resolution
4. Cluster pills in weekly mode only show clusters with sufficient density
5. Switching back to monthly restores the full history view

**Step 11: Commit**

```bash
git add scripts/temporal_recency.py tests/test_weekly_recency.py \
        scripts/2_temporal_analysis.py server.py config.yaml \
        frontend/src/lib/api.ts frontend/src/pages/Index.tsx \
        output/cluster_trend_snapshots_1W_recent.parquet \
        output/cluster_trend_snapshots_1W_recent.csv
git commit -m "feat: add weekly recency layer — density-gated weekly snapshots for last 12 months with dashboard toggle"
```

---

## Updated Full Run Order (Summary)

```bash
# From your Mac Terminal (not Cursor)
cd /Users/james/momentum

# 0. Preprocess (remove spam, cap authors)
python scripts/0_preprocess_data.py --force

# 1. Re-cluster — uses all 12 cores, ~15-25 min
python scripts/1_cluster_embeddings.py --force

# 2. Temporal analysis — generates both monthly AND weekly recency snapshots
python scripts/2_temporal_analysis.py --force

# 4. Theme activation — all 28 themes, 12 cores
python scripts/4_theme_activation_tfidf.py --mode windows --n-jobs 12 --force

# 4b. Build theme→cluster map
python scripts/4b_build_theme_cluster_map.py --force

# 5. Rebuild cluster state database
python scripts/5_event_direction_forecast.py --force

# 6. Retrain model
python scripts/6_train_direction_model.py --force

# Start server
uvicorn server:app --reload --port 8000
```

Total time: ~20–30 minutes (dominated by UMAP in Script 1).
