# Branch 2: Forecast Engine — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Branch:** `feature/forecast-engine`
**Depends on:** `feature/data-pipeline` merged into `main` first.

Before starting, rebase onto main to get the fresh `output/theme_activations_windows.parquet` with all 28 themes:
```bash
git fetch origin
git rebase main
```

**Goal:** Fix the broken `theme_id == cluster_id` assumption in the forecast engine. Build an explicit theme→cluster mapping from activation scores. Update `event_forecast.py` and `server.py` to use it. Retrain the direction model on clean data with all 12 cores.

**Architecture:** New `scripts/4b_build_theme_cluster_map.py` reads `theme_activations_windows.parquet` and finds the best-matching cluster for each theme. `event_forecast.py` gets a `theme_cluster_map` parameter. `server.py` loads the map at startup and passes it to all predictions. Script 6 retrains with `n_jobs=-1` on all tree models + xgboost/lightgbm.

**Tech Stack:** Python 3.13, pandas, scikit-learn, joblib, xgboost, lightgbm, FastAPI.

**Outputs produced (consumed by branch 3):**
- `output/theme_cluster_map.json` — explicit theme→cluster mapping
- `output/models/direction_model.pkl` — retrained model on clean data
- `output/models/direction_model_metrics.json` — model quality report

---

## Task 4: Build theme→cluster mapping (`scripts/4b_build_theme_cluster_map.py`)

**Files:**
- Create: `scripts/4b_build_theme_cluster_map.py`
- Create: `tests/test_theme_cluster_map.py`

**Why this matters:** Currently `event_forecast.py` assumes theme 4 (Protein Folding) corresponds to cluster 4. After re-clustering, cluster 4 is a random trading/sports cluster. This script finds which HDBSCAN cluster actually best represents each theme, by reading the TF-IDF activation scores computed in Script 4.

**Step 1: Write the failing tests**

```python
# tests/test_theme_cluster_map.py
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.build_theme_cluster_map import build_theme_cluster_map, validate_map

def _make_activations():
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
    warnings = validate_map(mapping, expected_theme_count=3)
    assert len(warnings) == 2
```

**Step 2: Run tests to verify they fail**

```bash
cd /Users/james/momentum
python -m pytest tests/test_theme_cluster_map.py -v
```
Expected: `ImportError`

**Step 3: Implement `scripts/4b_build_theme_cluster_map.py`**

```python
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

Inspect the table. Themes like "Protein Folding", "NHS Data Governance", "Game-Playing Superhuman AI", "AlphaFold Scientific Impact" should map to clusters with matching keywords.

**Step 6: Commit**

```bash
git add scripts/4b_build_theme_cluster_map.py tests/test_theme_cluster_map.py \
        output/theme_cluster_map.json
git commit -m "feat: build explicit theme→cluster mapping — fixes broken theme_id==cluster_id assumption"
```

---

## Task 5: Update `event_forecast.py` and `server.py` to use the mapping

**Files:**
- Modify: `scripts/event_forecast.py`
- Modify: `server.py`
- Create: `tests/test_event_forecast_mapping.py`

**Step 1: Write the failing test**

```python
# tests/test_event_forecast_mapping.py
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.event_forecast import estimate_event_direction

def _make_profiles():
    # Theme 4 maps to cluster 33 via the map — cluster 4 has no entry
    return pd.DataFrame([{
        "cluster": 33, "n_windows": 10, "avg_post_count": 100,
        "avg_volatility": 0.2, "avg_market_share": 0.05,
        "recent_growth": 0.3, "recent_momentum": 10.0,
        "recent_growth_lag1": 0.25, "recent_growth_lag2": 0.2,
        "recent_momentum_lag1": 8.0, "recent_momentum_lag2": 7.0,
        "trend_slope": 2.0, "direction_score": 0.3, "direction_label": "up",
    }])

def test_uses_cluster_map_not_theme_id():
    """Without map, theme 4 looks up cluster 4 (missing) → no activation.
    With map, it correctly uses cluster 33 → activates."""
    themes = [{
        "id": 4, "name": "Protein Folding",
        "keywords": ["alphafold", "protein", "folding"],
        "synonyms": [], "phrases": [],
    }]
    profiles = _make_profiles()
    theme_cluster_map = {
        "4": {"cluster_id": 33, "cluster_label": "alphafold", "avg_score": 0.8}
    }
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

def test_falls_back_to_theme_id_when_no_map():
    """Without a map, behaviour is unchanged (uses tid as cluster id)."""
    themes = [{
        "id": 0, "name": "Test",
        "keywords": ["test", "example"],
        "synonyms": [], "phrases": [],
    }]
    profiles = pd.DataFrame([{
        "cluster": 0, "n_windows": 5, "avg_post_count": 50,
        "avg_volatility": 0.1, "avg_market_share": 0.02,
        "recent_growth": 0.1, "recent_momentum": 2.0,
        "recent_growth_lag1": 0.1, "recent_growth_lag2": 0.1,
        "recent_momentum_lag1": 2.0, "recent_momentum_lag2": 2.0,
        "trend_slope": 0.5, "direction_score": 0.1, "direction_label": "flat",
    }])
    result = estimate_event_direction(
        event_text="test example content",
        themes=themes,
        cluster_profiles_df=profiles,
        activation_threshold=0.1,
        theme_cluster_map=None,
    )
    assert result["activated_theme_count"] == 1
    assert result["activated_themes"][0]["mapped_cluster_id"] == 0
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_event_forecast_mapping.py -v
```
Expected: FAIL — `estimate_event_direction` doesn't accept `theme_cluster_map` yet.

**Step 3: Modify `scripts/event_forecast.py`**

Add `theme_cluster_map=None` to `estimate_event_direction()` signature:

```python
def estimate_event_direction(
    event_text,
    themes,
    cluster_profiles_df,
    activation_threshold=0.03,
    per_theme_thresholds=None,
    llm_config=None,
    theme_cluster_map=None,    # ADD
):
```

In the `for theme in themes:` loop, replace:
```python
profile = profile_by_cluster.get(tid)
if not profile:
    continue
activated.append({
    "theme_id": tid,
    ...
})
```
With:
```python
# Resolve cluster via explicit map, fall back to tid
if theme_cluster_map and str(tid) in theme_cluster_map:
    mapped_cluster_id = theme_cluster_map[str(tid)].get("cluster_id", tid)
else:
    mapped_cluster_id = tid

profile = profile_by_cluster.get(mapped_cluster_id)
if not profile:
    continue
activated.append({
    "theme_id": tid,
    "mapped_cluster_id": mapped_cluster_id,    # ADD for transparency
    "theme_name": theme.get("name", str(tid)),
    "score": score,
    "direction_score": float(profile["direction_score"]),
    "direction_label": profile["direction_label"],
    "avg_volatility": float(profile["avg_volatility"]),
    "avg_market_share": float(profile.get("avg_market_share", 0.0)),
    "recent_momentum": float(profile.get("recent_momentum", 0.0)),
    "recent_growth_lag1": float(profile.get("recent_growth_lag1", 0.0)),
    "recent_growth_lag2": float(profile.get("recent_growth_lag2", 0.0)),
    "recent_momentum_lag1": float(profile.get("recent_momentum_lag1", 0.0)),
    "recent_momentum_lag2": float(profile.get("recent_momentum_lag2", 0.0)),
    "threshold": threshold,
})
```

**Step 4: Modify `server.py` — load map + pass to predict**

After the `THEMES` loading block, add:

```python
THEME_CLUSTER_MAP: dict = {}
_theme_map_path = BASE / "output" / "theme_cluster_map.json"
if _theme_map_path.exists():
    with open(_theme_map_path, encoding="utf-8") as _f:
        THEME_CLUSTER_MAP = json.load(_f)

# Reverse: cluster_id (int) → theme_name (str) — used for label enrichment
CLUSTER_THEME_LABELS: dict[int, str] = {
    int(v["cluster_id"]): v["theme_name"]
    for v in THEME_CLUSTER_MAP.values()
    if v.get("cluster_id") is not None and v.get("theme_name")
}
```

In the `/api/predict` endpoint, add `theme_cluster_map=THEME_CLUSTER_MAP` to the `estimate_event_direction()` call:

```python
prediction = estimate_event_direction(
    event_text=req.event_text,
    themes=THEMES,
    cluster_profiles_df=profiles_df,
    activation_threshold=threshold,
    per_theme_thresholds=per_theme_thresholds,
    llm_config=LLM_CFG,
    theme_cluster_map=THEME_CLUSTER_MAP,    # ADD
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
git commit -m "feat: use explicit theme→cluster map in forecast — fixes wrong cluster profile lookups"
```

---

## Task 6: Retrain the direction model (Script 6)

**Files:**
- Modify: `scripts/6_train_direction_model.py` — enable `n_jobs=-1` on all tree models
- Run: Script 5 then Script 6

**Step 1: Enable parallelism in `model_candidates()`**

In `scripts/6_train_direction_model.py`, find the `model_candidates()` function and make these changes:

```python
# RandomForest — add n_jobs=-1
out["random_forest"] = RandomForestClassifier(
    n_estimators=300,
    n_jobs=-1,                      # ADD
    random_state=random_state,
    class_weight="balanced_subsample",
    min_samples_leaf=1,
)

# HistGBM — add n_jobs=-1 (sklearn 1.4+)
out["hist_gbm"] = HistGradientBoostingClassifier(
    max_depth=8,
    learning_rate=0.05,
    max_iter=400,
    n_jobs=-1,                      # ADD
    random_state=random_state,
)

# XGBoost — add nthread=-1
out["xgboost"] = XGBClassifier(
    objective="multi:softprob",
    num_class=len(DIRECTION_LABELS),
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=random_state,
    eval_metric="mlogloss",
    nthread=-1,                     # ADD
)

# LightGBM — add n_jobs=-1
out["lightgbm"] = LGBMClassifier(
    objective="multiclass",
    num_class=len(DIRECTION_LABELS),
    n_estimators=220,
    learning_rate=0.05,
    random_state=random_state,
    n_jobs=-1,                      # ADD
    verbosity=-1,
)
```

ExtraTrees already has `n_jobs=-1` — confirm it's present.

**Step 2: Install xgboost and lightgbm**

```bash
pip install xgboost lightgbm
python3 -c "import xgboost, lightgbm; print('xgb:', xgboost.__version__, 'lgbm:', lightgbm.__version__)"
```

**Step 3: Commit code changes**

```bash
git add scripts/6_train_direction_model.py
git commit -m "feat: enable n_jobs=-1 on all tree models, add xgboost/lightgbm for model diversity"
```

**Step 4: Run Script 5 (rebuild cluster state database)**

```bash
python scripts/5_event_direction_forecast.py --config config.yaml --force
```

**Step 5: Run Script 6 (train)**

```bash
python scripts/6_train_direction_model.py --config config.yaml --force
```

Expected output:
```
Best model: soft_voting_ensemble
Heuristic vs best: accuracy X.XXXX -> X.XXXX, macro_f1 X.XXXX -> X.XXXX
Saved model artifact to output/models/direction_model.pkl
```

**Step 6: Verify metrics**

```bash
python3 -c "
import json
with open('output/models/direction_model_metrics.json') as f:
    m = json.load(f)
best = m['best_model']
print('Best model:', best)
print('Test accuracy:', round(m['models'][best]['accuracy'], 4))
print('Test macro_f1:', round(m['models'][best]['macro_f1'], 4))
print('Heuristic accuracy:', round(m['models']['heuristic']['accuracy'], 4))
print('Heuristic macro_f1:', round(m['models']['heuristic']['macro_f1'], 4))
"
```

**Step 7: Run all tests**

```bash
python -m pytest tests/ -v
```
Expected: all PASS

**Step 8: Commit model outputs**

```bash
git add output/models/direction_model.pkl output/models/direction_model_metrics.json \
        output/cluster_state_database.parquet
git commit -m "data: retrain direction model on clean data — all cores, xgboost+lightgbm included"
```

---

## Merge Signal

All tests pass, `output/theme_cluster_map.json` exists with 28 entries, and `output/models/direction_model.pkl` is fresh. Merge into `main`, then branch 3 can rebase and use `THEME_CLUSTER_MAP` + `CLUSTER_THEME_LABELS` from `server.py`.

```bash
git checkout main
git merge feature/forecast-engine
```
