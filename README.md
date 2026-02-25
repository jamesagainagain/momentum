# InsightAgent — Temporal Topic Analysis Pipeline

Analyses social media posts with embeddings (e.g. Gemini) to discover topics, track their evolution over time, and predict direction (down / flat / up) for new events.

## Setup

```bash
pip install -r requirements.txt
```

**Inputs** (place in project root or paths in `config.yaml`):

- `embeddings.npy` — embedding vectors (rows aligned to raw CSV)
- `clustered_data_labeled.csv` — raw posts (text, timestamp, Author, engagement columns)

Optional: run **Script 0** first to clean the raw CSV; Script 1 will then use `output/data_clean.csv` and `output/data_clean_indices.npy` so clustering runs on deduplicated, author-capped data.

---

## Pipeline Overview

Run scripts in order. Each step caches its output; use `--force` where supported to rerun.

| Step | Script | Purpose |
|------|--------|---------|
| 0 | `0_preprocess_data.py` | (Optional) Clean raw CSV: dedupe, cap author dominance |
| 1 | `1_cluster_embeddings.py` | HDBSCAN clustering + UMAP 2D + TF-IDF cluster labels |
| 2 | `2_temporal_analysis.py` | Per-cluster temporal metrics, events, lifecycle, monthly + weekly snapshots |
| 3 | `3_visualize_clusters.py` | Interactive HTML: cluster map, heatmap, event timeline, volatility dashboard |
| 4 | `4_theme_activation_tfidf.py` | Theme activations from lexicon + TF-IDF (windows and/or events) |
| 4b | `4b_build_theme_cluster_map.py` | Map each theme to best-matching cluster → `theme_cluster_map.json` |
| 5 | `5_event_direction_forecast.py` | Build cluster state DB + predict direction for an event text |
| 6 | `6_train_direction_model.py` | Train direction model (walk-forward CV, stacking ensemble), save best artifact |
| 6b | `6b_visualize_forecast.py` | Standalone HTML dashboard for latest forecast |

Design rationale and research citations for clustering, temporal analysis, and preprocessing are in `docs/plans/2026-02-25-analysis-research-citations.md`.

---

## Pipeline Steps (detail)

### 0. Preprocess data (optional)

Deduplicate (text + author), cap single-author share, and write a cleaned CSV plus indices for slicing embeddings.

```bash
python scripts/0_preprocess_data.py --config config.yaml
python scripts/0_preprocess_data.py --config config.yaml --force   # rerun
```

**Outputs:**

- `output/data_clean.csv` — cleaned posts (aligned to config `input.data`)
- `output/data_clean_indices.npy` — row indices kept (for slicing `embeddings.npy`)
- `output/preprocessing_report.json` — row counts, duplicates removed, author-cap removed

If these exist, **Script 1** uses them; otherwise it uses the raw CSV and full embeddings.

---

### 1. Cluster embeddings

HDBSCAN clustering, UMAP 2D reduction, and TF-IDF-based cluster labels (with optional URL-token filtering via `cluster_label_utils`).

```bash
python scripts/1_cluster_embeddings.py --config config.yaml
python scripts/1_cluster_embeddings.py --config config.yaml --force   # rerun
```

**Outputs:**

- `output/clusters.parquet` — columns: cluster, umap_x, umap_y, cluster_label, cluster_size, text, timestamp, etc.

**Config:** `clustering.*`, `umap.*`, `pca.*` (optional PCA before HDBSCAN), `max_cores`.

---

### 2. Temporal analysis

Per-cluster, per-window metrics (volume, volatility, momentum, market share), anomaly/event detection, lifecycle classification, and frontend-ready snapshots.

```bash
python scripts/2_temporal_analysis.py                      # monthly (default)
python scripts/2_temporal_analysis.py --window "1W"        # weekly
python scripts/2_temporal_analysis.py --window "1D"      # daily
python scripts/2_temporal_analysis.py --window "3M"      # quarterly
```

**Outputs:**

- `output/temporal_stats.parquet` — per cluster × time_window metrics
- `output/temporal_events.parquet` — detected spikes/drops (anomaly events)
- `output/cluster_trend_snapshots_1M.parquet` / `.csv` — monthly snapshots
- `output/cluster_trend_snapshots_1W_recent.parquet` / `.csv` — weekly recency snapshots (dense clusters only)

**Config:** `temporal.*`, `output.cluster_trend_snapshots_*`.

---

### 3. Visualizations

Standalone HTML files (no server required to view).

```bash
python scripts/3_visualize_clusters.py
python scripts/3_visualize_clusters.py --open-browser      # auto-open in browser
```

**Outputs:**

- `output/visualizations/cluster_map.html` — UMAP scatter, colour by cluster
- `output/visualizations/temporal_heatmap.html`
- `output/visualizations/event_timeline.html`
- `output/visualizations/volatility_dashboard.html`

---

### 4. Theme activation (TF-IDF, no LLM)

Score clusters (and optionally events) against a curated theme lexicon; produces activation tables for model training and API.

```bash
python scripts/4_theme_activation_tfidf.py --config config.yaml --mode windows
python scripts/4_theme_activation_tfidf.py --config config.yaml --mode events
python scripts/4_theme_activation_tfidf.py --config config.yaml --mode windows --n-jobs 8 --force
```

**Outputs:**

- `output/theme_activations_windows.parquet` — theme scores per cluster × time window
- `output/theme_activations_events.parquet` — when using `--mode events`

**Config:** `theme_activation.*` (lexicon path, thresholds, TF-IDF params). Lexicon: `theme_lexicon.yaml`.

---

### 4b. Theme–cluster map

Build a mapping from each theme to its best-matching cluster (for API/labelling).

```bash
python scripts/4b_build_theme_cluster_map.py --config config.yaml
python scripts/4b_build_theme_cluster_map.py --config config.yaml --force
```

**Outputs:**

- `output/theme_cluster_map.json` — theme_id → { theme_name, cluster_id, … }

---

### 5. Event direction forecast

Build the per-cluster state database (volatility, direction signals) and optionally run a prediction for a given event text.

```bash
python scripts/5_event_direction_forecast.py --config config.yaml
python scripts/5_event_direction_forecast.py --config config.yaml --event-text "Google launches new Gemini robotics model"
python scripts/5_event_direction_forecast.py --config config.yaml --event-text "..." --model-path output/models/direction_model.pkl --force
```

**Outputs:**

- `output/cluster_state_database.parquet` — historical cluster/window state for the model
- `output/event_direction_prediction.json` — when `--event-text` is provided

**Config:** `event_forecast.*` (thresholds, risk bands, paths).

---

### 6. Train direction model

Walk-forward CV over epsilon and models, threshold tuning, optional stacking ensemble and feature pruning. Saves the best model and full metrics (including overfitting diagnostics).

```bash
python scripts/6_train_direction_model.py --config config.yaml
python scripts/6_train_direction_model.py --config config.yaml --force   # retrain
```

**Outputs:**

- `output/models/direction_model.pkl` — artifact (best model, feature columns, epsilon, thresholds, risk bands; stacking includes meta-learner + base estimators)
- `output/models/direction_model_metrics.json` — split info, per-model accuracy/F1, best model name, walk-forward CV summary, epsilon tuning, threshold tuning, calibration, overfitting diagnostics, comparison vs heuristic

**Overfitting diagnostics (in metrics JSON and terminal):**

- For single-model best: train F1 vs test F1 → verdict `no_significant_overfitting` / `mild_overfitting` / `likely_overfitting`.
- For ensemble best (stacking/soft voting): train F1 not computed; verdict from test vs walk-forward CV → `ensemble_stable_vs_cv` or `ensemble_unstable_vs_cv` (plus `temporal_stability` and `holdout_cv_f1_gap`).

**Config:** `model_training.*` — `models`, `test_fraction`, `target_epsilon_grid`, `walk_forward_splits`, `calibrate_probabilities`, `class_threshold_candidates`, `output_model_path`, `output_metrics_path`, etc.

---

### 6b. Forecast dashboard

Generate a standalone HTML dashboard for the latest forecast.

```bash
python scripts/6b_visualize_forecast.py --config config.yaml
```

**Outputs:**

- `output/visualizations/forecast_dashboard.html`

---

## Configuration

Edit `config.yaml` to tune behaviour. Key sections:

| Section | Examples |
|--------|----------|
| `max_cores` | 0 = use all cores; set to N to cap parallelism |
| `input` | `embeddings`, `raw_data`, `data` (cleaned CSV), `text_column`, `timestamp_column` |
| `pca` | `enabled`, `n_components` — optional PCA before HDBSCAN |
| `clustering` | `min_cluster_size`, `min_samples`, `n_jobs` |
| `umap` | `n_components`, `n_neighbors`, `min_dist`, `metric` |
| `temporal` | `default_window`, `recency_*` for weekly snapshots |
| `output` | Paths for parquet, CSV, viz dir |
| `cluster_labeling` | `domain_stop_words`, `ctfidf_*`, `representative_docs` |
| `llm` | `enabled`, `provider`, `model` — for optional Gemini semantic rerank / chat |
| `theme_activation` | `theme_lexicon`, `output`, `n_jobs`, thresholds, `tfidf.*` |
| `event_forecast` | `target_epsilon`, `risk_band_quantiles`, `activation_threshold`, `per_theme_threshold` |
| `model_training` | `models`, `test_fraction`, `target_epsilon_grid`, `walk_forward_splits`, `walk_forward_min_train_windows`, `max_lag`, `calibrate_probabilities`, `class_threshold_candidates`, `force_feature_prune`, `force_prune_max_f1_drop`, `output_model_path`, `output_metrics_path` |

**Built-in model names:** `logistic`, `random_forest`, `extra_trees`, `hist_gbm`; optional: `xgboost`, `lightgbm`, `catboost` (skipped if not installed).

---

## Frontend + API Server

The API server wraps the pipeline for the interactive dashboard and chat.

### Quick start

```bash
# Terminal 1: API server
python server.py
# → http://localhost:8000  (root redirects to /docs)
```

```bash
# Terminal 2: Frontend (optional)
cd frontend
npm install
npm run dev
# → http://localhost:8080  (proxies /api/* to backend)
```

- **Root:** `http://localhost:8000/` redirects to **http://localhost:8000/docs** (Swagger UI).
- **API docs:** `http://localhost:8000/docs`
- **Health:** `http://localhost:8000/api/health`

### News Impact Chat

The **News Impact Chat** page (frontend `/news-chat`) lets you enter a headline; the backend runs the prediction pipeline and (if configured) Gemini LLM analysis. Set `GOOGLE_API_KEY` in `.env` for Gemini.

### API endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Redirects to `/docs` |
| `/api/health` | GET | Status, model loaded, theme/cluster counts |
| `/api/predict` | POST | Run event prediction pipeline |
| `/api/chat` | POST | Gemini LLM analysis of prediction |
| `/api/clusters` | GET | Cluster data |
| `/api/snapshots` | GET | Monthly trend snapshots |
| `/api/snapshots/weekly-recent` | GET | Weekly recency snapshots |
| `/api/kpis` | GET | Dashboard KPIs |
| `/api/lifecycle-distribution` | GET | Lifecycle state counts |
| `/api/temporal-events` | GET | Detected anomaly events |
| `/api/model-metrics` | GET | Direction model metrics JSON |
| `/api/themes` | GET | Theme lexicon list |

---

## Data files and Git

These are typically local-only (in `.gitignore`); do not commit them:

- `clustered_data_labeled.csv`
- `embeddings.npy`
- `output/*.parquet`, `output/*.csv`, `output/models/*`, `output/visualizations/*`
- `catboost_info/`

Keep them locally for pipeline runs.
