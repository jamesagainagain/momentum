"""
FastAPI server bridging the InsightAgent pipeline to the frontend.

Run:  uvicorn server:app --reload --port 8000
"""

import json
import math
import os
import sys
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))
from scripts.event_forecast import build_cluster_profiles, estimate_event_direction
from scripts.forecast_model import DIRECTION_LABELS, confidence_from_proba, pick_label_from_proba, risk_band
from scripts.io_utils import read_parquet_safe

BASE = Path(__file__).parent
CONFIG_PATH = BASE / "config.yaml"

with open(CONFIG_PATH, encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)


def _resolve(p: str) -> Path:
    return Path(p) if os.path.isabs(p) else BASE / p


TEMPORAL_STATS_PATH = _resolve(CONFIG["output"]["temporal_stats"])
SNAPSHOTS_CSV_PATH = _resolve(CONFIG["output"]["cluster_trend_snapshots_1m_csv"])
WEEKLY_RECENT_CSV_PATH = _resolve(
    CONFIG["output"].get(
        "cluster_trend_snapshots_1w_recent_csv",
        "output/cluster_trend_snapshots_1W_recent.csv",
    )
)
LEXICON_PATH = _resolve(CONFIG.get("theme_activation", {}).get("theme_lexicon", "theme_lexicon.yaml"))
MODEL_PATH = _resolve(CONFIG.get("model_training", {}).get("output_model_path", "output/models/direction_model.pkl"))
METRICS_PATH = _resolve(CONFIG.get("model_training", {}).get("output_metrics_path", "output/models/direction_model_metrics.json"))
PREDICTION_JSON = _resolve(CONFIG.get("event_forecast", {}).get("prediction_output", "output/event_direction_prediction.json"))
FORECAST_CFG = CONFIG.get("event_forecast", {})
LLM_CFG = CONFIG.get("llm", {}) or {}

temporal_df = read_parquet_safe(str(TEMPORAL_STATS_PATH))
profiles_df = build_cluster_profiles(temporal_df)

with open(LEXICON_PATH, encoding="utf-8") as f:
    THEMES = yaml.safe_load(f).get("themes", [])

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

model_bundle = None
if MODEL_PATH.exists():
    try:
        model_bundle = joblib.load(str(MODEL_PATH))
    except Exception as exc:
        print(f"WARN: model load failed: {exc}")

model_metrics = {}
if METRICS_PATH.exists():
    with open(METRICS_PATH, encoding="utf-8") as f:
        model_metrics = json.load(f)

vol_series = pd.to_numeric(
    temporal_df.get("volume_volatility", pd.Series([0.0])), errors="coerce"
).fillna(0.0)
risk_cfg = FORECAST_CFG.get("risk_band_quantiles", {}) or {}
if model_bundle and "risk_bands" in model_bundle:
    LOW_Q = float(model_bundle["risk_bands"].get("low_value", vol_series.quantile(float(risk_cfg.get("low", 0.33)))))
    HIGH_Q = float(model_bundle["risk_bands"].get("high_value", vol_series.quantile(float(risk_cfg.get("high", 0.66)))))
else:
    LOW_Q = float(vol_series.quantile(float(risk_cfg.get("low", 0.33))))
    HIGH_Q = float(vol_series.quantile(float(risk_cfg.get("high", 0.66))))


def _weighted_profile_features(activated_themes):
    if not activated_themes:
        return {k: 0.0 for k in [
            "volume_volatility", "market_share", "momentum",
            "volume_pct_change_lag1", "volume_pct_change_lag2", "volume_pct_change_lag3", "volume_pct_change_lag4",
            "momentum_lag1", "momentum_lag2", "momentum_lag3", "momentum_lag4", "trend_slope_4",
            "theme_active_count", "theme_top1_score", "theme_top2_score", "theme_entropy",
            "rolling_post_mean_3", "rolling_post_mean_6", "anomaly_score",
        ]}
    return {
        "volume_volatility": float(sum(a["weight"] * a.get("avg_volatility", 0.0) for a in activated_themes)),
        "market_share": float(sum(a["weight"] * a.get("avg_market_share", 0.0) for a in activated_themes)),
        "momentum": float(sum(a["weight"] * a.get("recent_momentum", 0.0) for a in activated_themes)),
        "volume_pct_change_lag1": float(sum(a["weight"] * a.get("recent_growth_lag1", 0.0) for a in activated_themes)),
        "volume_pct_change_lag2": float(sum(a["weight"] * a.get("recent_growth_lag2", 0.0) for a in activated_themes)),
        "volume_pct_change_lag3": float(sum(a["weight"] * a.get("recent_growth_lag3", 0.0) for a in activated_themes)),
        "volume_pct_change_lag4": float(sum(a["weight"] * a.get("recent_growth_lag4", 0.0) for a in activated_themes)),
        "momentum_lag1": float(sum(a["weight"] * a.get("recent_momentum_lag1", 0.0) for a in activated_themes)),
        "momentum_lag2": float(sum(a["weight"] * a.get("recent_momentum_lag2", 0.0) for a in activated_themes)),
        "momentum_lag3": float(sum(a["weight"] * a.get("recent_momentum_lag3", 0.0) for a in activated_themes)),
        "momentum_lag4": float(sum(a["weight"] * a.get("recent_momentum_lag4", 0.0) for a in activated_themes)),
        "trend_slope_4": float(sum(a["weight"] * a.get("trend_slope", 0.0) for a in activated_themes)),
        "rolling_post_mean_3": float(sum(a["weight"] * a.get("rolling_post_mean_3", 0.0) for a in activated_themes)),
        "rolling_post_mean_6": float(sum(a["weight"] * a.get("rolling_post_mean_6", 0.0) for a in activated_themes)),
        "anomaly_score": float(sum(a["weight"] * a.get("recent_anomaly_score", 0.0) for a in activated_themes)),
    }


def _add_theme_summary_features(x, theme_scores):
    vals = [max(0.0, float(v)) for v in theme_scores.values()]
    if not vals:
        x.update({"theme_active_count": 0.0, "theme_top1_score": 0.0, "theme_top2_score": 0.0, "theme_entropy": 0.0})
        return
    arr = sorted(vals)
    total = sum(vals)
    entropy = 0.0
    if total > 0:
        for v in vals:
            p = v / total
            entropy -= p * np.log(p + 1e-12)
    x["theme_active_count"] = float(sum(1 for v in vals if v > 0))
    x["theme_top1_score"] = float(arr[-1])
    x["theme_top2_score"] = float(arr[-2]) if len(arr) > 1 else float(arr[-1])
    x["theme_entropy"] = float(entropy)


def _predict_from_estimator(estimator, X, labels):
    if hasattr(estimator, "predict_proba"):
        raw = estimator.predict_proba(X)[0]
        cls = list(getattr(estimator, "classes_", labels))
        probs = {label: 0.0 for label in labels}
        for idx, c in enumerate(cls):
            label_key = None
            try:
                c_int = int(c)
                if 0 <= c_int < len(labels):
                    label_key = labels[c_int]
            except Exception:
                pass
            if label_key is None:
                c_str = str(c)
                if c_str in labels:
                    label_key = c_str
            if label_key is not None:
                probs[label_key] = float(raw[idx])
    else:
        pred = str(estimator.predict(X)[0])
        probs = {label: 0.0 for label in labels}
        probs[pred] = 1.0
    return probs


def _infer_with_trained_model(prediction, bundle):
    feature_cols = bundle.get("feature_columns", [])
    labels = bundle.get("labels", DIRECTION_LABELS)
    thresholds = bundle.get("class_thresholds", {"down": 1.0, "flat": 1.0, "up": 1.0})
    x = {col: 0.0 for col in feature_cols}

    theme_scores = prediction.get("theme_scores", {})
    for col in feature_cols:
        if col.startswith("theme_") and col.endswith("_score"):
            tid = col[len("theme_"):-len("_score")]
            x[col] = float(theme_scores.get(str(tid), 0.0))

    aggregate = _weighted_profile_features(prediction.get("activated_themes", []))
    for key, value in aggregate.items():
        if key in x:
            x[key] = float(value)
    _add_theme_summary_features(x, theme_scores)

    if "momentum_x_volatility" in x:
        x["momentum_x_volatility"] = x.get("momentum", 0.0) * x.get("volume_volatility", 0.0)
    if "volume_acceleration" in x:
        x["volume_acceleration"] = x.get("volume_pct_change_lag1", 0.0) - x.get("volume_pct_change_lag2", 0.0)
    if "momentum_acceleration" in x:
        x["momentum_acceleration"] = x.get("momentum_lag1", 0.0) - x.get("momentum_lag2", 0.0)
    # theme_entropy_change requires a prior window; at single-event inference
    # we leave it at 0.0 (no delta available). The model was trained on diffs,
    # so 0.0 = "no change" which is the safest single-snapshot assumption.

    X = pd.DataFrame([x], columns=feature_cols).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if "stacking_meta_learner" in bundle and "ensemble_members" in bundle:
        meta_learner = bundle["stacking_meta_learner"]
        stacking_meta_cols = bundle.get("stacking_meta_cols", [])
        meta_features = {col: 1.0 / max(len(labels), 1) for col in stacking_meta_cols}
        for member in bundle.get("ensemble_members", []):
            est = member.get("estimator")
            name = member.get("name", "")
            if est is None:
                continue
            probs = _predict_from_estimator(est, X, labels)
            for label in labels:
                meta_features[f"{name}_prob_{label}"] = probs.get(label, 0.0)
        meta_X = pd.DataFrame([meta_features])[stacking_meta_cols].astype(float)
        stacking_proba = meta_learner.predict_proba(meta_X)[0]
        stacking_classes = list(meta_learner.classes_)
        class_probs = {label: 0.0 for label in labels}
        for idx, c in enumerate(stacking_classes):
            if c in labels:
                class_probs[c] = float(stacking_proba[idx])
    elif "ensemble_members" in bundle:
        member_probs = []
        for member in bundle.get("ensemble_members", []):
            estimator = member.get("estimator")
            if estimator is None:
                continue
            member_probs.append(_predict_from_estimator(estimator, X, labels))
        if not member_probs:
            return "flat", 0.0, {}
        class_probs = {label: 0.0 for label in labels}
        for row in member_probs:
            for label in labels:
                class_probs[label] += float(row.get(label, 0.0))
        class_probs = {k: v / max(1, len(member_probs)) for k, v in class_probs.items()}
    else:
        estimator = bundle["model"]
        class_probs = _predict_from_estimator(estimator, X, labels)

    pred, adjusted_probs = pick_label_from_proba(
        proba_map=class_probs,
        threshold_multipliers=thresholds,
        labels=labels,
    )
    confidence = confidence_from_proba([adjusted_probs[label] for label in labels])
    return str(pred), float(confidence), adjusted_probs


app = FastAPI(title="InsightAgent API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    event_text: str


class ChatRequest(BaseModel):
    event_text: str
    prediction: dict = Field(default_factory=dict)
    history: list = Field(default_factory=list)


def _sanitize(obj):
    """Recursively replace NaN/Inf with None for JSON serialization."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    return obj


@app.get("/")
def root():
    """Redirect root to API docs."""
    return RedirectResponse(url="/docs", status_code=302)


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


@app.post("/api/predict")
def predict(req: PredictRequest):
    threshold = float(FORECAST_CFG.get("activation_threshold", 4.0))
    per_theme_thresholds = FORECAST_CFG.get("per_theme_threshold", {}) or {}

    prediction = estimate_event_direction(
        event_text=req.event_text,
        themes=THEMES,
        cluster_profiles_df=profiles_df,
        activation_threshold=threshold,
        per_theme_thresholds=per_theme_thresholds,
        llm_config=LLM_CFG,
        theme_cluster_map=THEME_CLUSTER_MAP,
    )
    prediction["event_text"] = req.event_text
    prediction["activation_threshold"] = threshold
    prediction["heuristic_predicted_direction"] = prediction["predicted_direction"]
    prediction["heuristic_predicted_direction_score"] = prediction["predicted_direction_score"]
    prediction["method"] = "heuristic"
    prediction["confidence"] = float(min(1.0, abs(prediction["predicted_direction_score"])))
    prediction["class_probabilities"] = {}

    if model_bundle is not None:
        trained_dir, conf, class_probs = _infer_with_trained_model(prediction, model_bundle)
        prediction["predicted_direction"] = trained_dir
        prediction["method"] = "trained"
        prediction["confidence"] = float(conf)
        prediction["class_probabilities"] = class_probs
        prediction["model_name"] = model_bundle.get("model_name")

    prediction["risk_band"] = risk_band(prediction.get("predicted_volatility", 0.0), low_q=LOW_Q, high_q=HIGH_Q)
    prediction["risk_thresholds"] = {"low": LOW_Q, "high": HIGH_Q}

    activated_cluster_ids = [t["theme_id"] for t in prediction.get("activated_themes", [])]
    trend_data = _get_cluster_trends(activated_cluster_ids)
    prediction["trend_data"] = trend_data

    return _sanitize(prediction)


@app.post("/api/chat")
def chat_analysis(req: ChatRequest):
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY not set in .env")

    try:
        from google import genai
    except ImportError:
        raise HTTPException(status_code=500, detail="google-genai package not installed")

    prediction = req.prediction
    if not prediction:
        raise HTTPException(status_code=400, detail="No prediction data provided")

    direction = prediction.get("predicted_direction", "flat")
    confidence = prediction.get("confidence", 0)
    volatility = prediction.get("predicted_volatility", 0)
    risk = prediction.get("risk_band", "medium")
    method = prediction.get("method", "heuristic")
    themes = prediction.get("activated_themes", [])
    class_probs = prediction.get("class_probabilities", {})
    event_text = req.event_text

    theme_details = "\n".join([
        f"  - {t.get('theme_name', 'Unknown')} (score: {t.get('score', 0):.2f}, "
        f"weight: {t.get('weight', 0):.2f}, direction: {t.get('direction_label', '?')})"
        for t in themes
    ]) or "  None activated"

    history_context = ""
    if req.history:
        recent = req.history[-3:]
        history_context = "\n\nPrevious predictions in this session:\n" + "\n".join([
            f"- \"{h.get('event_text', '')}\" → {h.get('predicted_direction', '?')} "
            f"(conf: {h.get('confidence', 0):.0%})"
            for h in recent
        ])

    prompt = f"""You are InsightAgent, an expert AI analyst specializing in social media trend analysis and forecasting.

A user submitted this news event for impact prediction: "{event_text}"

The ML pipeline produced this prediction:
- Predicted Direction: {direction} ({"upward trend expected" if direction == "up" else "downward trend expected" if direction == "down" else "no significant directional change"})
- Prediction Method: {method} model
- Confidence: {confidence:.1%}
- Class Probabilities: down={class_probs.get("down", 0):.1%}, flat={class_probs.get("flat", 0):.1%}, up={class_probs.get("up", 0):.1%}
- Predicted Volatility: {volatility:.4f}
- Risk Band: {risk}
- Activated Themes ({len(themes)}):
{theme_details}
{history_context}

Provide a detailed analysis covering:
1. **Direction Analysis**: Explain what the {direction} prediction means for social media discussion trends. Reference the specific activated themes and their contribution weights.
2. **Confidence Assessment**: Interpret the {confidence:.1%} confidence and class probability spread. Is this a strong or uncertain signal?
3. **Volatility & Risk**: Explain what {volatility:.4f} volatility and {risk} risk mean practically. Will discussions be stable or turbulent?
4. **Theme Breakdown**: For each activated theme, explain how it influences the prediction and what specific aspects of the event drive that theme.
5. **Trend Forecast**: Based on the activated cluster profiles, predict how social media engagement will evolve over the next 1-4 weeks. Be specific about expected patterns.
6. **Strategic Implications**: What should stakeholders watch for? Any early warning signals?

Format your response in clear markdown with headers. Be specific and data-driven, referencing the actual numbers. Keep it thorough but readable (400-600 words)."""

    try:
        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model=LLM_CFG.get("model", "gemini-2.5-flash"),
            contents=prompt,
        )
        analysis_text = getattr(resp, "text", "") or ""
    except Exception as exc:
        analysis_text = f"LLM analysis unavailable: {exc}"

    return {"analysis": analysis_text, "event_text": event_text}


def _best_window(df: pd.DataFrame, min_clusters: int = 5) -> str:
    """Pick the latest time_window that has at least min_clusters entries."""
    counts = df.groupby("time_window").size()
    viable = counts[counts >= min_clusters]
    if viable.empty:
        return str(df["time_window"].max())
    return str(viable.index.max())


@app.get("/api/clusters")
def get_clusters():
    if not SNAPSHOTS_CSV_PATH.exists():
        raise HTTPException(status_code=404, detail="Snapshot CSV not found")
    df = pd.read_csv(str(SNAPSHOTS_CSV_PATH), low_memory=False)
    df = df[df["cluster"] != -1].copy()

    latest_tw = _best_window(df)
    latest = df[df["time_window"] == latest_tw].copy()

    clusters = []
    for _, row in latest.iterrows():
        clusters.append({
            "id": f"cluster-{int(row['cluster'])}",
            "cluster": int(row["cluster"]),
            "cluster_label": str(row.get("cluster_label", f"Cluster {int(row['cluster'])}")),
            "theme_name": CLUSTER_THEME_LABELS.get(int(row["cluster"]), ""),
            "size": int(row.get("post_count", 0)),
            "market_share": float(row.get("market_share", 0) * 100),
            "volume_pct_change": float(row.get("volume_pct_change", 0) * 100),
            "volume_volatility": float(row.get("volume_volatility", 0)),
            "momentum": float(row.get("momentum", 0)),
            "lifecycle": str(row.get("lifecycle_state", "stable")).capitalize(),
            "anomaly_score": float(row.get("anomaly_score", 0)),
        })

    return {"clusters": _sanitize(clusters), "latest_window": latest_tw, "total": len(clusters)}


@app.get("/api/snapshots")
def get_snapshots():
    if not SNAPSHOTS_CSV_PATH.exists():
        raise HTTPException(status_code=404, detail="Snapshot CSV not found")
    df = pd.read_csv(str(SNAPSHOTS_CSV_PATH), low_memory=False)
    df = df[df["cluster"] != -1].copy()

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
        })

    return _sanitize(snapshots)


@app.get("/api/snapshots/weekly-recent")
def get_weekly_recent_snapshots():
    """Weekly snapshots for the last 12 months, dense clusters only."""
    if not WEEKLY_RECENT_CSV_PATH.exists():
        return []
    df = pd.read_csv(str(WEEKLY_RECENT_CSV_PATH), low_memory=False)
    snapshots = []
    for _, row in df.iterrows():
        cid = int(row["cluster"])
        snapshots.append({
            "cluster_id": f"cluster-{cid}",
            "cluster_label": str(row.get("cluster_label", f"Cluster {cid}")),
            "theme_name": CLUSTER_THEME_LABELS.get(cid, ""),
            "time_window": str(row.get("time_window", "")),
            "post_count": int(row.get("post_count", 0)),
            "market_share": float(row.get("market_share", 0) * 100),
            "momentum": float(row.get("momentum", 0)),
            "volatility": float(row.get("volume_volatility", 0)),
            "window_type": "1W",
        })
    return _sanitize(snapshots)


@app.get("/api/temporal-events")
def get_temporal_events():
    events_path = _resolve(CONFIG["output"]["temporal_events"])
    if not events_path.exists():
        return []
    try:
        df = read_parquet_safe(str(events_path))
    except Exception:
        return []

    events = []
    for _, row in df.head(50).iterrows():
        events.append({
            "date": str(row.get("time_window", "")),
            "cluster_label": str(row.get("cluster_label", f"Cluster {row.get('cluster', '?')}")),
            "event_type": str(row.get("event_type", "spike")),
            "description": str(row.get("description", "")),
            "sigma": float(row.get("sigma", row.get("anomaly_score", 0))),
        })
    return _sanitize(events)


@app.get("/api/model-metrics")
def get_model_metrics():
    return _sanitize(model_metrics)


@app.get("/api/lifecycle-distribution")
def get_lifecycle_distribution():
    if not SNAPSHOTS_CSV_PATH.exists():
        return []
    df = pd.read_csv(str(SNAPSHOTS_CSV_PATH), low_memory=False)
    df = df[df["cluster"] != -1].copy()
    latest_tw = _best_window(df)
    latest = df[df["time_window"] == latest_tw]
    if "lifecycle_state" not in latest.columns:
        return []
    counts = latest["lifecycle_state"].value_counts().to_dict()
    return [{"state": k.capitalize(), "count": int(v)} for k, v in counts.items()]


@app.get("/api/kpis")
def get_kpis():
    if not SNAPSHOTS_CSV_PATH.exists():
        return {}
    df = pd.read_csv(str(SNAPSHOTS_CSV_PATH), low_memory=False)
    df = df[df["cluster"] != -1].copy()
    latest_tw = _best_window(df)
    latest = df[df["time_window"] == latest_tw]

    active = len(latest)
    lifecycle_col = "lifecycle_state" if "lifecycle_state" in latest.columns else None
    emerging = int((latest[lifecycle_col] == "emerging").sum()) if lifecycle_col else 0
    declining = int((latest[lifecycle_col] == "declining").sum()) if lifecycle_col else 0
    avg_vol = float(latest["volume_volatility"].mean()) if "volume_volatility" in latest.columns else 0
    event_spikes = int((latest["anomaly_score"].abs() > 2).sum()) if "anomaly_score" in latest.columns else 0

    return _sanitize({
        "activeClusters": active,
        "emergingCount": emerging,
        "decliningCount": declining,
        "avgVolatility": round(avg_vol, 3),
        "eventSpikes": event_spikes,
    })


@app.get("/api/themes")
def get_themes():
    return [{"id": t["id"], "name": t.get("name", str(t["id"]))} for t in THEMES]


def _get_cluster_trends(cluster_ids: list[int]) -> list[dict]:
    """Get historical trend data for specific clusters for charting."""
    if not SNAPSHOTS_CSV_PATH.exists() or not cluster_ids:
        return []
    df = pd.read_csv(str(SNAPSHOTS_CSV_PATH), low_memory=False)
    df = df[df["cluster"].isin(cluster_ids)].copy()
    if df.empty:
        return []

    result = []
    for cid in cluster_ids:
        cdf = df[df["cluster"] == cid].sort_values("time_window")
        if cdf.empty:
            continue
        label = str(cdf.iloc[0].get("cluster_label", f"Cluster {cid}"))
        points = []
        for _, row in cdf.iterrows():
            points.append({
                "time_window": str(row["time_window"]),
                "post_count": int(row.get("post_count", 0)),
                "market_share": float(row.get("market_share", 0) * 100),
                "momentum": float(row.get("momentum", 0)),
                "volatility": float(row.get("volume_volatility", 0)),
            })
        result.append({
            "cluster_id": cid,
            "cluster_label": label,
            "data": points,
        })
    return _sanitize(result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
