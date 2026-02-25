"""
Script 5: Event Direction Forecast

Builds per-cluster state database from temporal stats and
predicts likely direction/volatility for a new event text.
"""

import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import yaml

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from scripts.event_forecast import build_cluster_profiles, estimate_event_direction
    from scripts.forecast_model import (
        DIRECTION_LABELS,
        confidence_from_proba,
        pick_label_from_proba,
        risk_band,
    )
    from scripts.io_utils import (
        build_temporal_fallback_from_clusters_csv,
        read_parquet_safe,
    )
else:
    from .event_forecast import build_cluster_profiles, estimate_event_direction
    from .forecast_model import DIRECTION_LABELS, confidence_from_proba, pick_label_from_proba, risk_band
    from .io_utils import build_temporal_fallback_from_clusters_csv, read_parquet_safe


def load_config(config_path):
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_theme_lexicon(path):
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    themes = data.get("themes", [])
    if not themes:
        raise ValueError(f"No themes found in lexicon file: {path}")
    return themes


def weighted_profile_features(activated_themes):
    if not activated_themes:
        return {
            "volume_volatility": 0.0,
            "market_share": 0.0,
            "momentum": 0.0,
            "volume_pct_change_lag1": 0.0,
            "volume_pct_change_lag2": 0.0,
            "volume_pct_change_lag3": 0.0,
            "volume_pct_change_lag4": 0.0,
            "momentum_lag1": 0.0,
            "momentum_lag2": 0.0,
            "momentum_lag3": 0.0,
            "momentum_lag4": 0.0,
            "trend_slope_4": 0.0,
            "theme_active_count": 0.0,
            "theme_top1_score": 0.0,
            "theme_top2_score": 0.0,
            "theme_entropy": 0.0,
            "rolling_post_mean_3": 0.0,
            "rolling_post_mean_6": 0.0,
            "anomaly_score": 0.0,
        }
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
        x["theme_active_count"] = 0.0
        x["theme_top1_score"] = 0.0
        x["theme_top2_score"] = 0.0
        x["theme_entropy"] = 0.0
        return
    arr = sorted(vals)
    top1 = arr[-1]
    top2 = arr[-2] if len(arr) > 1 else arr[-1]
    total = sum(vals)
    entropy = 0.0
    if total > 0:
        for v in vals:
            p = v / total
            entropy -= p * np.log(p + 1e-12)
    x["theme_active_count"] = float(sum(1 for v in vals if v > 0))
    x["theme_top1_score"] = float(top1)
    x["theme_top2_score"] = float(top2)
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


def infer_with_trained_model(prediction, model_bundle):
    feature_cols = model_bundle.get("feature_columns", [])
    labels = model_bundle.get("labels", DIRECTION_LABELS)
    thresholds = model_bundle.get("class_thresholds", {"down": 1.0, "flat": 1.0, "up": 1.0})
    x = {col: 0.0 for col in feature_cols}

    theme_scores = prediction.get("theme_scores", {})
    for col in feature_cols:
        if col.startswith("theme_") and col.endswith("_score"):
            tid = col[len("theme_") : -len("_score")]
            x[col] = float(theme_scores.get(str(tid), 0.0))

    aggregate = weighted_profile_features(prediction.get("activated_themes", []))
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

    X = pd.DataFrame([x], columns=feature_cols).fillna(0.0)
    if "stacking_meta_learner" in model_bundle and "ensemble_members" in model_bundle:
        meta_learner = model_bundle["stacking_meta_learner"]
        stacking_meta_cols = model_bundle.get("stacking_meta_cols", [])
        meta_features = {col: 1.0 / max(len(labels), 1) for col in stacking_meta_cols}
        for member in model_bundle.get("ensemble_members", []):
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
    elif "ensemble_members" in model_bundle:
        member_probs = []
        for member in model_bundle.get("ensemble_members", []):
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
        estimator = model_bundle["model"]
        class_probs = _predict_from_estimator(estimator, X, labels)

    pred, adjusted_probs = pick_label_from_proba(
        proba_map=class_probs,
        threshold_multipliers=thresholds,
        labels=labels,
    )
    confidence = confidence_from_proba([adjusted_probs[label] for label in labels])
    return str(pred), float(confidence), adjusted_probs


def main():
    parser = argparse.ArgumentParser(description="Forecast direction from event text using cluster profiles.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--event-text", default=None, help="Event description/news snippet to score.")
    parser.add_argument("--activation-threshold", type=float, default=None, help="Override activation threshold.")
    parser.add_argument("--model-path", default=None, help="Optional trained model artifact path.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    config = load_config(config_path)
    config_dir = os.path.dirname(config_path)

    def resolve(path):
        return path if os.path.isabs(path) else os.path.join(config_dir, path)

    temporal_stats_path = resolve(config["output"]["temporal_stats"])
    lexicon_path = resolve(config.get("theme_activation", {}).get("theme_lexicon", "theme_lexicon.yaml"))
    clusters_csv_path = resolve(config["input"]["data"])
    ts_col = config["input"]["timestamp_column"]
    default_window = config.get("temporal", {}).get("default_window", "1M")
    forecast_cfg = config.get("event_forecast", {})
    training_cfg = config.get("model_training", {})
    db_output_path = resolve(forecast_cfg.get("cluster_db_output", "output/cluster_state_database.parquet"))
    pred_output_path = resolve(forecast_cfg.get("prediction_output", "output/event_direction_prediction.json"))
    threshold = (
        args.activation_threshold
        if args.activation_threshold is not None
        else float(forecast_cfg.get("activation_threshold", config.get("theme_activation", {}).get("global_threshold", 0.03)))
    )
    per_theme_thresholds = forecast_cfg.get("per_theme_threshold", {}) or {}
    llm_cfg = config.get("llm", {}) or {}

    if not os.path.exists(lexicon_path):
        raise FileNotFoundError(f"Theme lexicon not found: {lexicon_path}")

    try:
        temporal_df = read_parquet_safe(temporal_stats_path)
    except Exception as e:
        print(f"WARN: Could not read temporal stats parquet. Building fallback from CSV ({e})")
        temporal_df = build_temporal_fallback_from_clusters_csv(
            clusters_csv_path,
            ts_col=ts_col,
            window=default_window,
        )
    profiles_df = build_cluster_profiles(temporal_df)

    os.makedirs(os.path.dirname(db_output_path) or ".", exist_ok=True)
    if args.force or not os.path.exists(db_output_path):
        profiles_df.to_parquet(db_output_path, index=False)
    print(f"Saved cluster state database ({len(profiles_df)} rows) to {db_output_path}")

    if not args.event_text:
        print("No --event-text provided. Database build complete.")
        return

    themes = load_theme_lexicon(lexicon_path)
    prediction = estimate_event_direction(
        event_text=args.event_text,
        themes=themes,
        cluster_profiles_df=profiles_df,
        activation_threshold=threshold,
        per_theme_thresholds=per_theme_thresholds,
        llm_config=llm_cfg,
    )
    prediction["event_text"] = args.event_text
    prediction["activation_threshold"] = threshold

    model_path = args.model_path or resolve(training_cfg.get("output_model_path", "output/models/direction_model.pkl"))
    model_bundle = None
    if model_path and os.path.exists(model_path):
        try:
            model_bundle = joblib.load(model_path)
        except Exception as exc:
            print(f"WARN: Could not load model artifact at {model_path}: {exc}")

    prediction["heuristic_predicted_direction"] = prediction["predicted_direction"]
    prediction["heuristic_predicted_direction_score"] = prediction["predicted_direction_score"]
    prediction["method"] = "heuristic"
    prediction["confidence"] = float(min(1.0, abs(prediction["predicted_direction_score"])))
    prediction["class_probabilities"] = {}

    if model_bundle is not None:
        trained_direction, confidence, class_probs = infer_with_trained_model(prediction, model_bundle)
        prediction["predicted_direction"] = trained_direction
        prediction["method"] = "trained"
        prediction["confidence"] = float(confidence)
        prediction["class_probabilities"] = class_probs
        prediction["model_name"] = model_bundle.get("model_name")
        prediction["model_path"] = model_path

    risk_cfg = forecast_cfg.get("risk_band_quantiles", {}) or {}
    low_q_cfg = float(risk_cfg.get("low", 0.33))
    high_q_cfg = float(risk_cfg.get("high", 0.66))
    vol_series = pd.to_numeric(temporal_df.get("volume_volatility", pd.Series([0.0])), errors="coerce").fillna(0.0)

    if model_bundle and "risk_bands" in model_bundle:
        low_q = float(model_bundle["risk_bands"].get("low_value", vol_series.quantile(low_q_cfg)))
        high_q = float(model_bundle["risk_bands"].get("high_value", vol_series.quantile(high_q_cfg)))
    else:
        low_q = float(vol_series.quantile(low_q_cfg))
        high_q = float(vol_series.quantile(high_q_cfg))

    prediction["risk_band"] = risk_band(prediction.get("predicted_volatility", 0.0), low_q=low_q, high_q=high_q)
    prediction["risk_thresholds"] = {"low": low_q, "high": high_q}

    os.makedirs(os.path.dirname(pred_output_path) or ".", exist_ok=True)
    with open(pred_output_path, "w", encoding="utf-8") as f:
        json.dump(prediction, f, ensure_ascii=True, indent=2)

    print(
        "Prediction: direction="
        f"{prediction['predicted_direction']} "
        f"(method={prediction['method']}, confidence={prediction['confidence']:.4f}), "
        f"volatility={prediction['predicted_volatility']:.4f}, "
        f"risk={prediction['risk_band']}, "
        f"activated_themes={prediction['activated_theme_count']}"
    )
    print(f"Saved prediction JSON to {pred_output_path}")


if __name__ == "__main__":
    main()
