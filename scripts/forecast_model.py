import math

import numpy as np
import pandas as pd


DIRECTION_LABELS = ["down", "flat", "up"]


def direction_from_change(change, epsilon):
    if change > epsilon:
        return "up"
    if change < -epsilon:
        return "down"
    return "flat"


def add_growth_lag_features(df, lag_columns=None, max_lag=4):
    lag_columns = lag_columns or ["volume_pct_change", "momentum"]
    out = df.copy()
    for col in lag_columns:
        if col not in out.columns:
            out[col] = 0.0
        for lag in range(1, max_lag + 1):
            out[f"{col}_lag{lag}"] = out.groupby("cluster")[col].shift(lag).fillna(0.0)
    return out


def _entropy_from_row(scores):
    vals = np.asarray([max(0.0, float(v)) for v in scores], dtype=float)
    total = float(vals.sum())
    if total <= 0:
        return 0.0
    p = vals / total
    return float(-(p * np.log(p + 1e-12)).sum())


def build_training_frame(temporal_df, theme_df, epsilon, max_lag=4):
    base = temporal_df.copy()
    base["time_window"] = pd.to_datetime(base["time_window"], errors="coerce")
    base = base.dropna(subset=["time_window"]).sort_values(["cluster", "time_window"])
    base = add_growth_lag_features(base, max_lag=max_lag)

    if "post_count" not in base.columns:
        base["post_count"] = 0.0
    if "anomaly_score" not in base.columns:
        mean_vol = base.groupby("cluster")["post_count"].transform("mean")
        std_vol = base.groupby("cluster")["post_count"].transform("std").replace(0, np.nan)
        base["anomaly_score"] = ((base["post_count"] - mean_vol) / std_vol).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    base["rolling_post_mean_3"] = base.groupby("cluster")["post_count"].transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )
    base["rolling_post_mean_6"] = base.groupby("cluster")["post_count"].transform(
        lambda s: s.rolling(6, min_periods=1).mean()
    )
    base["trend_slope_4"] = base.groupby("cluster")["post_count"].transform(_rolling_slope_4)

    if theme_df is not None and not theme_df.empty:
        theme = theme_df.copy()
        theme["time_window"] = pd.to_datetime(theme["time_window"], errors="coerce")
        keep_cols = ["cluster", "time_window"] + [
            c for c in theme.columns if c.startswith("theme_") and c.endswith("_score")
        ]
        base = base.merge(theme[keep_cols], on=["cluster", "time_window"], how="left")

    theme_cols = [c for c in base.columns if c.startswith("theme_") and c.endswith("_score")]
    if theme_cols:
        theme_scores = base[theme_cols].fillna(0.0)
        base["theme_active_count"] = (theme_scores > 0).sum(axis=1)
        sorted_vals = np.sort(theme_scores.to_numpy(dtype=float), axis=1)
        base["theme_top1_score"] = sorted_vals[:, -1]
        base["theme_top2_score"] = sorted_vals[:, -2] if sorted_vals.shape[1] > 1 else sorted_vals[:, -1]
        base["theme_entropy"] = theme_scores.apply(lambda row: _entropy_from_row(row.values), axis=1)
    else:
        base["theme_active_count"] = 0.0
        base["theme_top1_score"] = 0.0
        base["theme_top2_score"] = 0.0
        base["theme_entropy"] = 0.0

    for col in ("momentum", "volume_volatility"):
        if col not in base.columns:
            base[col] = 0.0
    base["momentum_x_volatility"] = (
        pd.to_numeric(base["momentum"], errors="coerce").fillna(0.0)
        * pd.to_numeric(base["volume_volatility"], errors="coerce").fillna(0.0)
    )
    base["volume_acceleration"] = (
        pd.to_numeric(base.get("volume_pct_change_lag1", 0.0), errors="coerce").fillna(0.0)
        - pd.to_numeric(base.get("volume_pct_change_lag2", 0.0), errors="coerce").fillna(0.0)
    )
    base["momentum_acceleration"] = (
        pd.to_numeric(base.get("momentum_lag1", 0.0), errors="coerce").fillna(0.0)
        - pd.to_numeric(base.get("momentum_lag2", 0.0), errors="coerce").fillna(0.0)
    )
    base["theme_entropy_change"] = base.groupby("cluster")["theme_entropy"].diff().fillna(0.0)

    feature_cols = [
        *sorted(theme_cols),
        "momentum",
        "volume_volatility",
        "market_share",
        "anomaly_score",
        "rolling_post_mean_3",
        "rolling_post_mean_6",
        "trend_slope_4",
        "theme_active_count",
        "theme_top1_score",
        "theme_top2_score",
        "theme_entropy",
        "momentum_x_volatility",
        "volume_acceleration",
        "momentum_acceleration",
        "theme_entropy_change",
        "volume_pct_change_lag1",
        "volume_pct_change_lag2",
        "volume_pct_change_lag3",
        "volume_pct_change_lag4",
        "momentum_lag1",
        "momentum_lag2",
        "momentum_lag3",
        "momentum_lag4",
    ]

    for col in feature_cols:
        if col not in base.columns:
            base[col] = 0.0
        base[col] = (
            pd.to_numeric(base[col], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

    base["next_volume_pct_change"] = base.groupby("cluster")["volume_pct_change"].shift(-1)
    base["next_volume_pct_change"] = (
        pd.to_numeric(base["next_volume_pct_change"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    base["target"] = base["next_volume_pct_change"].apply(
        lambda x: direction_from_change(0.0 if pd.isna(x) else float(x), epsilon)
    )
    base = base.dropna(subset=["next_volume_pct_change"]).copy()
    return base, feature_cols


def heuristic_predict_direction(row, epsilon):
    momentum_norm = float(row.get("momentum", 0.0)) / max(float(row.get("post_count", 1.0)), 1.0)
    score = (0.7 * float(row.get("volume_pct_change", 0.0))) + (0.3 * momentum_norm)
    return direction_from_change(score, epsilon)


def apply_probability_thresholds(proba_map, threshold_multipliers=None, labels=None):
    labels = labels or DIRECTION_LABELS
    threshold_multipliers = threshold_multipliers or {}
    adjusted = {}
    for label in labels:
        p = float(proba_map.get(label, 0.0))
        mult = float(threshold_multipliers.get(label, 1.0))
        adjusted[label] = p * mult
    total = sum(adjusted.values())
    if total > 0:
        adjusted = {k: (v / total) for k, v in adjusted.items()}
    return adjusted


def pick_label_from_proba(proba_map, threshold_multipliers=None, labels=None):
    labels = labels or DIRECTION_LABELS
    adjusted = apply_probability_thresholds(
        proba_map=proba_map,
        threshold_multipliers=threshold_multipliers,
        labels=labels,
    )
    best = max(labels, key=lambda l: adjusted.get(l, 0.0))
    return best, adjusted


def confidence_from_proba(proba):
    arr = np.asarray(proba, dtype=float).ravel()
    if arr.size == 0:
        return 0.0
    top = np.sort(arr)[::-1]
    if top.size == 1:
        return float(top[0])
    return float(max(0.0, top[0] - top[1]))


def risk_band(volatility, low_q, high_q):
    v = 0.0 if volatility is None or (isinstance(volatility, float) and math.isnan(volatility)) else float(volatility)
    if v <= low_q:
        return "low"
    if v >= high_q:
        return "high"
    return "medium"


def _rolling_slope_4(series):
    vals = pd.to_numeric(series, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    out = np.zeros(len(vals), dtype=float)
    for idx in range(len(vals)):
        start = max(0, idx - 3)
        y = vals[start : idx + 1]
        if y.size <= 1:
            out[idx] = 0.0
            continue
        x = np.arange(y.size, dtype=float)
        out[idx] = float(np.polyfit(x, y, 1)[0])
    return pd.Series(out, index=series.index)
