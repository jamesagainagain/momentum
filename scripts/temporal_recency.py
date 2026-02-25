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
