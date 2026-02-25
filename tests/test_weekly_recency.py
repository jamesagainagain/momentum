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
