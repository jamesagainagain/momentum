"""
Script 2: Temporal Analysis
- Loads clusters.parquet
- Calculates per-cluster time-windowed metrics (volume, engagement, volatility, momentum)
- Detects event spikes/drops
- Classifies lifecycle states
- Saves temporal_stats.parquet and temporal_events.parquet
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import yaml

from scripts.temporal_recency import build_weekly_recency_snapshots


def _cpu_count():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def _get_n_workers(config):
    max_cores = int(config.get("max_cores", 0))
    available = _cpu_count()
    return max_cores if max_cores > 0 else available


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def classify_lifecycle(series):
    """Classify cluster lifecycle state based on recent trend."""
    if len(series) < 2:
        return 'stable'

    recent_growth = series['volume_pct_change'].tail(3).mean()
    recent_volume = series['post_count'].tail(3).mean()
    overall_volume = series['post_count'].mean()

    if overall_volume == 0:
        return 'dormant'
    if recent_volume < overall_volume * 0.2:
        return 'dormant'
    elif recent_growth > 0.5:
        return 'emerging'
    elif recent_growth > 0.1:
        return 'trending'
    elif recent_growth < -0.3:
        return 'declining'
    else:
        return 'stable'


def detect_events(df, spike_threshold=2.0):
    """Return rows where anomaly_score exceeds threshold."""
    events = df[df['anomaly_score'].abs() > spike_threshold].copy()
    events['event_type'] = events['anomaly_score'].apply(
        lambda x: 'spike' if x > 0 else 'drop'
    )
    return events


def compute_sentiment_distribution(group, sentiment_col):
    """Return dict with positive/neutral/negative ratios."""
    if sentiment_col not in group.columns:
        return {'positive': None, 'neutral': None, 'negative': None}
    counts = group[sentiment_col].str.lower().value_counts(normalize=True)
    return {
        'positive': round(counts.get('positive', 0.0), 4),
        'neutral': round(counts.get('neutral', 0.0), 4),
        'negative': round(counts.get('negative', 0.0), 4),
    }


def _process_single_cluster(args):
    """Process one cluster's temporal metrics. Top-level function for pickling."""
    cluster_id, cdf_parquet_bytes, window, col_flags, ts_col = args
    has_author, has_likes, has_reach, has_shares, has_comments, has_label = col_flags

    import io
    cdf = pd.read_parquet(io.BytesIO(cdf_parquet_bytes))
    cdf[ts_col] = pd.to_datetime(cdf[ts_col], errors='coerce')
    cdf = cdf.dropna(subset=[ts_col]).set_index(ts_col)

    label = cdf['cluster_label'].iloc[0] if has_label else str(cluster_id)

    resampled = cdf.resample(window).agg(
        post_count=('cluster', 'count'),
        unique_authors=('Author', 'nunique') if has_author else ('cluster', 'count'),
        total_likes=('likes', 'sum') if has_likes else ('cluster', 'count'),
        total_reach=('reach', 'sum') if has_reach else ('cluster', 'count'),
        total_shares=('shares', 'sum') if has_shares else ('cluster', 'count'),
        total_comments=('comments', 'sum') if has_comments else ('cluster', 'count'),
    ).fillna(0)

    resampled['cluster'] = cluster_id
    resampled['cluster_label'] = label
    resampled['avg_likes'] = resampled['total_likes'] / resampled['post_count'].replace(0, np.nan)
    resampled['avg_reach'] = resampled['total_reach'] / resampled['post_count'].replace(0, np.nan)
    resampled['volume_pct_change'] = resampled['post_count'].pct_change().fillna(0)
    resampled['engagement_growth'] = (resampled['total_likes'] + resampled['total_shares']).pct_change().fillna(0)
    resampled['volume_volatility'] = resampled['post_count'].rolling(3, min_periods=1).std().fillna(0)
    rolling_mean = resampled['post_count'].rolling(3, min_periods=1).mean()
    resampled['momentum'] = resampled['post_count'] - rolling_mean

    mean_vol = resampled['post_count'].mean()
    std_vol = resampled['post_count'].std()
    resampled['anomaly_score'] = (resampled['post_count'] - mean_vol) / (std_vol if std_vol > 0 else 1)
    resampled['market_share'] = np.nan
    resampled['lifecycle_state'] = classify_lifecycle(resampled)

    return resampled.reset_index()


def run_temporal_analysis(df, window, config):
    ts_col = config['input']['timestamp_column']
    # Pandas 2.2+: 'M' deprecated for month-end, use 'ME'
    if window == '1M':
        window = '1ME'
    df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')
    df = df.dropna(subset=[ts_col])

    for col in ['likes', 'reach', 'shares', 'comments']:
        if col not in df.columns:
            df[col] = 0

    clusters = sorted(df['cluster'].unique())
    n_workers = _get_n_workers(config)
    print(f"Analysing {len(clusters)} clusters with window='{window}' using {n_workers} workers...")

    col_flags = (
        'Author' in df.columns,
        'likes' in df.columns,
        'reach' in df.columns,
        'shares' in df.columns,
        'comments' in df.columns,
        'cluster_label' in df.columns,
    )

    # Serialize per-cluster DataFrames as parquet bytes for safe pickling
    import io
    task_args = []
    for cid in clusters:
        buf = io.BytesIO()
        df[df['cluster'] == cid].to_parquet(buf, index=False)
        task_args.append((cid, buf.getvalue(), window, col_flags, ts_col))

    if n_workers > 1 and len(clusters) > 4:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            records = list(ex.map(_process_single_cluster, task_args))
    else:
        records = [_process_single_cluster(a) for a in task_args]

    temporal_df = pd.concat(records, ignore_index=True)
    temporal_df = temporal_df.rename(columns={ts_col: 'time_window'})

    # Market share (cross-cluster, must be computed after merge)
    total_per_window = temporal_df.groupby('time_window')['post_count'].transform('sum')
    temporal_df['market_share'] = temporal_df['post_count'] / total_per_window.replace(0, np.nan)

    return temporal_df


def build_frontend_snapshots(temporal_df):
    """Create a frontend-oriented snapshot table from temporal stats."""
    columns = [
        "time_window",
        "cluster",
        "cluster_label",
        "post_count",
        "market_share",
        "volume_pct_change",
        "volume_volatility",
        "momentum",
        "lifecycle_state",
        "anomaly_score",
    ]
    snapshots = temporal_df.copy()
    for col in columns:
        if col not in snapshots.columns:
            snapshots[col] = np.nan
    snapshots = snapshots[columns].copy()
    snapshots["time_window"] = pd.to_datetime(snapshots["time_window"], errors="coerce")
    snapshots = snapshots.dropna(subset=["time_window"])
    snapshots = snapshots.sort_values(["time_window", "cluster"]).reset_index(drop=True)
    return snapshots


def _read_clusters_parquet(path):
    """Read clusters.parquet; avoid PyArrow 'Repetition level histogram' errors."""
    try:
        return pd.read_parquet(path)
    except OSError as e:
        if "histogram" in str(e) or "Repetition" in str(e) or "mismatch" in str(e).lower():
            try:
                import pyarrow.parquet as pq
                table = pq.read_table(path, use_legacy_dataset=True)
                return table.to_pandas()
            except Exception as e2:
                raise RuntimeError(
                    f"Could not read parquet (tried legacy reader): {e2}"
                ) from e
        raise

def main():
    parser = argparse.ArgumentParser(description="Temporal topic analysis")
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--window', default=None, help="Pandas frequency string e.g. 1W, 1M, 1D")
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    config = load_config(config_path)
    config_dir = os.path.dirname(config_path)
    def resolve(p):
        return p if os.path.isabs(p) else os.path.join(config_dir, p)

    window = args.window or config['temporal']['default_window']
    stats_path = resolve(config['output']['temporal_stats'])
    events_path = resolve(config['output']['temporal_events'])
    snapshots_1m_path = resolve(
        config['output'].get('cluster_trend_snapshots_1m', 'output/cluster_trend_snapshots_1M.parquet')
    )
    snapshots_1m_csv_path = resolve(
        config['output'].get('cluster_trend_snapshots_1m_csv', 'output/cluster_trend_snapshots_1M.csv')
    )
    clusters_path = resolve(config['output']['clusters'])

    if os.path.exists(stats_path) and not args.force:
        print(f"Output exists: {stats_path} (use --force to rerun)")
        sys.exit(0)

    if not os.path.exists(clusters_path):
        print(f"ERROR: clusters.parquet not found at {clusters_path}. Run script 1 first.")
        sys.exit(1)

    df = _read_clusters_parquet(clusters_path)
    print(f"Loaded {len(df)} rows from {clusters_path}")

    temporal_df = run_temporal_analysis(df, window, config)
    events_df = detect_events(temporal_df)
    monthly_temporal_df = run_temporal_analysis(df.copy(), '1M', config)
    monthly_snapshots = build_frontend_snapshots(monthly_temporal_df)

    os.makedirs(os.path.dirname(stats_path) or '.', exist_ok=True)
    temporal_df.to_parquet(stats_path, index=False)
    events_df.to_parquet(events_path, index=False)
    os.makedirs(os.path.dirname(snapshots_1m_path) or '.', exist_ok=True)
    monthly_snapshots.to_parquet(snapshots_1m_path, index=False)
    monthly_snapshots.to_csv(snapshots_1m_csv_path, index=False)

    print(f"Saved temporal stats ({len(temporal_df)} rows) to {stats_path}")
    print(f"Saved {len(events_df)} events to {events_path}")
    print(f"Saved monthly frontend snapshots ({len(monthly_snapshots)} rows) to {snapshots_1m_path}")
    print(f"Saved monthly frontend snapshots CSV to {snapshots_1m_csv_path}")

    # Lifecycle summary
    lifecycle_counts = temporal_df.groupby('cluster')['lifecycle_state'].first().value_counts()
    print("\nLifecycle distribution:")
    print(lifecycle_counts.to_string())

    # --- Weekly recency snapshot ---
    recency_cfg = config.get("temporal", {})
    weekly_recency = build_weekly_recency_snapshots(
        df,
        lookback_months=int(recency_cfg.get("recency_lookback_months", 12)),
        min_posts=int(recency_cfg.get("recency_min_posts_per_window", 3)),
        min_density=float(recency_cfg.get("recency_min_density", 0.5)),
        n_workers=_get_n_workers(config),
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


if __name__ == '__main__':
    main()
