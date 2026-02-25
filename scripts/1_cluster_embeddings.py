"""
Script 1: Cluster Embeddings
- Loads Gemini embeddings (50k x 3072) + CSV data
- Runs HDBSCAN clustering
- Generates UMAP 2D coordinates
- Creates TF-IDF cluster labels
- Saves output/clusters.parquet
Uses multiple cores for HDBSCAN and UMAP when configured.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use all CPUs for Numba (UMAP); set before importing umap
def _cpu_count():
    """Number of available CPUs (respects cgroups/affinity on Linux)."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def _set_numba_threads(n=None):
    if n is None:
        n = _cpu_count()
    n = max(1, n)
    os.environ["NUMBA_NUM_THREADS"] = str(n)
    return n

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from scripts.cluster_label_utils import clean_label_text


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def validate_inputs(embeddings_path, data_path, config, pbar=None):
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings not found: {embeddings_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found: {data_path}")

    if pbar:
        pbar.set_description("Loading embeddings & CSV")
    embeddings = np.load(embeddings_path)
    data = pd.read_csv(data_path, low_memory=False)

    if len(embeddings) != len(data):
        raise ValueError(f"Mismatch: {len(embeddings)} embeddings vs {len(data)} rows")

    text_col = config['input']['text_column']
    ts_col = config['input']['timestamp_column']
    required_cols = [text_col, ts_col]
    missing = set(required_cols) - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Validate timestamps
    pd.to_datetime(data[ts_col], errors='coerce')

    return embeddings, data


def reduce_embeddings(embeddings, config, pbar=None):
    """Optionally reduce embedding dims with PCA before clustering (faster HDBSCAN)."""
    pca_cfg = config.get('pca') or {}
    if not pca_cfg.get('enabled', False):
        return embeddings
    if pbar:
        pbar.set_description("PCA reduction")
    from sklearn.decomposition import PCA
    n = pca_cfg.get('n_components', 128)
    n = min(n, embeddings.shape[0], embeddings.shape[1])
    pca = PCA(n_components=n, random_state=42)
    reduced = pca.fit_transform(embeddings)
    var = pca.explained_variance_ratio_.sum()
    if pbar:
        pbar.set_postfix_str(f"retained {100*var:.1f}% variance")
    return reduced.astype(np.float32)


def generate_cluster_labels(df, text_column, n_top_terms=3, pbar=None):
    """Generate TF-IDF keyword labels per cluster."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    labels = {}
    unique_clusters = sorted(df['cluster'].unique())
    if pbar:
        pbar.set_description("TF-IDF cluster labels")

    for cluster_id in tqdm(unique_clusters, desc="Cluster labels", leave=False):
        if cluster_id == -1:
            labels[-1] = "Uncategorized/Noise"
            continue

        cluster_texts = df[df['cluster'] == cluster_id][text_column].fillna("").tolist()
        if not cluster_texts:
            labels[cluster_id] = f"Cluster {cluster_id}"
            continue

        try:
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english', max_df=0.9, min_df=2)
            tfidf_matrix = vectorizer.fit_transform(cluster_texts)
            feature_names = vectorizer.get_feature_names_out()
            mean_tfidf = tfidf_matrix.mean(axis=0).A1
            top_indices = mean_tfidf.argsort()[-n_top_terms:][::-1]
            top_terms = [feature_names[i] for i in top_indices]
            labels[cluster_id] = clean_label_text(top_terms, n_top=3)
        except Exception:
            labels[cluster_id] = f"Cluster {cluster_id}"

    return labels


def run_clustering(embeddings, config, pbar=None):
    import hdbscan
    import umap

    n_jobs = config.get('clustering', {}).get('n_jobs', -1)

    if pbar:
        pbar.set_description("HDBSCAN clustering")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=config['clustering']['min_cluster_size'],
        min_samples=config['clustering']['min_samples'],
        metric='euclidean',
        cluster_selection_method='eom',
        core_dist_n_jobs=n_jobs,
    )
    cluster_labels = clusterer.fit_predict(embeddings)

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = (cluster_labels == -1).sum()
    if pbar:
        pbar.set_postfix_str(f"{n_clusters} clusters, {100*n_noise/len(cluster_labels):.1f}% noise")
        pbar.update(1)

    # UMAP uses Numba; NUMBA_NUM_THREADS (set at startup) uses all cores
    if pbar:
        pbar.set_description("UMAP 2D reduction")
        pbar.set_postfix_str("")
    reducer = umap.UMAP(
        n_components=config['umap']['n_components'],
        n_neighbors=config['umap']['n_neighbors'],
        min_dist=config['umap']['min_dist'],
        metric=config['umap']['metric'],
        verbose=False,
    )
    umap_coords = reducer.fit_transform(embeddings)
    if pbar:
        pbar.update(1)

    return cluster_labels, umap_coords


def main():
    parser = argparse.ArgumentParser(description="Cluster embeddings with HDBSCAN + UMAP")
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--force', action='store_true', help='Rerun even if output exists')
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    config = load_config(config_path)
    # Resolve paths relative to config file's directory (repo root when config is in root)
    config_dir = os.path.dirname(config_path)
    def resolve(path):
        return path if os.path.isabs(path) else os.path.join(config_dir, path)
    embeddings_path = resolve(config['input']['embeddings'])
    data_path = resolve(config['input']['data'])
    output_path = resolve(config['output']['clusters'])

    if os.path.exists(output_path) and not args.force:
        print(f"Output already exists: {output_path} (use --force to rerun)")
        sys.exit(0)

    n_cores = _cpu_count()
    max_cores = config.get('max_cores') or 0
    effective_cores = min(n_cores, max_cores) if max_cores else n_cores
    if max_cores:
        os.environ["NUMBA_NUM_THREADS"] = str(effective_cores)
    n_jobs_cfg = config.get('clustering', {}).get('n_jobs', -1)
    n_jobs_effective = effective_cores if n_jobs_cfg == -1 else max(1, min(n_jobs_cfg, effective_cores))
    numba_threads = os.environ.get("NUMBA_NUM_THREADS", str(effective_cores))
    print(f"Cores: {n_cores} available, using {effective_cores} (max_cores={max_cores or 'all'}) | HDBSCAN: {n_jobs_effective} job(s) | UMAP (Numba): {numba_threads} thread(s)")

    # Overall progress: Load -> [PCA] -> HDBSCAN -> UMAP -> Labels -> Save (5 or 6 steps)
    n_steps = 6 if config.get('pca', {}).get('enabled', False) else 5

    with tqdm(total=n_steps, desc="Pipeline", unit="step", dynamic_ncols=True) as pbar:
        embeddings, data = validate_inputs(embeddings_path, data_path, config, pbar=pbar)
        pbar.update(1)

        embeddings_for_clustering = reduce_embeddings(embeddings, config, pbar=pbar)
        if config.get('pca', {}).get('enabled', False):
            pbar.update(1)

        cluster_labels, umap_coords = run_clustering(embeddings_for_clustering, config, pbar=pbar)

        data['cluster'] = cluster_labels
        data['umap_x'] = umap_coords[:, 0]
        data['umap_y'] = umap_coords[:, 1]

        text_col = config['input']['text_column']
        cluster_label_map = generate_cluster_labels(data, text_col, pbar=pbar)
        data['cluster_label'] = data['cluster'].map(cluster_label_map)
        pbar.update(1)

        pbar.set_description("Saving parquet")
        output_dir = resolve(config['output']['dir'])
        os.makedirs(output_dir, exist_ok=True)
        # Normalize engagement columns for script 2 (likes, reach, shares, comments)
        for dst, src_candidates in [
            ('likes', ['likes', 'Likes', 'Facebook Likes', 'X Likes']),
            ('reach', ['reach', 'Reach (new)']),
            ('shares', ['shares', 'Shares', 'Facebook Shares', 'Social Shares']),
            ('comments', ['comments', 'Comments', 'Facebook Comments']),
        ]:
            if dst not in data.columns:
                for c in src_candidates:
                    if c in data.columns:
                        data[dst] = pd.to_numeric(data[c], errors='coerce').fillna(0)
                        break
            if dst not in data.columns:
                data[dst] = 0
        # Keep only columns safe for parquet (avoids PyArrow "Repetition level histogram" errors)
        keep = [config['input']['timestamp_column'], config['input']['text_column'],
                'cluster', 'umap_x', 'umap_y', 'cluster_label', 'cluster_size',
                'likes', 'reach', 'shares', 'comments', 'Author', 'Sentiment', 'sentiment']
        keep = [c for c in keep if c in data.columns]
        out_df = data[keep].copy()
        for col in out_df.select_dtypes(include=['object']).columns:
            out_df[col] = out_df[col].astype(str)
        out_df.to_parquet(output_path, index=False)
        pbar.update(1)
        pbar.set_postfix_str(f"→ {output_path}")

    print(f"Saved {len(data)} rows to {output_path}")


if __name__ == '__main__':
    _set_numba_threads()
    main()
