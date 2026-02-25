# Temporal Topic Analysis Pipeline - Design Document

**Date:** 2026-02-21
**Project:** InsightAgent - Social Intelligence Analysis
**Goal:** Analyze 50k social media posts with embeddings to discover topics, track evolution over time, and prepare for LLM-powered volatility analysis

---

## Executive Summary

We're building a modular Python pipeline to:
1. **Cluster** 50k posts into ~50 topics using HDBSCAN on Gemini embeddings
2. **Track** topic activity over flexible time windows (default: monthly)
3. **Visualize** cluster maps, temporal heatmaps, and event timelines
4. **Prepare** for future LLM-powered volatility analysis

**Architecture:** 3 standalone Python scripts + config file + cached Parquet outputs

### Research & citations

The clustering (HDBSCAN, UMAP), temporal metrics, lifecycle classification, and event detection methodology used in this pipeline are supported by published work. A dedicated document with full citations and relevance notes is maintained at [docs/plans/2026-02-25-analysis-research-citations.md](docs/plans/2026-02-25-analysis-research-citations.md).

---

## 1. Architecture & Data Flow

### Project Structure
```
InsightAgent/
├── config.yaml                    # Central configuration
├── embeddings.npy                 # Input: Gemini embeddings (50k × 3072)
├── clustered_data_labeled.csv     # Input: Original posts with metadata
├── scripts/
│   ├── 1_cluster_embeddings.py    # HDBSCAN clustering + UMAP
│   ├── 2_temporal_analysis.py     # Time-based metrics + event detection
│   └── 3_visualize_clusters.py    # Interactive Plotly visualizations
└── output/
    ├── clusters.parquet           # Clustered data with labels
    ├── temporal_stats.parquet     # Time-windowed statistics
    ├── temporal_events.parquet    # Detected spikes/drops
    └── visualizations/
        ├── cluster_map.html       # UMAP scatter plot
        ├── temporal_heatmap.html  # Activity heatmap
        ├── event_timeline.html    # Timeline with annotations
        └── volatility_dashboard.html  # 4-panel metrics view
```

### Data Flow
1. **Script 1**: `embeddings.npy` + `clustered_data_labeled.csv` → HDBSCAN clustering → `clusters.parquet`
2. **Script 2**: `clusters.parquet` + time config → temporal aggregation + metrics → `temporal_stats.parquet`
3. **Script 3**: Both parquet files → interactive visualizations → HTML files

### Key Design Choices
- **Parquet format**: 5-10x faster than CSV, preserves types, supports compression
- **Cached results**: Each script saves output for quick downstream iterations
- **Config-driven**: Change parameters without editing code
- **Standalone scripts**: Each runs independently with `python scripts/X_name.py`

---

## 2. Configuration Design

**config.yaml:**
```yaml
# Input files
input:
  embeddings: "embeddings.npy"
  data: "clustered_data_labeled.csv"
  text_column: "text"
  timestamp_column: "timestamp"

# Clustering parameters
clustering:
  algorithm: "hdbscan"
  min_cluster_size: 50        # Minimum posts per cluster
  min_samples: 10             # HDBSCAN sensitivity
  target_clusters: 50         # Approximate target (HDBSCAN finds natural boundaries)

# UMAP for visualization
umap:
  n_components: 2
  n_neighbors: 15
  min_dist: 0.1
  metric: "cosine"

# Temporal analysis
temporal:
  default_window: "1M"        # 1 Month (Pandas frequency string)
  # Can override: "1W" (week), "1D" (day), "3M" (quarter), etc.

# Output paths
output:
  dir: "output"
  clusters: "output/clusters.parquet"
  temporal_stats: "output/temporal_stats.parquet"
  temporal_events: "output/temporal_events.parquet"
  viz_dir: "output/visualizations"

# LLM labeling (for future volatility analysis)
llm:
  enabled: false
  provider: "anthropic"
  model: "claude-3-5-sonnet-20241022"
  sample_size: 20        # Posts to sample per cluster
```

**Flexibility:**
- Change time windows: `1W` (weekly), `1M` (monthly), `1D` (daily)
- Adjust HDBSCAN sensitivity without code changes
- Enable/disable LLM features via config flag

---

## 3. Script 1 - Clustering (`1_cluster_embeddings.py`)

### Responsibilities
1. Load embeddings (50k × 3072) and original CSV data
2. Run HDBSCAN clustering on embeddings
3. Generate UMAP 2D coordinates for visualization
4. Create automatic cluster labels (top TF-IDF terms per cluster)
5. Save results to `output/clusters.parquet`

HDBSCAN and UMAP are widely used for embedding-based topic discovery; see research addendum for citations.

### Core Algorithm
```python
def cluster_embeddings(embeddings, config):
    # HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=config['min_cluster_size'],
        min_samples=config['min_samples'],
        metric='euclidean',
        cluster_selection_method='eom'  # Excess of Mass
    )
    cluster_labels = clusterer.fit_predict(embeddings)

    # UMAP for 2D visualization
    reducer = umap.UMAP(**config['umap'])
    umap_coords = reducer.fit_transform(embeddings)

    return cluster_labels, umap_coords

def generate_cluster_labels(df, n_top_terms=3):
    # TF-IDF on text per cluster
    # Returns: {cluster_id: "keyword1, keyword2, keyword3"}
```

### Output Schema (`clusters.parquet`)
- All original CSV columns
- `cluster` (int): Cluster ID (-1 for noise/outliers)
- `umap_x`, `umap_y` (float): 2D visualization coordinates
- `cluster_label` (str): Auto-generated topic label
- `cluster_size` (int): Number of posts in this cluster

### Performance
- HDBSCAN faster than K-Means for high-dimensional data
- Progress bars with `tqdm` for long operations
- Resume capability: skip if output exists (use `--force` to rerun)

---

## 4. Script 2 - Advanced Temporal Analysis (`2_temporal_analysis.py`)

### Responsibilities
1. Load clustered data from parquet
2. Parse timestamps and create flexible time windows
3. Calculate sophisticated metrics: volatility, growth rates, momentum
4. Detect events: spikes, sentiment shifts, viral moments
5. Classify topic lifecycle: emerging, trending, stable, declining, dormant
6. Save enriched data for LLM analysis

### Core Metrics

**Base Aggregation:**
- Post volume per cluster per time window
- Unique authors, total engagement (likes + shares + comments)
- Sentiment distribution
- Reach and impressions

**Advanced Metrics:**
- `volume_pct_change` (%): Period-over-period growth rate
- `engagement_growth` (%): Engagement trend
- `volume_volatility`: Rolling standard deviation of volume
- `momentum`: Deviation from 3-period moving average
- `anomaly_score`: Z-score for spike detection
- `market_share` (%): Cluster's share of total activity
- `lifecycle_state`: ['emerging', 'trending', 'stable', 'declining', 'dormant']

### Lifecycle Classification
```python
def classify_lifecycle(cluster_time_series):
    recent_growth = cluster_time_series['volume_pct_change'].tail(3).mean()
    recent_volume = cluster_time_series['post_count'].tail(3).mean()
    overall_volume = cluster_time_series['post_count'].mean()

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
```

### Event Detection
```python
def detect_events(df, spike_threshold=2.0):
    # Identify anomaly_score > 2.0 (2 std devs from mean)
    events = df[df['anomaly_score'].abs() > spike_threshold]
    events['event_type'] = events['anomaly_score'].apply(
        lambda x: 'spike' if x > 0 else 'drop'
    )
    return events
```

Lifecycle and event detection are informed by temporal topic and anomaly-detection literature; see research addendum.

### Output Schema

**`temporal_stats.parquet`:**
- `cluster` (int), `time_window` (datetime), `cluster_label` (str)
- `post_count`, `unique_authors`
- `total_likes`, `total_reach`, `total_shares`, `total_comments`
- `avg_likes`, `avg_reach`
- `sentiment_distribution` (dict): {'positive': X, 'neutral': Y, 'negative': Z}
- `volume_pct_change`, `engagement_growth`, `volume_volatility`
- `momentum`, `anomaly_score`, `market_share`
- `lifecycle_state`

**`temporal_events.parquet`:**
- All detected spikes/drops (anomaly_score > 2.0)
- Event metadata: timestamp, cluster, event_type, magnitude
- Top 10 sample posts from each event for context

### Flexible Time Windows
```bash
python scripts/2_temporal_analysis.py --window "1W"  # Weekly
python scripts/2_temporal_analysis.py --window "1M"  # Monthly (default)
python scripts/2_temporal_analysis.py --window "1D"  # Daily
```

### LLM Preparation
This structure enables future LLM analysis like:
- "Cluster 12 was dormant, then spiked 300% in week 3 with positive sentiment shift"
- "Cluster 5 shows high volatility (±40%) with declining momentum"
- "Emerging topics: Clusters 8, 15, 23 showing sustained growth >50%"

---

## 5. Script 3 - Interactive Visualizations (`3_visualize_clusters.py`)

### Responsibilities
1. Load clustered data and temporal stats
2. Create 4 interactive visualizations using Plotly
3. Export as standalone HTML files (no Python needed to view)
4. Optionally open in browser with `--open-browser` flag

### Visualizations

**1. Cluster Map** (`cluster_map.html`)
- UMAP scatter plot, color-coded by cluster
- Size represents cluster size
- Hover: cluster label, sample text, engagement metrics
- Interactive: zoom, pan, click to filter

**2. Temporal Heatmap** (`temporal_heatmap.html`)
- Rows = clusters, Columns = time periods
- Color intensity = post volume
- Lifecycle badges: 🟢 emerging, 🔥 trending, 🔵 stable, 🔻 declining, ⚫ dormant
- Hover: volume, engagement, sentiment, lifecycle state
- Sorted by total activity or emergence date

**3. Event Timeline** (`event_timeline.html`)
- Multi-line chart (one per cluster)
- Spike annotations with cluster labels
- Hover shows all clusters at that time point
- Toggle clusters on/off, zoom to time ranges

**4. Volatility Dashboard** (`volatility_dashboard.html`)
- 4-panel view:
  1. Volume volatility by cluster
  2. Growth rate trends
  3. Market share evolution
  4. Lifecycle state distribution

### Features
- Fully interactive (Plotly)
- Export buttons (PNG, SVG for presentations)
- Responsive design (mobile/tablet compatible)
- Shareable HTML files

---

## 6. Error Handling & Edge Cases

### Data Validation
```python
def validate_inputs(embeddings_path, data_path, config):
    # Check file existence
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings not found: {embeddings_path}")

    # Check shape consistency
    embeddings = np.load(embeddings_path)
    data = pd.read_csv(data_path)

    if len(embeddings) != len(data):
        raise ValueError(f"Mismatch: {len(embeddings)} embeddings vs {len(data)} rows")

    # Check required columns
    required_cols = ['text', 'timestamp', 'likes', 'reach']
    missing = set(required_cols) - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Validate timestamps
    pd.to_datetime(data['timestamp'])  # Raises if invalid
```

### Graceful Degradation
- Missing engagement metrics → fill with 0
- HDBSCAN noise cluster (-1) → label as "Uncategorized/Noise"
- Empty time windows → fill with 0 post_count
- Missing sentiment field → handle both 'Sentiment' and 'sentiment'

### Progress Tracking
- `tqdm` progress bars for long operations
- Informative logging: cluster counts, sizes, noise percentage
- Resume capability: skip if output exists, use `--force` to rerun

### Edge Cases Handled
- ✅ Empty clusters (min_cluster_size prevents)
- ✅ Noise points (HDBSCAN assigns -1, labeled as "Uncategorized")
- ✅ Missing timestamps (filter out with warning)
- ✅ Time periods with zero activity (fill with 0)
- ✅ Single-post clusters (adjustable via min_cluster_size)
- ✅ Sentiment field variations

### Command-Line Interface
```bash
# Script 1: Clustering
python scripts/1_cluster_embeddings.py --config config.yaml --force

# Script 2: Temporal analysis with custom window
python scripts/2_temporal_analysis.py --window "1W" --config config.yaml

# Script 3: Visualization
python scripts/3_visualize_clusters.py --config config.yaml --open-browser
```

---

## 7. Future Extensibility - LLM Integration

### Preparation for Volatility Analysis

The pipeline prepares for LLM-powered volatility analysis:

**Future Script 4:** `4_llm_volatility_analysis.py`
```python
def analyze_cluster_volatility(cluster_id, temporal_data, llm_client):
    """Use Claude to interpret volatility patterns"""

    cluster_timeline = temporal_data[temporal_data['cluster'] == cluster_id]

    context = f"""
    Cluster: {cluster_timeline['cluster_label'].iloc[0]}
    Volatility: {cluster_timeline['volume_volatility'].mean():.2f}
    Growth: {cluster_timeline['volume_pct_change'].mean():.1%}
    Lifecycle: {cluster_timeline['lifecycle_state'].mode()[0]}
    Events: {events_summary}
    Sample posts: {sample_posts}
    """

    response = llm_client.messages.create(
        model="claude-3-5-sonnet-20241022",
        messages=[{"role": "user", "content": f"Analyze this topic's volatility: {context}"}]
    )

    return response.content
```

### LLM Config
```yaml
llm:
  enabled: true
  provider: "anthropic"
  model: "claude-3-5-sonnet-20241022"
  analysis_types:
    - volatility_explanation
    - event_causation
    - sentiment_analysis
    - predictive_trends
  sample_size: 20  # Posts per cluster for context
  batch_size: 10   # Parallel cluster analysis
```

### Easy Future Additions
- **Script 4**: LLM volatility analysis
- **Script 5**: Predictive modeling (forecast cluster trends)
- **Script 6**: Executive summary PDF report
- Sentiment time series analysis
- Author influence tracking
- Hashtag trend extraction
- Geographic analysis (if location added)

### No Rework Needed
When ready for LLM analysis:
1. Run Scripts 1-3 (as designed)
2. Add Script 4 (LLM analysis)
3. Use same parquet files as input
4. Enable via config flag

---

## Dependencies

**Required Python Packages:**
```
numpy
pandas
pyarrow  # For Parquet
hdbscan
umap-learn
scikit-learn
plotly
tqdm
pyyaml
anthropic  # For future LLM integration
```

**Installation:**
```bash
pip install numpy pandas pyarrow hdbscan umap-learn scikit-learn plotly tqdm pyyaml anthropic
```

---

## Success Criteria

✅ **Clustering Quality:**
- ~50 meaningful clusters (HDBSCAN finds natural boundaries)
- <10% noise points
- Clear topic separation in UMAP plot

✅ **Temporal Analysis:**
- Flexible time windows (daily, weekly, monthly)
- Accurate event detection (spikes >2σ)
- Valid lifecycle classification for all clusters

✅ **Visualizations:**
- Interactive HTML files load in browser without errors
- Hover tooltips show relevant metrics
- Lifecycle states visually distinct

✅ **Performance:**
- Clustering completes in <30 minutes (50k posts)
- Temporal analysis <5 minutes
- Visualization generation <2 minutes
- Subsequent runs use cached results (seconds)

✅ **Extensibility:**
- Easy to add Script 4 (LLM analysis) without modifying existing scripts
- Config changes don't require code changes
- New metrics can be added to Script 2 without breaking downstream

---

## Next Steps

1. **Implementation Planning**: Create detailed plan with file-by-file implementation steps
2. **Development**: Build Scripts 1-3 with comprehensive error handling
3. **Testing**: Validate on full 50k dataset
4. **Documentation**: Add README with usage examples
5. **Future**: Add Script 4 for LLM-powered volatility analysis

---

## Appendix: Data Schema Reference

### Input Data
- `embeddings.npy`: (50k, 3072) float32 array
- `clustered_data_labeled.csv`: 50k rows with columns: text, timestamp, likes, reach, shares, comments, Author, Sentiment, etc.

### Output Data
- `clusters.parquet`: 50k rows + cluster, umap_x, umap_y, cluster_label, cluster_size
- `temporal_stats.parquet`: ~50 clusters × N time windows with all metrics
- `temporal_events.parquet`: Detected spikes/drops with sample posts
- HTML visualizations: Standalone interactive files

---

**End of Design Document**
