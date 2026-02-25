# Branch 3: Dashboard UI — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Branch:** `feature/dashboard-ui`
**Depends on:** Both `feature/data-pipeline` AND `feature/forecast-engine` merged into `main`.

Before starting, rebase onto main to get `THEME_CLUSTER_MAP` and `CLUSTER_THEME_LABELS` from `server.py`, and the weekly recency CSV from branch 1:
```bash
git fetch origin
git rebase main
```

**Goal:** Surface theme names in the dashboard cluster labels (replacing raw TF-IDF keywords), and add a monthly/weekly toggle to the trend chart that switches to weekly-resolution data for the last 12 months when density permits.

**Architecture:** `server.py` already has `THEME_CLUSTER_MAP` and `CLUSTER_THEME_LABELS` loaded (from branch 2). This branch wires those into the API responses, adds a new `/api/snapshots/weekly-recent` endpoint, and updates the React frontend with the toggle UI.

**Tech Stack:** FastAPI, React 18, TypeScript, Tailwind CSS, Recharts.

---

## Task 7: Surface theme names in the dashboard

**Files:**
- Modify: `server.py` — enrich `/api/clusters` with `theme_name`, add `/api/snapshots/weekly-recent`
- Modify: `frontend/src/lib/api.ts` — add `theme_name` to `ClusterInfo`, new fetch
- Modify: `frontend/src/pages/Index.tsx` — show theme names in cluster pills, add toggle

---

### Step 1: Add `theme_name` to `/api/clusters` in `server.py`

In `get_clusters()`, find the dict where each cluster is constructed and add `theme_name`:

```python
cid = int(row["cluster"])
clusters.append({
    "id": f"cluster-{cid}",
    "cluster": cid,
    "cluster_label": str(row.get("cluster_label", f"Cluster {cid}")),
    "theme_name": CLUSTER_THEME_LABELS.get(cid, ""),   # ADD — empty string if unmapped
    "size": int(row.get("post_count", 0)),
    # ... rest of existing fields unchanged ...
})
```

### Step 2: Add `/api/snapshots/weekly-recent` endpoint to `server.py`

After the `SNAPSHOTS_CSV_PATH` assignment, add:

```python
WEEKLY_RECENT_CSV_PATH = _resolve(
    CONFIG["output"].get(
        "cluster_trend_snapshots_1w_recent_csv",
        "output/cluster_trend_snapshots_1W_recent.csv",
    )
)
```

Add the endpoint after the existing `/api/snapshots`:

```python
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
```

### Step 3: Update `/api/health` to report weekly dense cluster count

In `health()`, add `weekly_dense_clusters`:

```python
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
        "weekly_dense_clusters": weekly_clusters,   # ADD
    }
```

### Step 4: Verify `server.py` changes manually

```bash
cd /Users/james/momentum
uvicorn server:app --port 8001 &

# Check cluster theme names appear
curl -s http://localhost:8001/api/clusters | python3 -c "
import json, sys
d = json.load(sys.stdin)
for c in d['clusters'][:8]:
    print(f\"  {c['cluster']:>3}: theme='{c.get('theme_name', '')}' | label={c['cluster_label']}\")
"

# Check weekly endpoint
curl -s http://localhost:8001/api/snapshots/weekly-recent | python3 -c "
import json, sys
d = json.load(sys.stdin)
print('Weekly snapshots:', len(d))
if d:
    print('Example:', json.dumps(d[0], indent=2))
"

kill %1
```

### Step 5: Commit server changes

```bash
git add server.py
git commit -m "feat: add theme_name to cluster API, add /api/snapshots/weekly-recent endpoint"
```

---

## Task 8: Update frontend types and API client

**Files:**
- Modify: `frontend/src/lib/api.ts`

### Step 1: Add `theme_name` to `ClusterInfo` and `Snapshot`, add `fetchWeeklySnapshots`

Find the `ClusterInfo` interface and add `theme_name`:
```typescript
export interface ClusterInfo {
  id: string;
  cluster: number;
  cluster_label: string;
  theme_name: string;        // ADD
  size: number;
  market_share: number;
  volume_pct_change: number;
  volume_volatility: number;
  momentum: number;
  lifecycle: string;
  anomaly_score: number;
}
```

Find the `Snapshot` interface and add `window_type` + `theme_name`:
```typescript
export interface Snapshot {
  cluster_id: string;
  cluster_label: string;
  theme_name?: string;          // ADD
  time_window: string;
  post_count: number;
  market_share: number;
  momentum: number;
  volatility: number;
  window_type?: "1M" | "1W";    // ADD
}
```

Add the new fetch function at the end of the exports:
```typescript
export async function fetchWeeklySnapshots(): Promise<Snapshot[]> {
  return request("/snapshots/weekly-recent");
}
```

### Step 2: Commit

```bash
cd frontend && npm run build 2>&1 | tail -5
cd ..
git add frontend/src/lib/api.ts
git commit -m "feat: add WeeklySnapshot type, fetchWeeklySnapshots, theme_name to ClusterInfo"
```

---

## Task 9: Add monthly/weekly toggle + theme names to `Index.tsx`

**Files:**
- Modify: `frontend/src/pages/Index.tsx`

This is the biggest change. Work through it section by section.

### Step 1: Read the current file first

Before editing, read `frontend/src/pages/Index.tsx` to understand the exact current structure — import order, state declarations, useMemo shapes, JSX layout of the Monthly Trends card.

### Step 2: Add imports at the top

```typescript
import { fetchWeeklySnapshots, type Snapshot } from "@/lib/api";
```
(Already importing `Snapshot` for monthly — just ensure `fetchWeeklySnapshots` is added.)

### Step 3: Add state declarations

After the existing `displaySnapshots` / `selectedMetric` state, add:

```typescript
const [granularity, setGranularity] = useState<"monthly" | "weekly">("monthly");
const [weeklySnapshots, setWeeklySnapshots] = useState<Snapshot[]>([]);
```

### Step 4: Add `fetchWeeklySnapshots` to the `useEffect` data-loading block

Inside the existing `useEffect` that fetches snapshots/clusters, add:

```typescript
fetchWeeklySnapshots()
  .then(setWeeklySnapshots)
  .catch(() => {}),   // silent fail if file doesn't exist yet
```

### Step 5: Add derived memos

After the existing `clusterOptions` memo, add:

```typescript
// Which clusters have weekly data
const weeklyClusterIds = useMemo(
  () => new Set(weeklySnapshots.map((s) => s.cluster_id)),
  [weeklySnapshots]
);
const hasWeeklyData = weeklySnapshots.length > 0;

// In weekly mode, only show clusters that have sufficient density
const activeClusterOptions = useMemo(() => {
  if (granularity === "weekly" && hasWeeklyData) {
    return clusterOptions.filter((c) => weeklyClusterIds.has(c.id));
  }
  return clusterOptions;
}, [clusterOptions, granularity, hasWeeklyData, weeklyClusterIds]);

// Source of truth for the chart — switches between monthly and weekly data
const activeSnapshots = granularity === "weekly" && hasWeeklyData
  ? weeklySnapshots
  : displaySnapshots;
```

### Step 6: Update `trendData` memo to use `activeSnapshots`

Replace the current `trendData` useMemo. The key change is replacing `displaySnapshots` with `activeSnapshots` and formatting the time label based on granularity:

```typescript
const trendData = useMemo(() => {
  const ids = selectedClusterIds.length
    ? selectedClusterIds
    : activeClusterOptions.slice(0, 3).map((c) => c.id);

  const timeWindows = [...new Set(activeSnapshots.map((s) => s.time_window))].sort();

  return timeWindows.map((tw) => {
    const row: Record<string, string | number> = {
      // Weekly: show full date (YYYY-MM-DD); Monthly: show YYYY-MM
      time_window: granularity === "weekly" ? tw.slice(0, 10) : tw.slice(0, 7),
    };
    ids.forEach((id) => {
      const snap = activeSnapshots.find((s) => s.cluster_id === id && s.time_window === tw);
      if (snap) row[id] = snap[selectedMetric] ?? 0;
    });
    return row;
  });
}, [selectedClusterIds, selectedMetric, activeClusterOptions, activeSnapshots, granularity]);
```

### Step 7: Update cluster pill labels to prefer `theme_name`

In the cluster pill buttons (inside the Monthly Trends card), find where `c.label` is displayed and change to:

```tsx
{c.theme_name || c.label}
```

Also update `clusterOptions` memo to carry `theme_name`:

```typescript
const clusterOptions = useMemo(
  () =>
    displaySnapshots
      .filter(...)
      .map((s) => ({
        id: s.cluster_id,
        label: s.cluster_label,
        theme_name: (s as Snapshot).theme_name || "",   // ADD
      })),
  [displaySnapshots]
);
```

### Step 8: Add the granularity toggle UI

In the Monthly Trends card header section, next to the existing metric buttons (post_count, market_share, etc.), add the toggle:

```tsx
{/* Monthly / Weekly toggle — hidden until weekly data is available */}
{hasWeeklyData && (
  <div className="flex items-center gap-1 rounded border border-border p-0.5">
    <button
      onClick={() => setGranularity("monthly")}
      className={cn(
        "rounded px-2.5 py-1 text-[10px] font-medium transition-colors",
        granularity === "monthly"
          ? "bg-foreground text-background"
          : "text-muted-foreground hover:text-foreground"
      )}
    >
      Monthly
    </button>
    <button
      onClick={() => setGranularity("weekly")}
      className={cn(
        "rounded px-2.5 py-1 text-[10px] font-medium transition-colors",
        granularity === "weekly"
          ? "bg-foreground text-background"
          : "text-muted-foreground hover:text-foreground"
      )}
    >
      Weekly (recent)
    </button>
  </div>
)}
```

### Step 9: Add the chart subtitle

Below the card title "Monthly Trends", add a subtitle paragraph:

```tsx
<p className="mt-0.5 text-[11px] text-muted-foreground">
  {granularity === "weekly"
    ? `Weekly · last 12 months · ${weeklyClusterIds.size} dense cluster${weeklyClusterIds.size !== 1 ? "s" : ""}`
    : `Monthly · full history`}
</p>
```

### Step 10: Replace `clusterOptions` refs in pill map with `activeClusterOptions`

Find everywhere `clusterOptions` is used to render the cluster pill buttons and change to `activeClusterOptions`. This ensures that in weekly mode, only density-qualified clusters appear as pills.

Also update the `selectedClusterIds` reset logic — when switching granularity, clear the selection if selected clusters don't exist in the new option set:

```typescript
useEffect(() => {
  if (selectedClusterIds.length > 0) {
    const validIds = new Set(activeClusterOptions.map((c) => c.id));
    const stillValid = selectedClusterIds.filter((id) => validIds.has(id));
    if (stillValid.length !== selectedClusterIds.length) {
      setSelectedClusterIds(stillValid);
    }
  }
}, [granularity]);
```

### Step 11: Build and verify no TypeScript errors

```bash
cd /Users/james/momentum/frontend
npm run build
```
Fix any TypeScript errors before committing.

### Step 12: Smoke test end-to-end

```bash
cd /Users/james/momentum
uvicorn server:app --reload --port 8000 &
cd frontend && npm run dev &
```

Open `http://localhost:5173` and verify:
1. Monthly Trends card shows "Monthly · full history" subtitle
2. Cluster pills show theme names (e.g. "AlphaFold Scientific Impact") where mapped, raw label otherwise
3. If `output/cluster_trend_snapshots_1W_recent.csv` exists and has dense clusters: the monthly/weekly toggle appears
4. Switching to "Weekly (recent)" — chart shows weekly resolution for last 12 months only
5. Cluster pills in weekly mode only show dense clusters
6. Switching back to "Monthly" restores full history view and all cluster pills

### Step 13: Commit

```bash
cd /Users/james/momentum
git add frontend/src/pages/Index.tsx frontend/src/lib/api.ts
git commit -m "feat: theme names in cluster pills, monthly/weekly trend toggle with density gating"
```

---

## Task 10: Run all tests + final validation

**Step 1: Run all tests**

```bash
cd /Users/james/momentum
python -m pytest tests/ -v
```
Expected: all PASS (including tests from branches 1 and 2 that are now on main).

**Step 2: Full API smoke test**

```bash
uvicorn server:app --port 8000 &

# Health check — should show weekly_dense_clusters
curl -s http://localhost:8000/api/health | python3 -m json.tool

# Clusters — should show theme_name
curl -s http://localhost:8000/api/clusters | python3 -c "
import json, sys
d = json.load(sys.stdin)
print('Total clusters:', d['total'])
mapped = [(c['cluster'], c['theme_name'], c['cluster_label'])
          for c in d['clusters'] if c.get('theme_name')]
print(f'{len(mapped)} clusters have theme names:')
for cid, tname, label in sorted(mapped):
    print(f'  {cid}: {tname} (label: {label})')
"

# Weekly snapshots
curl -s http://localhost:8000/api/snapshots/weekly-recent | python3 -c "
import json, sys
d = json.load(sys.stdin)
print('Weekly snapshots:', len(d))
clusters = list({s['cluster_id'] for s in d})
print('Clusters:', clusters[:10])
"

# Prediction using theme map
curl -s -X POST http://localhost:8000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{\"event_text\": \"AlphaFold discovers new protein structure for cancer treatment\"}' \
  | python3 -c "
import json, sys
d = json.load(sys.stdin)
print('Direction:', d['predicted_direction'])
print('Confidence:', d['confidence'])
for t in d.get('activated_themes', []):
    print(f\"  theme {t['theme_id']} → cluster {t.get('mapped_cluster_id')} | {t['theme_name']}\")
"

kill %1
```

**Step 3: Commit and merge**

```bash
git add -A
git commit -m "feat: complete dashboard UI — theme names, weekly trend toggle, full smoke test passing"

git checkout main
git merge feature/dashboard-ui
```

---

## Merge Order (All Branches)

```
main
 └─ feature/data-pipeline      → merge first (produces output files all others need)
     └─ feature/forecast-engine → merge second (uses theme_activations, produces theme_cluster_map)
         └─ feature/dashboard-ui → merge last (uses theme_cluster_map + weekly CSV)
```

After all three are on main:
```bash
git log --oneline main | head -15
python -m pytest tests/ -v
uvicorn server:app --reload --port 8000
```
