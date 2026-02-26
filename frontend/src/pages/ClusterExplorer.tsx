import { useState, useMemo, useEffect } from "react";
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer,
} from "recharts";
import { DashboardLayout } from "@/components/DashboardLayout";
import { ClusterDetailPanel } from "@/components/ClusterDetailPanel";
import {
  fetchClusters, fetchSnapshots,
  type ClusterInfo, type Snapshot,
} from "@/lib/api";
import { mockClusters, mockClusterSnapshots } from "@/lib/mock-data";
import { cn } from "@/lib/utils";

const statusStyles: Record<string, string> = {
  Emerging: "text-status-emerging",
  Trending: "text-status-trending",
  Stable: "text-status-stable",
  Declining: "text-status-declining",
  Dormant: "text-status-dormant",
};

const tooltipStyle = {
  background: "hsl(0, 0%, 100%)",
  border: "1px solid hsl(220, 8%, 90%)",
  borderRadius: "4px",
  fontSize: "11px",
  boxShadow: "0 1px 4px rgba(0,0,0,0.06)",
};

const ClusterExplorer = () => {
  const [selectedCluster, setSelectedCluster] = useState<ClusterInfo | null>(null);
  const [drillCluster, setDrillCluster] = useState<string | null>(null);
  const [compareIds, setCompareIds] = useState<string[]>([]);
  const [sortKey, setSortKey] = useState<"anomaly_score" | "market_share" | "momentum">("anomaly_score");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");

  const [clusters, setClusters] = useState<ClusterInfo[]>([]);
  const [snapshots, setSnapshots] = useState<Snapshot[]>([]);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    Promise.all([
      fetchClusters().then((r) => setClusters(r.clusters)).catch(() => {}),
      fetchSnapshots().then(setSnapshots).catch(() => {}),
    ]).finally(() => setLoaded(true));
  }, []);

  const useReal = loaded && clusters.length > 0;
  const displayClusters = useReal ? clusters : (mockClusters as unknown as ClusterInfo[]);
  const displaySnapshots = useReal ? snapshots : mockClusterSnapshots;

  const sorted = useMemo(() => {
    return [...displayClusters].sort((a, b) => {
      const av = Math.abs(a[sortKey] ?? 0);
      const bv = Math.abs(b[sortKey] ?? 0);
      return sortDir === "desc" ? bv - av : av - bv;
    });
  }, [displayClusters, sortKey, sortDir]);

  const toggleSort = (key: typeof sortKey) => {
    if (sortKey === key) setSortDir((d) => d === "asc" ? "desc" : "asc");
    else { setSortKey(key); setSortDir("desc"); }
  };

  const toggleCompare = (id: string) => {
    setCompareIds((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : prev.length < 3 ? [...prev, id] : prev
    );
  };

  const drillData = useMemo(() => {
    if (!drillCluster) return [];
    return displaySnapshots
      .filter((s) => s.cluster_id === drillCluster)
      .sort((a, b) => a.time_window.localeCompare(b.time_window))
      .map((s) => ({ ...s, time_window: s.time_window.slice(0, 7) }));
  }, [drillCluster, displaySnapshots]);

  const compareData = useMemo(() => {
    if (!compareIds.length) return [];
    const tws = [...new Set(displaySnapshots.map((s) => s.time_window))].sort();
    return tws.map((tw) => {
      const row: Record<string, string | number> = { time_window: tw.slice(0, 7) };
      compareIds.forEach((id) => {
        const snap = displaySnapshots.find((s) => s.cluster_id === id && s.time_window === tw);
        if (snap) row[id] = snap.post_count;
      });
      return row;
    });
  }, [compareIds, displaySnapshots]);

  const lineColors = ["hsl(210, 30%, 48%)", "hsl(152, 28%, 40%)", "hsl(255, 18%, 50%)"];

  return (
    <DashboardLayout title="Cluster Explorer">
      <div className="space-y-8">
        {compareIds.length > 0 && (
          <div className="glass-card px-6 py-6">
            <div className="flex items-center justify-between mb-5">
              <div>
                <h3 className="text-section-label">Comparison</h3>
                <p className="mt-1 text-[11px] text-muted-foreground">
                  {compareIds.map((id) => displayClusters.find((c) => c.id === id)?.cluster_label).join(" vs ")}
                </p>
              </div>
              <button onClick={() => setCompareIds([])} className="text-[10px] text-muted-foreground hover:text-foreground transition-colors">
                Clear
              </button>
            </div>
            <ResponsiveContainer width="100%" height={200}>
              <AreaChart data={compareData} margin={{ top: 5, right: 10, bottom: 0, left: 0 }}>
                <XAxis dataKey="time_window" tick={{ fontSize: 10, fill: "hsl(220, 8%, 46%)" }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fontSize: 10, fill: "hsl(220, 8%, 46%)" }} axisLine={false} tickLine={false} width={36} />
                <Tooltip contentStyle={tooltipStyle} />
                {compareIds.map((id, i) => (
                  <Area key={id} type="monotone" dataKey={id}
                    name={displayClusters.find((c) => c.id === id)?.cluster_label || id}
                    stroke={lineColors[i]} strokeWidth={1.5} fill={lineColors[i]} fillOpacity={0.06} />
                ))}
              </AreaChart>
            </ResponsiveContainer>
          </div>
        )}

        {drillCluster && (
          <div className="glass-card px-6 py-6">
            <div className="flex items-center justify-between mb-5">
              <div>
                <h3 className="text-section-label">Timeline</h3>
                <p className="mt-1 text-[11px] text-muted-foreground">
                  {displayClusters.find((c) => c.id === drillCluster)?.cluster_label}
                </p>
              </div>
              <button onClick={() => setDrillCluster(null)} className="text-[10px] text-muted-foreground hover:text-foreground transition-colors">
                Close
              </button>
            </div>
            <ResponsiveContainer width="100%" height={180}>
              <AreaChart data={drillData} margin={{ top: 5, right: 10, bottom: 0, left: 0 }}>
                <XAxis dataKey="time_window" tick={{ fontSize: 10, fill: "hsl(220, 8%, 46%)" }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fontSize: 10, fill: "hsl(220, 8%, 46%)" }} axisLine={false} tickLine={false} width={36} />
                <Tooltip contentStyle={tooltipStyle} />
                <Area type="monotone" dataKey="post_count" stroke="hsl(220, 14%, 30%)" strokeWidth={1.5} fill="hsl(220, 14%, 30%)" fillOpacity={0.06} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        )}

        <div className="glass-card overflow-hidden">
          <div className="px-6 py-5 border-b border-border">
            <h3 className="text-section-label">All Clusters</h3>
            <p className="mt-1 text-[11px] text-muted-foreground">
              {displayClusters.length} clusters{useReal ? " (live)" : ""} · click row to drill down · checkbox to compare (max 3)
            </p>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-[11px]">
              <thead>
                <tr className="border-b border-border text-[9px] uppercase tracking-widest text-muted-foreground">
                  <th className="px-6 py-3 text-left w-8"></th>
                  <th className="px-3 py-3 text-left">Label</th>
                  <th className="px-3 py-3 text-left">Theme</th>
                  <th className="px-3 py-3 text-right">Posts</th>
                  <th className="px-3 py-3 text-right cursor-pointer hover:text-foreground" onClick={() => toggleSort("market_share")}>
                    Mkt Share {sortKey === "market_share" && (sortDir === "desc" ? "\u2193" : "\u2191")}
                  </th>
                  <th className="px-3 py-3 text-right">Vol \u0394%</th>
                  <th className="px-3 py-3 text-right">Volatility</th>
                  <th className="px-3 py-3 text-right cursor-pointer hover:text-foreground" onClick={() => toggleSort("momentum")}>
                    Momentum {sortKey === "momentum" && (sortDir === "desc" ? "\u2193" : "\u2191")}
                  </th>
                  <th className="px-3 py-3 text-center" title="Based on recent post volume change">
                    Volume trend
                  </th>
                  <th className="px-3 py-3 text-right cursor-pointer hover:text-foreground" onClick={() => toggleSort("anomaly_score")}>
                    Anomaly {sortKey === "anomaly_score" && (sortDir === "desc" ? "\u2193" : "\u2191")}
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/50">
                {sorted.map((c) => (
                  <tr key={c.id} className="hover:bg-accent/40 transition-colors">
                    <td className="px-6 py-2.5">
                      <input
                        type="checkbox"
                        checked={compareIds.includes(c.id)}
                        onChange={() => toggleCompare(c.id)}
                        className="h-3.5 w-3.5 accent-foreground"
                      />
                    </td>
                    <td
                      className="px-3 py-2.5 cursor-pointer text-foreground hover:underline"
                      onClick={() => setDrillCluster(c.id)}
                      title={c.theme_name ? `Topic (from content). Theme: ${c.theme_name}` : "Topic label from cluster content"}
                    >
                      {c.cluster_label}
                    </td>
                    <td className="px-3 py-2.5 text-muted-foreground text-[10px]">
                      {c.theme_name || "—"}
                    </td>
                    <td className="px-3 py-2.5 text-right font-mono-data text-muted-foreground">{c.size}</td>
                    <td className="px-3 py-2.5 text-right font-mono-data text-foreground">{c.market_share?.toFixed(2)}%</td>
                    <td className={cn("px-3 py-2.5 text-right font-mono-data",
                      (c.volume_pct_change ?? 0) > 0 ? "text-viz-emerald" : (c.volume_pct_change ?? 0) < 0 ? "text-viz-rose" : "text-muted-foreground"
                    )}>
                      {(c.volume_pct_change ?? 0) > 0 ? "+" : ""}{(c.volume_pct_change ?? 0).toFixed(1)}%
                    </td>
                    <td className="px-3 py-2.5 text-right font-mono-data text-muted-foreground">{(c.volume_volatility ?? 0).toFixed(3)}</td>
                    <td className="px-3 py-2.5 text-right font-mono-data text-muted-foreground">{(c.momentum ?? 0).toFixed(1)}</td>
                    <td className={cn("px-3 py-2.5 text-center font-medium", statusStyles[c.lifecycle ?? "Stable"])}>
                      {c.lifecycle}
                    </td>
                    <td className="px-3 py-2.5 text-right font-mono-data text-viz-rose">
                      {Math.abs(c.anomaly_score ?? 0).toFixed(2)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
      <ClusterDetailPanel cluster={selectedCluster} onClose={() => setSelectedCluster(null)} />
    </DashboardLayout>
  );
};

export default ClusterExplorer;
