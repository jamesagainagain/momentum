import { useState, useMemo, useEffect } from "react";
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer,
} from "recharts";
import { DashboardLayout } from "@/components/DashboardLayout";
import { KPICards } from "@/components/KPICards";
import { EventTimeline } from "@/components/EventTimeline";
import { ClusterDetailPanel } from "@/components/ClusterDetailPanel";
import {
  fetchClusters, fetchSnapshots, fetchWeeklySnapshots, fetchLifecycleDistribution, fetchTemporalEvents,
  type ClusterInfo, type Snapshot, type LifecycleItem, type TemporalEvent,
} from "@/lib/api";
import {
  mockClusters, mockClusterSnapshots, mockLifecycleDistribution, mockTemporalEvents,
} from "@/lib/mock-data";
import { cn } from "@/lib/utils";

type Cluster = ClusterInfo | typeof mockClusters[0];

const metricOptions = ["post_count", "market_share", "momentum", "volatility"] as const;
type Metric = typeof metricOptions[number];

const statusColors: Record<string, string> = {
  Emerging: "hsl(210, 30%, 48%)",
  Trending: "hsl(152, 28%, 40%)",
  Stable: "hsl(215, 25%, 50%)",
  Declining: "hsl(30, 35%, 46%)",
  Dormant: "hsl(220, 6%, 55%)",
};

const tooltipStyle = {
  background: "hsl(0, 0%, 100%)",
  border: "1px solid hsl(220, 8%, 90%)",
  borderRadius: "4px",
  fontSize: "11px",
  boxShadow: "0 1px 4px rgba(0,0,0,0.06)",
};

const Index = () => {
  const [selectedCluster, setSelectedCluster] = useState<Cluster | null>(null);
  const [selectedMetric, setSelectedMetric] = useState<Metric>("post_count");
  const [selectedClusterIds, setSelectedClusterIds] = useState<string[]>([]);
  const [granularity, setGranularity] = useState<"monthly" | "weekly">("monthly");
  const [weeklySnapshots, setWeeklySnapshots] = useState<Snapshot[]>([]);

  const [clusters, setClusters] = useState<ClusterInfo[]>([]);
  const [snapshots, setSnapshots] = useState<Snapshot[]>([]);
  const [lifecycleDist, setLifecycleDist] = useState<LifecycleItem[]>([]);
  const [events, setEvents] = useState<TemporalEvent[]>([]);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    Promise.all([
      fetchClusters().then((r) => setClusters(r.clusters)).catch(() => {}),
      fetchSnapshots().then(setSnapshots).catch(() => {}),
      fetchWeeklySnapshots().then(setWeeklySnapshots).catch(() => {}),
      fetchLifecycleDistribution().then(setLifecycleDist).catch(() => {}),
      fetchTemporalEvents().then(setEvents).catch(() => {}),
    ]).finally(() => setLoaded(true));
  }, []);

  const useRealData = loaded && clusters.length > 0;
  const displayClusters = useRealData ? clusters : mockClusters as unknown as ClusterInfo[];
  const displaySnapshots = useRealData ? snapshots : mockClusterSnapshots;
  const displayLifecycle = lifecycleDist.length > 0 ? lifecycleDist : mockLifecycleDistribution;
  const displayEvents = events.length > 0 ? events : mockTemporalEvents;

  const clusterOptions = useMemo(() => {
    const seen = new Set<string>();
    return displaySnapshots
      .filter((s) => { if (seen.has(s.cluster_id)) return false; seen.add(s.cluster_id); return true; })
      .map((s) => ({ id: s.cluster_id, label: s.cluster_label, theme_name: s.theme_name || "" }));
  }, [displaySnapshots]);

  const weeklyClusterIds = useMemo(
    () => new Set(weeklySnapshots.map((s) => s.cluster_id)),
    [weeklySnapshots]
  );
  const hasWeeklyData = weeklySnapshots.length > 0;

  const activeClusterOptions = useMemo(() => {
    if (granularity === "weekly" && hasWeeklyData) {
      return clusterOptions.filter((c) => weeklyClusterIds.has(c.id));
    }
    return clusterOptions;
  }, [clusterOptions, granularity, hasWeeklyData, weeklyClusterIds]);

  const activeSnapshots = granularity === "weekly" && hasWeeklyData
    ? weeklySnapshots
    : displaySnapshots;

  useEffect(() => {
    if (selectedClusterIds.length > 0) {
      const validIds = new Set(activeClusterOptions.map((c) => c.id));
      const stillValid = selectedClusterIds.filter((id) => validIds.has(id));
      if (stillValid.length !== selectedClusterIds.length) {
        setSelectedClusterIds(stillValid);
      }
    }
  }, [granularity]);

  const trendData = useMemo(() => {
    const ids = selectedClusterIds.length ? selectedClusterIds : activeClusterOptions.slice(0, 3).map((c) => c.id);
    const timeWindows = [...new Set(activeSnapshots.map((s) => s.time_window))].sort();
    return timeWindows.map((tw) => {
      const row: Record<string, string | number> = {
        time_window: granularity === "weekly" ? tw.slice(0, 10) : tw.slice(0, 7),
      };
      ids.forEach((id) => {
        const snap = activeSnapshots.find((s) => s.cluster_id === id && s.time_window === tw);
        if (snap) row[id] = snap[selectedMetric] ?? 0;
      });
      return row;
    });
  }, [selectedClusterIds, selectedMetric, activeClusterOptions, activeSnapshots, granularity]);

  const activeClusterIds = selectedClusterIds.length ? selectedClusterIds : activeClusterOptions.slice(0, 3).map((c) => c.id);
  const lineColors = ["hsl(210, 30%, 48%)", "hsl(152, 28%, 40%)", "hsl(255, 18%, 50%)"];

  const topAnomalous = [...displayClusters]
    .sort((a, b) => Math.abs(b.anomaly_score) - Math.abs(a.anomaly_score))
    .slice(0, 5);

  const toggleCluster = (id: string) => {
    setSelectedClusterIds((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : prev.length < 5 ? [...prev, id] : prev
    );
  };

  const totalClusters = displayClusters.length || 47;

  return (
    <DashboardLayout title="Trends">
      <div className="space-y-8">
        <KPICards />

        <div className="glass-card px-6 py-6">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h3 className="text-[11px] font-semibold text-foreground uppercase tracking-[0.1em]">Monthly Trends</h3>
              <p className="mt-0.5 text-[11px] text-muted-foreground">
                {granularity === "weekly"
                  ? `Weekly · last 12 months · ${weeklyClusterIds.size} dense cluster${weeklyClusterIds.size !== 1 ? "s" : ""}`
                  : `Monthly · full history`}
              </p>
            </div>
            <div className="flex items-center gap-2">
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
              {metricOptions.map((m) => (
                <button
                  key={m}
                  onClick={() => setSelectedMetric(m)}
                  className={cn(
                    "px-3 py-1.5 rounded text-[10px] font-medium transition-colors border",
                    selectedMetric === m
                      ? "bg-foreground text-background border-foreground"
                      : "text-muted-foreground border-border hover:text-foreground"
                  )}
                >
                  {m.replace("_", " ")}
                </button>
              ))}
            </div>
          </div>

          <div className="flex flex-wrap gap-1.5 mb-5">
            {activeClusterOptions.map((c) => (
              <button
                key={c.id}
                onClick={() => toggleCluster(c.id)}
                className={cn(
                  "px-2.5 py-1 rounded text-[10px] font-medium border transition-colors",
                  activeClusterIds.includes(c.id)
                    ? "bg-foreground text-background border-foreground"
                    : "text-muted-foreground border-border hover:text-foreground"
                )}
                title={c.theme_name ? `${c.label} · Theme: ${c.theme_name}` : c.label}
              >
                {c.label}
                {c.theme_name ? (
                  <span className="ml-1 opacity-70 text-[9px]">· {c.theme_name}</span>
                ) : null}
              </button>
            ))}
          </div>

          <ResponsiveContainer width="100%" height={240}>
            <AreaChart data={trendData} margin={{ top: 5, right: 10, bottom: 0, left: 0 }}>
              <XAxis dataKey="time_window" tick={{ fontSize: 10, fill: "hsl(220, 8%, 46%)" }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fontSize: 10, fill: "hsl(220, 8%, 46%)" }} axisLine={false} tickLine={false} width={40} />
              <Tooltip contentStyle={tooltipStyle} labelStyle={{ color: "hsl(220, 14%, 10%)" }} />
              {activeClusterIds.map((id, i) => {
                const opt = activeClusterOptions.find((c) => c.id === id);
                return (
                  <Area
                    key={id}
                    type="monotone"
                    dataKey={id}
                    name={opt ? (opt.theme_name ? `${opt.label} · ${opt.theme_name}` : opt.label) : id}
                    stroke={lineColors[i % lineColors.length]}
                    strokeWidth={1.5}
                    fill={lineColors[i % lineColors.length]}
                    fillOpacity={0.06}
                  />
                );
              })}
            </AreaChart>
          </ResponsiveContainer>
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
          <div className="glass-card px-6 py-6">
            <h3 className="text-[11px] font-semibold text-foreground uppercase tracking-[0.1em] mb-1">Lifecycle Distribution</h3>
            <p className="text-[10px] text-muted-foreground mb-6" title="Based on recent post volume change">
              Volume trend
            </p>
            <div className="space-y-4">
              {displayLifecycle.map((item) => (
                <div key={item.state} className="flex items-center gap-3" title="Based on recent post volume change">
                  <span className="w-16 text-[10px] text-muted-foreground">{item.state}</span>
                  <div className="flex-1 h-2.5 rounded-sm bg-accent overflow-hidden">
                    <div
                      className="h-full rounded-sm transition-all"
                      style={{
                        width: `${(item.count / totalClusters) * 100}%`,
                        backgroundColor: statusColors[item.state] || "hsl(220, 6%, 55%)",
                        opacity: 0.6,
                      }}
                    />
                  </div>
                  <span className="font-mono-data text-[11px] text-foreground w-6 text-right">{item.count}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="glass-card px-6 py-6">
            <h3 className="text-[11px] font-semibold text-foreground uppercase tracking-[0.1em] mb-6">Top Anomalous</h3>
            <div className="space-y-3">
              {topAnomalous.map((c) => (
                <button
                  key={c.id}
                  onClick={() => setSelectedCluster(c)}
                  className="w-full flex items-center justify-between rounded border border-border px-4 py-3 hover:bg-accent/50 transition-colors text-left"
                >
                  <div>
                    <span className="text-[11px] font-medium text-foreground">{c.cluster_label}</span>
                  </div>
                  <span className="font-mono-data text-[12px] text-viz-rose">{Math.abs(c.anomaly_score).toFixed(2)}</span>
                </button>
              ))}
            </div>
          </div>

          <div className="glass-card px-6 py-6">
            <h3 className="text-[11px] font-semibold text-foreground uppercase tracking-[0.1em] mb-6">Event Log</h3>
            <div className="space-y-4">
              {displayEvents.slice(0, 6).map((e, i) => (
                <div key={i} className="flex gap-3 items-start">
                  <div className={cn(
                    "mt-1.5 h-[6px] w-[6px] rounded-full shrink-0",
                    e.event_type === "spike" ? "bg-viz-rose" : "bg-viz-amber"
                  )} />
                  <div>
                    <p className="text-[11px] text-foreground leading-snug">{e.description || `${e.event_type} in ${e.cluster_label}`}</p>
                    <p className="text-[10px] text-muted-foreground mt-1">
                      {e.cluster_label} · {String(e.date).slice(0, 10)} · {Number(e.sigma).toFixed(1)}\u03C3
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        <EventTimeline />
      </div>

      <ClusterDetailPanel cluster={selectedCluster} onClose={() => setSelectedCluster(null)} />
    </DashboardLayout>
  );
};

export default Index;
