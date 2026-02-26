import { useState } from "react";
import {
  Send, Download, Save, X, ArrowUpRight, ArrowDownRight, Minus, Loader2,
} from "lucide-react";
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer,
} from "recharts";
import { DashboardLayout } from "@/components/DashboardLayout";
import { predictEvent, type PredictionResponse } from "@/lib/api";
import { cn } from "@/lib/utils";

const directionConfig = {
  up: { icon: ArrowUpRight, label: "Up", className: "text-direction-up" },
  down: { icon: ArrowDownRight, label: "Down", className: "text-direction-down" },
  flat: { icon: Minus, label: "Flat", className: "text-direction-flat" },
};

const riskColors: Record<string, string> = {
  low: "text-viz-emerald",
  medium: "text-viz-amber",
  high: "text-viz-rose",
};

const palette = [
  "hsl(210, 30%, 48%)", "hsl(152, 28%, 40%)", "hsl(255, 18%, 50%)",
  "hsl(30, 35%, 46%)", "hsl(340, 30%, 48%)",
];

const tooltipStyle = {
  background: "hsl(0, 0%, 100%)",
  border: "1px solid hsl(220, 8%, 90%)",
  borderRadius: "4px",
  fontSize: "11px",
  boxShadow: "0 1px 4px rgba(0,0,0,0.06)",
};

interface Scenario extends PredictionResponse {
  id: string;
  timestamp: string;
}

function MiniTrendChart({ trendData }: { trendData: PredictionResponse["trend_data"] }) {
  if (!trendData?.length) return null;
  const first = trendData[0];
  const rows = first.data.slice(-12).map((d) => ({
    tw: d.time_window.slice(0, 7),
    posts: d.post_count,
  }));
  return (
    <div className="mt-3">
      <ResponsiveContainer width="100%" height={60}>
        <AreaChart data={rows} margin={{ top: 2, right: 4, bottom: 0, left: 0 }}>
          <Area type="monotone" dataKey="posts" stroke={palette[0]} strokeWidth={1} fill={palette[0]} fillOpacity={0.08} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

const ScenarioPanel = () => {
  const [input, setInput] = useState("");
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [loading, setLoading] = useState(false);
  const [saved, setSaved] = useState<Scenario[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [compareIds, setCompareIds] = useState<string[]>([]);

  const handleSubmit = async () => {
    if (!input.trim() || loading) return;
    setLoading(true);
    setError(null);
    const text = input.trim();
    setInput("");
    try {
      const prediction = await predictEvent(text);
      const scenario: Scenario = {
        ...prediction,
        id: crypto.randomUUID(),
        timestamp: new Date().toISOString(),
      };
      setScenarios((prev) => [...prev, scenario]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Prediction failed");
    } finally {
      setLoading(false);
    }
  };

  const removeScenario = (id: string) => {
    setScenarios((prev) => prev.filter((s) => s.id !== id));
    setCompareIds((prev) => prev.filter((x) => x !== id));
  };
  const saveScenario = (s: Scenario) => {
    if (!saved.find((x) => x.id === s.id)) setSaved((prev) => [...prev, s]);
  };
  const toggleCompare = (id: string) => {
    setCompareIds((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : prev.length < 5 ? [...prev, id] : prev
    );
  };

  const exportResults = (format: "csv" | "json") => {
    const data = scenarios.length ? scenarios : saved;
    if (!data.length) return;
    let content: string, mime: string, ext: string;
    if (format === "json") {
      content = JSON.stringify(data, null, 2); mime = "application/json"; ext = "json";
    } else {
      const header = "event_text,predicted_direction,confidence,predicted_volatility,risk_band,method,activated_themes\n";
      const rows = data.map((d) =>
        `"${d.event_text}",${d.predicted_direction},${d.confidence.toFixed(4)},${d.predicted_volatility.toFixed(4)},${d.risk_band},${d.method},${d.activated_theme_count}`
      ).join("\n");
      content = header + rows; mime = "text/csv"; ext = "csv";
    }
    const blob = new Blob([content], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a"); a.href = url; a.download = `scenarios.${ext}`; a.click();
    URL.revokeObjectURL(url);
  };

  const comparedScenarios = scenarios.filter((s) => compareIds.includes(s.id));
  const comparisonData = comparedScenarios.length > 1 ? (() => {
    const allWindows = [...new Set(
      comparedScenarios.flatMap((s) =>
        (s.trend_data ?? []).flatMap((t) => t.data.map((d) => d.time_window))
      )
    )].sort();
    return allWindows.map((tw) => {
      const row: Record<string, string | number> = { time_window: tw.slice(0, 7) };
      comparedScenarios.forEach((s) => {
        const total = (s.trend_data ?? []).reduce((sum, t) => {
          const pt = t.data.find((d) => d.time_window === tw);
          return sum + (pt?.post_count ?? 0);
        }, 0);
        row[s.id] = total;
      });
      return row;
    });
  })() : [];

  return (
    <DashboardLayout title="Forecast Scenarios">
      <div className="space-y-8 max-w-5xl">
        {/* Input */}
        <div className="glass-card px-6 py-6">
          <h3 className="text-section-label mb-5">New Scenario</h3>
          <div className="flex items-center gap-3">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
              placeholder="Enter a hypothetical news event\u2026"
              className="flex-1 rounded border border-border bg-background px-4 py-3 text-[12px] text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-foreground"
            />
            <button
              onClick={handleSubmit}
              disabled={!input.trim() || loading}
              className={cn(
                "flex items-center gap-2 rounded px-5 py-3 text-[11px] font-medium transition-colors",
                input.trim() && !loading ? "bg-foreground text-background" : "bg-accent text-muted-foreground"
              )}
            >
              {loading
                ? <Loader2 className="h-[13px] w-[13px] animate-spin" strokeWidth={1.75} />
                : <Send className="h-[13px] w-[13px]" strokeWidth={1.75} />}
              Predict
            </button>
          </div>
          {error && <p className="mt-3 text-[10px] text-viz-rose">{error}</p>}
        </div>

        {/* Export + actions */}
        {scenarios.length > 0 && (
          <div className="flex items-center gap-3">
            <button onClick={() => exportResults("csv")} className="flex items-center gap-2 rounded border border-border px-3.5 py-2 text-[10px] text-muted-foreground hover:text-foreground transition-colors">
              <Download className="h-[12px] w-[12px]" strokeWidth={1.75} /> Export CSV
            </button>
            <button onClick={() => exportResults("json")} className="flex items-center gap-2 rounded border border-border px-3.5 py-2 text-[10px] text-muted-foreground hover:text-foreground transition-colors">
              <Download className="h-[12px] w-[12px]" strokeWidth={1.75} /> Export JSON
            </button>
            {compareIds.length > 0 && (
              <button onClick={() => setCompareIds([])} className="text-[10px] text-muted-foreground hover:text-foreground transition-colors ml-auto">
                Clear comparison
              </button>
            )}
          </div>
        )}

        {/* Comparison chart */}
        {comparisonData.length > 0 && (
          <div className="glass-card px-6 py-6">
            <h3 className="text-section-label mb-3">
              Scenario Comparison
            </h3>
            <p className="text-[10px] text-muted-foreground mb-4">
              Total activated cluster volume over time
            </p>
            <ResponsiveContainer width="100%" height={200}>
              <AreaChart data={comparisonData} margin={{ top: 5, right: 10, bottom: 0, left: 0 }}>
                <XAxis dataKey="time_window" tick={{ fontSize: 9, fill: "hsl(220, 8%, 46%)" }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fontSize: 9, fill: "hsl(220, 8%, 46%)" }} axisLine={false} tickLine={false} width={36} />
                <Tooltip contentStyle={tooltipStyle} />
                {comparedScenarios.map((s, i) => (
                  <Area key={s.id} type="monotone" dataKey={s.id}
                    name={s.event_text.length > 30 ? s.event_text.slice(0, 28) + "\u2026" : s.event_text}
                    stroke={palette[i % palette.length]} strokeWidth={1.5}
                    fill={palette[i % palette.length]} fillOpacity={0.06}
                  />
                ))}
              </AreaChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Scenario cards */}
        {scenarios.length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-5">
            {scenarios.map((s) => {
              const dir = directionConfig[s.predicted_direction];
              const DirIcon = dir.icon;
              const isCompared = compareIds.includes(s.id);
              return (
                <div key={s.id} className={cn(
                  "glass-card px-5 py-5 relative animate-fade-in-up",
                  isCompared && "ring-1 ring-foreground/20"
                )}>
                  <div className="absolute top-4 right-4 flex items-center gap-1.5">
                    <button onClick={() => toggleCompare(s.id)}
                      className={cn("px-2 py-1 rounded text-[8px] font-medium border transition-colors",
                        isCompared ? "bg-foreground text-background border-foreground" : "text-muted-foreground border-border hover:text-foreground"
                      )}
                    >
                      {isCompared ? "Comparing" : "Compare"}
                    </button>
                    <button onClick={() => saveScenario(s)} className="p-1.5 text-muted-foreground hover:text-foreground transition-colors" title="Save">
                      <Save className="h-[13px] w-[13px]" strokeWidth={1.75} />
                    </button>
                    <button onClick={() => removeScenario(s.id)} className="p-1.5 text-muted-foreground hover:text-foreground transition-colors">
                      <X className="h-[13px] w-[13px]" strokeWidth={1.75} />
                    </button>
                  </div>

                  <p className="text-[11px] text-foreground font-medium mb-4 pr-28 line-clamp-2">{s.event_text}</p>

                  <div className="flex items-center gap-2.5 mb-4">
                    <DirIcon className={cn("h-[15px] w-[15px]", dir.className)} strokeWidth={1.75} />
                    <span className={cn("text-[13px] font-semibold", dir.className)}>{dir.label}</span>
                    <span className="font-mono-data text-[12px] text-foreground ml-auto">{(s.confidence * 100).toFixed(0)}%</span>
                  </div>

                  <div className="grid grid-cols-2 gap-2.5 text-center">
                    <div className="rounded border border-border px-3 py-2.5">
                      <p className="text-[8px] text-muted-foreground uppercase tracking-wide">Volatility</p>
                      <p className="font-mono-data text-[12px] text-foreground mt-1">{s.predicted_volatility.toFixed(2)}</p>
                    </div>
                    <div className="rounded border border-border px-3 py-2.5">
                      <p className="text-[8px] text-muted-foreground uppercase tracking-wide">Risk</p>
                      <p className={cn("text-[12px] font-medium capitalize mt-1", riskColors[s.risk_band])}>{s.risk_band}</p>
                    </div>
                  </div>

                  <div className="mt-4 space-y-1.5">
                    {s.activated_themes.slice(0, 3).map((t) => (
                      <div key={t.theme_id} className="flex items-center justify-between text-[10px]">
                        <span className="text-muted-foreground">{t.theme_name}</span>
                        <span className="font-mono-data text-foreground">{t.score.toFixed(2)}</span>
                      </div>
                    ))}
                  </div>

                  <MiniTrendChart trendData={s.trend_data} />

                  <p className="mt-3 text-[9px] text-muted-foreground">
                    {s.method} model · {new Date(s.timestamp).toLocaleTimeString()}
                  </p>
                </div>
              );
            })}
          </div>
        )}

        {/* Saved */}
        {saved.length > 0 && (
          <div className="glass-card px-6 py-6">
            <h3 className="text-section-label mb-5">
              Saved Scenarios ({saved.length})
            </h3>
            <div className="divide-y divide-border">
              {saved.map((s) => {
                const dir = directionConfig[s.predicted_direction];
                return (
                  <div key={s.id} className="flex items-center justify-between py-3">
                    <div className="flex items-center gap-3">
                      <dir.icon className={cn("h-[13px] w-[13px]", dir.className)} strokeWidth={1.75} />
                      <span className="text-[11px] text-foreground">{s.event_text}</span>
                    </div>
                    <div className="flex items-center gap-5 font-mono-data text-[10px] text-muted-foreground">
                      <span>{(s.confidence * 100).toFixed(0)}%</span>
                      <span className={cn("capitalize", riskColors[s.risk_band])}>{s.risk_band}</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {scenarios.length === 0 && saved.length === 0 && (
          <div className="glass-card px-6 py-20 text-center">
            <p className="text-[13px] text-muted-foreground">No scenarios yet.</p>
            <p className="text-[11px] text-muted-foreground/60 mt-2">Enter hypothetical events above to compare real predictions side-by-side.</p>
          </div>
        )}
      </div>
    </DashboardLayout>
  );
};

export default ScenarioPanel;
