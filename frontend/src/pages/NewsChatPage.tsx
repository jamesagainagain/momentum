import { useState, useRef, useEffect, useMemo } from "react";
import {
  Send, ArrowUpRight, ArrowDownRight, Minus, Clock, TrendingUp,
  BarChart3, AlertTriangle, Loader2, ChevronDown, ChevronUp,
} from "lucide-react";
import {
  AreaChart, Area, BarChart, Bar, XAxis, YAxis, Tooltip,
  ResponsiveContainer, Cell, ReferenceLine,
} from "recharts";
import { DashboardLayout } from "@/components/DashboardLayout";
import { LiquidGlassCanvas } from "@/components/LiquidGlassCanvas";
import { MarkdownRenderer } from "@/components/MarkdownRenderer";
import { predictEvent, chatAnalysis, type PredictionResponse } from "@/lib/api";
import { LIQUID_GLASS_BAR_PRESET } from "@/lib/liquid-glass/preset";
import { cn } from "@/lib/utils";

const directionConfig = {
  up: { icon: ArrowUpRight, label: "Upward", className: "text-direction-up" },
  down: { icon: ArrowDownRight, label: "Downward", className: "text-direction-down" },
  flat: { icon: Minus, label: "Flat", className: "text-direction-flat" },
};

const riskColors: Record<string, string> = {
  low: "text-viz-emerald",
  medium: "text-viz-amber",
  high: "text-viz-rose",
};

const riskBg: Record<string, string> = {
  low: "bg-viz-emerald/10 border-viz-emerald/20",
  medium: "bg-viz-amber/10 border-viz-amber/20",
  high: "bg-viz-rose/10 border-viz-rose/20",
};

const tooltipStyle = {
  background: "hsl(0, 0%, 100%)",
  border: "1px solid hsl(var(--border))",
  borderRadius: "8px",
  fontSize: "11px",
  boxShadow: "0 4px 12px rgba(0,0,0,0.08)",
};

const palette = [
  "hsl(210, 30%, 48%)", "hsl(152, 28%, 40%)", "hsl(255, 18%, 50%)",
  "hsl(30, 35%, 46%)", "hsl(340, 30%, 48%)",
];

interface ChatEntry {
  id: string;
  event_text: string;
  prediction: PredictionResponse;
  analysis: string | null;
  timestamp: string;
}

/* ── Charts ────────────────────────────────────────── */

function ClassProbChart({ probs }: { probs: Record<string, number> }) {
  const data = [
    { label: "Down", value: (probs.down || 0) * 100, fill: "hsl(0, 45%, 55%)" },
    { label: "Flat", value: (probs.flat || 0) * 100, fill: "hsl(220, 8%, 46%)" },
    { label: "Up", value: (probs.up || 0) * 100, fill: "hsl(152, 28%, 40%)" },
  ];
  return (
    <div className="mt-4">
      <p className="text-[10px] text-muted-foreground uppercase tracking-widest mb-3">Class probabilities</p>
      <ResponsiveContainer width="100%" height={80}>
        <BarChart data={data} layout="vertical" margin={{ top: 0, right: 10, bottom: 0, left: 32 }}>
          <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 9 }} axisLine={false} tickLine={false} tickFormatter={(v: number) => `${v}%`} />
          <YAxis type="category" dataKey="label" tick={{ fontSize: 10 }} axisLine={false} tickLine={false} width={30} />
          <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={14}>
            {data.map((d, i) => (
              <Cell key={i} fill={d.fill} fillOpacity={0.7} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function TrendChart({ trendData }: { trendData: PredictionResponse["trend_data"] }) {
  if (!trendData?.length) return null;
  const windows = [...new Set(trendData.flatMap((t) => t.data.map((d) => d.time_window)))].sort();
  const rows = windows.map((tw) => {
    const row: Record<string, string | number> = { time_window: tw.slice(0, 7) };
    trendData.forEach((t) => {
      const pt = t.data.find((d) => d.time_window === tw);
      if (pt) row[`c${t.cluster_id}`] = pt.post_count;
    });
    return row;
  });
  return (
    <div className="mt-5">
      <p className="text-[10px] text-muted-foreground uppercase tracking-widest mb-3 flex items-center gap-1.5">
        <TrendingUp className="h-[11px] w-[11px]" strokeWidth={1.75} />
        Historical trends
      </p>
      <ResponsiveContainer width="100%" height={180}>
        <AreaChart data={rows} margin={{ top: 5, right: 10, bottom: 0, left: 0 }}>
          <XAxis dataKey="time_window" tick={{ fontSize: 9 }} axisLine={false} tickLine={false} interval="preserveStartEnd" />
          <YAxis tick={{ fontSize: 9 }} axisLine={false} tickLine={false} width={36} />
          <Tooltip contentStyle={tooltipStyle} />
          {trendData.map((t, i) => (
            <Area key={t.cluster_id} type="monotone" dataKey={`c${t.cluster_id}`} name={t.cluster_label}
              stroke={palette[i % palette.length]} strokeWidth={1.5} fill={palette[i % palette.length]} fillOpacity={0.08} />
          ))}
        </AreaChart>
      </ResponsiveContainer>
      <div className="flex flex-wrap gap-3 mt-2">
        {trendData.map((t, i) => (
          <span key={t.cluster_id} className="flex items-center gap-1.5 text-[9px] text-muted-foreground">
            <span className="h-[6px] w-[6px] rounded-full" style={{ backgroundColor: palette[i % palette.length] }} />
            {t.cluster_label}
          </span>
        ))}
      </div>
    </div>
  );
}

function MomentumChart({ trendData }: { trendData: PredictionResponse["trend_data"] }) {
  if (!trendData?.length) return null;
  const rows = trendData.map((t) => {
    const last = t.data[t.data.length - 1];
    return {
      label: t.cluster_label.length > 16 ? t.cluster_label.slice(0, 14) + "\u2026" : t.cluster_label,
      volatility: last?.volatility ?? 0,
      momentum: last?.momentum ?? 0,
    };
  });
  return (
    <div className="mt-5">
      <p className="text-[10px] text-muted-foreground uppercase tracking-widest mb-3 flex items-center gap-1.5">
        <BarChart3 className="h-[11px] w-[11px]" strokeWidth={1.75} />
        Momentum & volatility
      </p>
      <ResponsiveContainer width="100%" height={120}>
        <BarChart data={rows} margin={{ top: 5, right: 10, bottom: 0, left: 0 }}>
          <XAxis dataKey="label" tick={{ fontSize: 8 }} axisLine={false} tickLine={false} />
          <YAxis tick={{ fontSize: 9 }} axisLine={false} tickLine={false} width={30} />
          <Tooltip contentStyle={tooltipStyle} />
          <ReferenceLine y={0} stroke="hsl(var(--border))" strokeDasharray="3 3" />
          <Bar dataKey="momentum" name="Momentum" fill="hsl(210, 30%, 48%)" fillOpacity={0.6} radius={[4, 4, 0, 0]} barSize={20} />
          <Bar dataKey="volatility" name="Volatility" fill="hsl(30, 35%, 46%)" fillOpacity={0.5} radius={[4, 4, 0, 0]} barSize={20} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function AnalysisPanel({ analysis }: { analysis: string }) {
  const [open, setOpen] = useState(true);
  return (
    <div className="mt-5 rounded-xl border border-border bg-accent/20 overflow-hidden">
      <button onClick={() => setOpen(!open)} className="w-full flex items-center justify-between px-5 py-3.5 text-left hover:bg-accent/30 transition-colors">
        <span className="text-[10px] text-muted-foreground uppercase tracking-widest font-medium">AI analysis</span>
        {open ? <ChevronUp className="h-3.5 w-3.5 text-muted-foreground" /> : <ChevronDown className="h-3.5 w-3.5 text-muted-foreground" />}
      </button>
      {open && (
        <div className="px-5 pb-5">
          <MarkdownRenderer text={analysis} />
        </div>
      )}
    </div>
  );
}

function ThemeTable({ themes }: { themes: PredictionResponse["activated_themes"] }) {
  if (!themes?.length) return null;
  return (
    <div className="mt-5">
      <p className="text-[10px] text-muted-foreground uppercase tracking-widest mb-3">Theme details</p>
      <div className="overflow-x-auto rounded-lg border border-border">
        <table className="w-full text-[10px]">
          <thead>
            <tr className="border-b border-border bg-accent/30 text-[8px] uppercase tracking-widest text-muted-foreground">
              <th className="px-3 py-2.5 text-left">Theme</th>
              <th className="px-3 py-2.5 text-right">Score</th>
              <th className="px-3 py-2.5 text-right">Weight</th>
              <th className="px-3 py-2.5 text-center">Dir</th>
              <th className="px-3 py-2.5 text-right">Vol</th>
              <th className="px-3 py-2.5 text-right">Mkt %</th>
              <th className="px-3 py-2.5 text-right">Mom</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border/50">
            {themes.map((t) => {
              const d = directionConfig[t.direction_label];
              const I = d.icon;
              return (
                <tr key={t.theme_id}>
                  <td className="px-3 py-2.5 text-foreground">{t.theme_name}</td>
                  <td className="px-3 py-2.5 text-right font-mono-data text-foreground">{t.score.toFixed(2)}</td>
                  <td className="px-3 py-2.5 text-right font-mono-data text-muted-foreground">{(t.weight * 100).toFixed(1)}%</td>
                  <td className="px-3 py-2.5 text-center">
                    <span className={cn("inline-flex items-center gap-1", d.className)}>
                      <I className="h-[10px] w-[10px]" strokeWidth={1.75} />
                      {d.label}
                    </span>
                  </td>
                  <td className="px-3 py-2.5 text-right font-mono-data text-muted-foreground">{t.avg_volatility.toFixed(2)}</td>
                  <td className="px-3 py-2.5 text-right font-mono-data text-muted-foreground">{(t.avg_market_share * 100).toFixed(2)}%</td>
                  <td className="px-3 py-2.5 text-right font-mono-data text-muted-foreground">{t.recent_momentum.toFixed(1)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* ── Prediction Card ───────────────────────────────── */

function PredictionCard({ entry, index }: { entry: ChatEntry; index: number }) {
  const { prediction, analysis } = entry;
  const dir = directionConfig[prediction.predicted_direction];
  const DirIcon = dir.icon;
  const [showDetails, setShowDetails] = useState(false);

  return (
    <article
      className="rounded-2xl border border-border bg-card shadow-sm overflow-hidden animate-fade-in-up"
      style={{ animationDelay: `${index * 60}ms`, animationFillMode: "backwards" }}
    >
      {/* Lead: direction + confidence */}
      <div className="px-8 pt-8 pb-6">
        <div className="flex flex-wrap items-end justify-between gap-6">
          <div className="flex items-center gap-4">
            <div className={cn("flex items-center justify-center h-14 w-14 rounded-xl border-2", dir.className, "border-current")}>
              <DirIcon className="h-7 w-7" strokeWidth={1.75} />
            </div>
            <div>
              <p className={cn("font-display text-2xl font-normal", dir.className)}>{dir.label}</p>
              <p className="text-[11px] text-muted-foreground mt-0.5">
                {prediction.method} model{prediction.model_name ? ` · ${prediction.model_name}` : ""}
              </p>
            </div>
          </div>
          <div className="text-right">
            <p className="font-mono-data text-3xl font-semibold text-foreground leading-none">
              {(prediction.confidence * 100).toFixed(0)}%
            </p>
            <p className="text-[11px] text-muted-foreground mt-1.5">confidence</p>
          </div>
        </div>

        {/* Summary row */}
        <div className="flex flex-wrap items-center gap-4 mt-6 pt-6 border-t border-border">
          <div className="flex items-center gap-2">
            <span className="text-[10px] text-muted-foreground uppercase tracking-widest">Volatility</span>
            <span className="font-mono-data text-sm font-medium text-foreground">{prediction.predicted_volatility.toFixed(2)}</span>
          </div>
          <div className={cn("flex items-center gap-2 px-3 py-1.5 rounded-lg border", riskBg[prediction.risk_band])}>
            <span className="text-[10px] text-muted-foreground uppercase tracking-widest">Risk</span>
            <span className={cn("text-sm font-semibold capitalize", riskColors[prediction.risk_band])}>{prediction.risk_band}</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-[10px] text-muted-foreground uppercase tracking-widest">Themes</span>
            <span className="font-mono-data text-sm font-medium text-foreground">{prediction.activated_theme_count}</span>
          </div>
          <p className="text-[10px] text-muted-foreground ml-auto flex items-center gap-1.5">
            <Clock className="h-[10px] w-[10px]" strokeWidth={1.75} />
            {new Date(entry.timestamp).toLocaleTimeString()}
          </p>
        </div>
      </div>

      {/* Collapsible details */}
      <div className="border-t border-border">
        <button
          onClick={() => setShowDetails(!showDetails)}
          className="w-full flex items-center justify-between px-8 py-4 text-left text-[11px] font-medium text-muted-foreground hover:text-foreground hover:bg-accent/30 transition-colors"
        >
          <span>{showDetails ? "Hide" : "Show"} details & charts</span>
          {showDetails ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
        </button>
        {showDetails && (
          <div className="px-8 pb-8 pt-2 bg-accent/10">
            {prediction.class_probabilities && Object.keys(prediction.class_probabilities).length > 0 && (
              <ClassProbChart probs={prediction.class_probabilities} />
            )}
            <ThemeTable themes={prediction.activated_themes} />
            <TrendChart trendData={prediction.trend_data} />
            <MomentumChart trendData={prediction.trend_data} />
          </div>
        )}
      </div>

      {/* AI analysis or short summary */}
      {analysis && (
        <div className="border-t border-border bg-card">
          <AnalysisPanel analysis={analysis} />
        </div>
      )}

      {!analysis && prediction.activated_themes?.length > 0 && (
        <div className="px-8 pb-8 pt-2 border-t border-border">
          <p className="text-[12px] leading-relaxed text-muted-foreground">
            Predicted{" "}
            <span className={cn("font-medium text-foreground", dir.className)}>{dir.label.toLowerCase()}</span>
            {" "}from {prediction.activated_themes.slice(0, 2).map((t) => t.theme_name).join(" and ")}.
            Risk is{" "}
            <span className={cn("font-medium", riskColors[prediction.risk_band])}>{prediction.risk_band}</span>
            {" "}with volatility {prediction.predicted_volatility.toFixed(2)}.
          </p>
        </div>
      )}

      {prediction.activated_theme_count === 0 && (
        <div className="px-8 pb-8 pt-4 border-t border-border flex items-start gap-3">
          <AlertTriangle className="h-5 w-5 text-viz-amber shrink-0 mt-0.5" strokeWidth={1.75} />
          <p className="text-[12px] leading-relaxed text-muted-foreground">
            No themes matched above threshold. The event may be outside current patterns — try a more specific headline or keyword.
          </p>
        </div>
      )}
    </article>
  );
}

/* ── Page ──────────────────────────────────────────── */

const EXAMPLES = [
  "Google launches new Gemini robotics model",
  "Fed announces surprise rate hike",
  "AlphaFold 3 solves protein interactions",
];

const NewsChatPage = () => {
  const [input, setInput] = useState("");
  const [history, setHistory] = useState<ChatEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [stage, setStage] = useState<"predicting" | "analyzing" | "">("");
  const [error, setError] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const barContainerRef = useRef<HTMLDivElement>(null);
  const [barSize, setBarSize] = useState({ width: 0, height: 0 });

  // Subtle gradient so liquid glass refraction/dispersion/Fresnel are visible (from liquid-glass-studio)
  const barGradientDataUrl = useMemo(() => {
    const light = "hsl(40, 33%, 99%)";
    const mid = "hsl(40, 28%, 96%)";
    const dark = "hsl(40, 25%, 94%)";
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64"><defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="${light}"/><stop offset="50%" stop-color="${mid}"/><stop offset="100%" stop-color="${dark}"/></linearGradient></defs><rect width="64" height="64" fill="url(#g)"/></svg>`;
    return `data:image/svg+xml,${encodeURIComponent(svg)}`;
  }, []);

  useEffect(() => {
    const el = barContainerRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const { width, height } = entries[0]?.contentRect ?? { width: 0, height: 0 };
      setBarSize({ width: Math.round(width), height: Math.round(height) });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [history, loading]);

  const handleSubmit = async () => {
    if (!input.trim() || loading) return;
    setLoading(true);
    setError(null);
    const text = input.trim();
    setInput("");

    try {
      setStage("predicting");
      const prediction = await predictEvent(text);

      const entryId = crypto.randomUUID();
      const entry: ChatEntry = {
        id: entryId,
        event_text: text,
        prediction: { ...prediction, id: entryId, timestamp: new Date().toISOString() },
        analysis: null,
        timestamp: new Date().toISOString(),
      };
      setHistory((prev) => [...prev, entry]);

      setStage("analyzing");
      try {
        const prev = history.map((h) => h.prediction);
        const chat = await chatAnalysis(text, prediction, prev);
        setHistory((h) =>
          h.map((e) => (e.id === entryId ? { ...e, analysis: chat.analysis } : e))
        );
      } catch {
        setHistory((h) =>
          h.map((e) =>
            e.id === entryId
              ? { ...e, analysis: "AI analysis unavailable — the prediction above is still valid." }
              : e
          )
        );
      }
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Prediction failed. Is the backend running on port 8000?"
      );
    } finally {
      setLoading(false);
      setStage("");
    }
  };

  return (
    <DashboardLayout title="Predict">
      <div className="flex flex-col min-h-[calc(100vh-3.5rem)] max-w-3xl mx-auto">
        <div ref={scrollRef} className="flex-1 overflow-y-auto scrollbar-thin">
          {history.length === 0 && !loading && (
            <div className="pt-20 pb-28">
              <div className="mx-auto max-w-2xl rounded-2xl glass-panel p-12 shadow-lg">
                <h1 className="font-display text-[2rem] leading-tight font-semibold text-foreground tracking-tight">
                  What happens next?
                </h1>
                <p className="mt-5 text-[1.0625rem] text-muted-foreground leading-relaxed max-w-xl">
                  Enter a news event or headline. We predict how it will move social and topic trends — and explain why.
                </p>
                <div className="mt-10 flex items-center gap-4 rounded-xl border border-border bg-background/60 px-5 py-1 shadow-sm focus-within:border-foreground/25 focus-within:ring-2 focus-within:ring-foreground/10 transition-all min-h-[3.5rem]">
                  <input
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
                    placeholder="Paste a headline or describe a news event…"
                    className="flex-1 bg-transparent text-[15px] text-foreground placeholder:text-muted-foreground focus:outline-none min-h-[3rem] py-3"
                  />
                  <button
                    onClick={handleSubmit}
                    disabled={!input.trim() || loading}
                    className={cn(
                      "flex items-center justify-center h-11 w-11 rounded-xl transition-all shrink-0",
                      input.trim()
                        ? "bg-foreground text-background hover:opacity-90"
                        : "bg-muted text-muted-foreground cursor-not-allowed"
                    )}
                  >
                    <Send className="h-5 w-5" strokeWidth={1.75} />
                  </button>
                </div>
                <div className="mt-8 flex flex-wrap gap-3">
                  {EXAMPLES.map((ex) => (
                    <button
                      key={ex}
                      onClick={() => setInput(ex)}
                      className="rounded-xl border border-border/80 bg-background/50 px-5 py-3 text-[13px] text-muted-foreground shadow-sm backdrop-blur-sm hover:border-foreground/20 hover:text-foreground hover:bg-background/70 transition-all min-h-[2.75rem]"
                    >
                      {ex}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}

          {history.map((entry, i) => (
            <div key={entry.id} className="space-y-6 pb-12">
              <div className="flex justify-end">
                <div className="rounded-2xl bg-foreground/5 border border-border px-6 py-4 max-w-xl">
                  <p className="text-[15px] text-foreground leading-snug">{entry.event_text}</p>
                </div>
              </div>
              <PredictionCard entry={entry} index={i} />
            </div>
          ))}

          {loading && (
            <div className="rounded-2xl border border-border bg-card/80 backdrop-blur px-8 py-6 flex items-center gap-5">
              <Loader2 className="h-6 w-6 text-muted-foreground animate-spin shrink-0" strokeWidth={1.75} />
              <div>
                <p className="text-[14px] font-medium text-foreground">
                  {stage === "predicting" ? "Predicting impact\u2026" : "Writing analysis\u2026"}
                </p>
                <p className="text-[12px] text-muted-foreground mt-1">
                  {stage === "predicting" ? "Running model on your event" : "Generating trend insights"}
                </p>
              </div>
            </div>
          )}

          {error && (
            <div className="rounded-2xl border border-viz-rose/30 bg-viz-rose/5 px-6 py-5">
              <p className="text-[13px] text-viz-rose font-medium">Something went wrong</p>
              <p className="text-[12px] text-muted-foreground mt-2">{error}</p>
            </div>
          )}
        </div>

        {/* Sticky input — no bar: transparent strip, only the glass pill is visible. Fits main content width. */}
        {(history.length > 0 || loading) && (
          <div className="sticky bottom-0 left-0 right-0 pt-6 pb-8">
            <div className="w-full px-page-x">
              <div
                ref={barContainerRef}
                className="relative h-[4.5rem] w-full"
              >
                <LiquidGlassCanvas
                  width={barSize.width || 640}
                  height={barSize.height || 72}
                  fillShape={false}
                  preset={LIQUID_GLASS_BAR_PRESET}
                  shapeWidthRatio={0.98}
                  shapeHeightRatio={0.92}
                  className="absolute inset-0 h-full w-full pointer-events-none"
                  interactive={false}
                  backgroundTextureUrl={barGradientDataUrl}
                  fallbackTransparent
                />
                <div className="absolute inset-0 flex items-center gap-3 px-4 py-3 bg-transparent pointer-events-auto predict-sticky-input">
                  <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
                    placeholder="Paste a headline or describe a news event…"
                    className="flex-1 w-full min-w-0 bg-transparent text-[15px] text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-0 focus:border-0 border-0 shadow-none min-h-[2.75rem] appearance-none outline-none overflow-hidden text-ellipsis whitespace-nowrap"
                    style={{ background: 'transparent', border: 'none', boxShadow: 'none' }}
                  />
                  <button
                    onClick={handleSubmit}
                    disabled={!input.trim() || loading}
                    className={cn(
                      "flex shrink-0 items-center justify-center h-10 w-10 rounded-xl transition-all",
                      input.trim() && !loading
                        ? "bg-foreground text-background hover:opacity-90"
                        : "bg-muted text-muted-foreground cursor-not-allowed"
                    )}
                  >
                    {loading
                      ? <Loader2 className="h-5 w-5 animate-spin" strokeWidth={1.75} />
                      : <Send className="h-5 w-5" strokeWidth={1.75} />}
                  </button>
                </div>
              </div>
              <p className="mt-2 text-[11px] text-muted-foreground text-center">
                {history.length} prediction{history.length !== 1 ? "s" : ""} this session
              </p>
            </div>
          </div>
        )}
      </div>
    </DashboardLayout>
  );
};

export default NewsChatPage;
