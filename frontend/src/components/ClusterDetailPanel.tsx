import { X } from "lucide-react";
import { cn } from "@/lib/utils";

interface ClusterDetailPanelProps {
  cluster: Record<string, unknown> | null;
  onClose: () => void;
}

export function ClusterDetailPanel({ cluster, onClose }: ClusterDetailPanelProps) {
  if (!cluster) return null;

  const label = String(cluster.cluster_label ?? "Cluster");
  const size = Number(cluster.size ?? 0);
  const marketShare = Number(cluster.market_share ?? 0);
  const anomaly = Number(cluster.anomaly_score ?? 0);
  const momentum = Number(cluster.momentum ?? 0);
  const volatility = Number(cluster.volume_volatility ?? 0);
  const volChange = Number(cluster.volume_pct_change ?? 0);
  const lifecycle = String(cluster.lifecycle ?? "Stable");

  const lifecycleColors: Record<string, string> = {
    Emerging: "text-status-emerging",
    Trending: "text-status-trending",
    Stable: "text-status-stable",
    Declining: "text-status-declining",
    Dormant: "text-status-dormant",
  };

  return (
    <div className="fixed inset-y-0 right-0 z-50 w-80 border-l border-border bg-card shadow-lg animate-slide-in-right">
      <div className="flex h-14 items-center justify-between border-b border-border px-6">
        <h3 className="text-section-label">Cluster Detail</h3>
        <button onClick={onClose} className="p-1.5 text-muted-foreground hover:text-foreground transition-colors">
          <X className="h-[14px] w-[14px]" strokeWidth={1.75} />
        </button>
      </div>

      <div className="px-6 py-7 space-y-7 overflow-y-auto h-[calc(100%-3.5rem)] scrollbar-thin">
        <div>
          <p className="text-[14px] font-semibold text-foreground leading-tight">{label}</p>
          <p className={cn("text-[11px] font-medium mt-2", lifecycleColors[lifecycle] || "text-muted-foreground")}>
            {lifecycle}
          </p>
        </div>

        <div className="h-px bg-border" />

        <div className="grid grid-cols-2 gap-3">
          {[
            { label: "Posts", value: size.toLocaleString() },
            { label: "Mkt Share", value: `${marketShare.toFixed(2)}%` },
            { label: "Anomaly", value: anomaly.toFixed(2) },
            { label: "Volatility", value: volatility.toFixed(3) },
          ].map((s) => (
            <div key={s.label} className="rounded border border-border px-3 py-3">
              <p className="text-[9px] text-muted-foreground uppercase tracking-widest">{s.label}</p>
              <p className="mt-1.5 text-[16px] font-semibold font-mono-data text-foreground leading-none">{s.value}</p>
            </div>
          ))}
        </div>

        <div>
          <p className="text-[9px] text-muted-foreground uppercase tracking-widest mb-4">Metrics</p>
          <div className="space-y-3.5">
            <div className="flex items-center justify-between">
              <span className="text-[11px] text-muted-foreground">Volume Change</span>
              <span className={cn("font-mono-data text-[11px]",
                volChange > 0 ? "text-viz-emerald" : volChange < 0 ? "text-viz-rose" : "text-muted-foreground"
              )}>
                {volChange > 0 ? "+" : ""}{volChange.toFixed(1)}%
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-[11px] text-muted-foreground">Momentum</span>
              <span className="font-mono-data text-[11px] text-foreground">
                {momentum > 0 ? "+" : ""}{momentum.toFixed(1)}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-[11px] text-muted-foreground">Volatility</span>
              <span className="font-mono-data text-[11px] text-foreground">{volatility.toFixed(3)}</span>
            </div>
          </div>
        </div>

        <div className="h-px bg-border" />

        <div>
          <div className="flex items-center justify-between mb-3">
            <p className="text-[9px] text-muted-foreground uppercase tracking-widest">Momentum</p>
            <span className="font-mono-data text-[11px] text-muted-foreground">
              {momentum > 0 ? "+" : ""}{momentum.toFixed(1)}
            </span>
          </div>
          <div className="h-1 w-full rounded-full bg-accent overflow-hidden">
            <div
              className="h-full rounded-full bg-foreground/20 transition-all"
              style={{ width: `${Math.min(Math.max((momentum + 50) / 2, 5), 100)}%` }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
