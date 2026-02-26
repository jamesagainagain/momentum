import { cn } from "@/lib/utils";
import { mockLifecycleData } from "@/lib/mock-data";

const statusStyles: Record<string, string> = {
  Emerging: "text-status-emerging",
  Trending: "text-status-trending",
  Stable: "text-status-stable",
  Declining: "text-status-declining",
  Dormant: "text-status-dormant",
};

export function LifecycleTracker() {
  return (
    <div className="glass-card px-6 py-6">
      <h3 className="text-section-label">Lifecycle</h3>
      <p className="mt-1 mb-6 text-[11px] text-muted-foreground">Cluster status ranking</p>

      <div className="grid grid-cols-[1fr_56px_56px_72px_64px] gap-2 px-2 pb-3 text-[9px] uppercase tracking-widest text-muted-foreground border-b border-border">
        <span>Topic</span>
        <span className="text-right">Posts</span>
        <span className="text-right">Δ</span>
        <span className="text-center">Trend</span>
        <span className="text-right">State</span>
      </div>

      <div className="divide-y divide-border/50">
        {mockLifecycleData.map((item) => {
          const growth = parseFloat(item.growthRate);
          return (
            <div
              key={item.id}
              className="grid grid-cols-[1fr_56px_56px_72px_64px] gap-2 items-center px-2 py-3 text-sm hover:bg-accent/40 transition-colors cursor-pointer"
            >
              <span className="text-[11px] text-foreground truncate">
                {item.cluster_label}
              </span>
              <span className="text-right font-mono-data text-[11px] text-muted-foreground">
                {item.size}
              </span>
              <span className={cn(
                "text-right text-[11px] font-mono-data",
                growth > 0 ? "text-viz-emerald" : growth < 0 ? "text-viz-rose" : "text-muted-foreground"
              )}>
                {growth > 0 ? "+" : ""}{growth}%
              </span>
              <div className="flex items-center justify-center gap-[2px]">
                {item.sparkline.map((v, i) => (
                  <div
                    key={i}
                    className="w-[3px] rounded-sm bg-foreground/12"
                    style={{ height: `${(v / 100) * 14 + 2}px` }}
                  />
                ))}
              </div>
              <span className={cn("text-right text-[10px] font-medium", statusStyles[item.lifecycle])}>
                {item.lifecycle}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
