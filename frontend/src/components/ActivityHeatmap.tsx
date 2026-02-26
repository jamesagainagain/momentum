import { mockHeatmapData } from "@/lib/mock-data";

export function ActivityHeatmap() {
  const maxVal = Math.max(...mockHeatmapData.flatMap((r) => r.data.map((d) => d.value)));

  const getOpacity = (value: number) => {
    const intensity = value / maxVal;
    if (intensity < 0.2) return 0.04;
    if (intensity < 0.4) return 0.1;
    if (intensity < 0.6) return 0.2;
    if (intensity < 0.8) return 0.35;
    return 0.55;
  };

  return (
    <div className="glass-card px-6 py-6">
      <h3 className="text-section-label">Activity Heatmap</h3>
      <p className="mt-1 mb-6 text-[11px] text-muted-foreground">Cluster intensity over time</p>

      <div className="space-y-0.5">
        <div className="grid grid-cols-[90px_repeat(7,1fr)] gap-0.5 mb-1">
          <span />
          {["M", "T", "W", "T", "F", "S", "S"].map((d, i) => (
            <span key={i} className="text-center text-[9px] text-muted-foreground">{d}</span>
          ))}
        </div>

        {mockHeatmapData.map((row) => (
          <div key={row.cluster} className="grid grid-cols-[90px_repeat(7,1fr)] gap-0.5 items-center">
            <span className="text-[10px] text-muted-foreground truncate pr-1">{row.cluster}</span>
            {row.data.map((cell, i) => (
              <div
                key={i}
                className="h-4 rounded-sm bg-foreground transition-opacity hover:opacity-70 cursor-pointer"
                style={{ opacity: getOpacity(cell.value) }}
                title={`${row.cluster} · ${cell.day}: ${cell.value}`}
              />
            ))}
          </div>
        ))}
      </div>

      <div className="mt-5 flex items-center gap-1 justify-end">
        <span className="text-[9px] text-muted-foreground mr-0.5">Low</span>
        {[4, 10, 20, 35, 55].map((o) => (
          <div key={o} className="h-[6px] w-4 rounded-sm bg-foreground" style={{ opacity: o / 100 }} />
        ))}
        <span className="text-[9px] text-muted-foreground ml-0.5">High</span>
      </div>
    </div>
  );
}
