import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceDot,
} from "recharts";
import { mockEventTimeline } from "@/lib/mock-data";

export function EventTimeline() {
  const anomalies = mockEventTimeline.filter((d) => d.anomaly);

  return (
    <div className="glass-card px-6 py-6">
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h3 className="text-section-label">Event Timeline</h3>
          <p className="mt-1 text-[11px] text-muted-foreground">Engagement · {anomalies.length} anomalies ({">"}2σ)</p>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={220}>
        <AreaChart data={mockEventTimeline} margin={{ top: 5, right: 10, bottom: 0, left: 0 }}>
          <defs>
            <linearGradient id="engGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="hsl(220, 14%, 10%)" stopOpacity={0.06} />
              <stop offset="100%" stopColor="hsl(220, 14%, 10%)" stopOpacity={0} />
            </linearGradient>
          </defs>
          <XAxis
            dataKey="date"
            tick={{ fontSize: 10, fill: "hsl(220, 8%, 46%)" }}
            tickFormatter={(v) => v.slice(5)}
            axisLine={false}
            tickLine={false}
          />
          <YAxis
            tick={{ fontSize: 10, fill: "hsl(220, 8%, 46%)" }}
            tickFormatter={(v) => (v / 1000).toFixed(0) + "k"}
            axisLine={false}
            tickLine={false}
            width={36}
          />
          <Tooltip
            contentStyle={{
              background: "hsl(0, 0%, 100%)",
              border: "1px solid hsl(220, 8%, 90%)",
              borderRadius: "4px",
              fontSize: "11px",
              boxShadow: "0 1px 4px rgba(0,0,0,0.06)",
            }}
            labelStyle={{ color: "hsl(220, 14%, 10%)" }}
            itemStyle={{ color: "hsl(220, 8%, 46%)" }}
            formatter={(value: number) => [value.toLocaleString(), "Engagement"]}
          />
          <Area type="monotone" dataKey="engagement" stroke="hsl(220, 14%, 30%)" strokeWidth={1} fill="url(#engGrad)" />
          {anomalies.map((a) => (
            <ReferenceDot
              key={a.date}
              x={a.date}
              y={a.engagement}
              r={3.5}
              fill="hsl(0, 40%, 50%)"
              stroke="hsl(0, 0%, 100%)"
              strokeWidth={2}
            />
          ))}
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
