import { Sparkles, ChevronRight } from "lucide-react";

export function AIInsightPanel() {
  const insights = [
    {
      cluster: "Climate Policy",
      summary:
        "340% engagement surge on Jan 13th driven by carbon taxation proposals. Sentiment polarized: LinkedIn +18% positive, X -12% negative. Viral thread reshared 4,200+ times.",
      confidence: 0.87,
    },
    {
      cluster: "AI Regulation",
      summary:
        "Moved from Emerging to Trending over 7 days. LinkedIn thought-leadership driving growth. Enterprise audience interest detected. Predicted peak in 3-5 days.",
      confidence: 0.74,
    },
    {
      cluster: "Crypto Markets",
      summary:
        "Declining engagement correlates with market stabilization. Engagement-per-post dropped 23%. Monitor for re-emergence tied to regulatory news.",
      confidence: 0.69,
    },
  ];

  return (
    <div className="glass-card px-6 py-6">
      <div className="mb-6 flex items-center gap-2.5">
        <Sparkles className="h-[14px] w-[14px] text-muted-foreground" strokeWidth={1.75} />
        <h3 className="text-section-label">AI Analysis</h3>
      </div>

      <div className="divide-y divide-border">
        {insights.map((insight, i) => (
          <div
            key={i}
            className="group py-5 first:pt-0 last:pb-0 hover:bg-accent/30 -mx-6 px-6 transition-colors cursor-pointer"
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-[11px] font-medium text-foreground">{insight.cluster}</span>
              <span className="font-mono-data text-[10px] text-muted-foreground">
                {(insight.confidence * 100).toFixed(0)}%
              </span>
            </div>
            <p className="text-[11px] leading-[1.7] text-muted-foreground">
              {insight.summary}
            </p>
            <div className="mt-2.5 flex items-center text-[10px] text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity">
              Details <ChevronRight className="ml-0.5 h-[10px] w-[10px]" strokeWidth={1.75} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
