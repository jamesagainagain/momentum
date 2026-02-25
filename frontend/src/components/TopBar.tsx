import { useState, useEffect } from "react";
import { Calendar, ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";

const defaultTimeWindows = ["Daily", "Weekly", "Monthly"] as const;

interface TopBarProps {
  title?: string;
  /** Controlled: current time window (e.g. "Monthly" | "Weekly"). When set, TopBar acts in controlled mode. */
  timeWindow?: string;
  /** Called when user selects a time window. When provided with timeWindow, drives Index granularity. */
  onTimeWindowChange?: (window: string) => void;
  /** Which buttons to show. Omit Daily when there is no daily data. */
  timeWindowOptions?: readonly string[];
}

export function TopBar({
  title = "Executive Dashboard",
  timeWindow: controlledWindow,
  onTimeWindowChange,
  timeWindowOptions = defaultTimeWindows,
}: TopBarProps) {
  const [localWindow, setLocalWindow] = useState<string>("Monthly");
  const isControlled = controlledWindow !== undefined;
  const activeWindow = isControlled ? controlledWindow : localWindow;

  useEffect(() => {
    if (isControlled && controlledWindow) setLocalWindow(controlledWindow);
  }, [isControlled, controlledWindow]);

  const handleChange = (w: string) => {
    if (!isControlled) setLocalWindow(w);
    onTimeWindowChange?.(w);
  };

  return (
    <header className="flex h-14 items-center justify-between border-b border-border px-8 glass-panel">
      <h1 className="text-[11px] font-medium text-muted-foreground uppercase tracking-[0.12em]">{title}</h1>

      <div className="flex items-center gap-3">
        <div className="flex items-center rounded border border-border overflow-hidden">
          {timeWindowOptions.map((w, i) => (
            <button
              key={w}
              onClick={() => handleChange(w)}
              className={cn(
                "px-3.5 py-1.5 text-[11px] font-medium transition-all",
                i > 0 && "border-l border-border",
                activeWindow === w
                  ? "bg-foreground text-background"
                  : "text-muted-foreground hover:text-foreground"
              )}
            >
              {w}
            </button>
          ))}
        </div>

        <button className="flex items-center gap-2 rounded border border-border px-3 py-1.5 text-[11px] text-muted-foreground hover:text-foreground transition-colors">
          <Calendar className="h-[13px] w-[13px]" strokeWidth={1.75} />
          <span>Jul 2024 – Jan 2025</span>
          <ChevronDown className="h-[11px] w-[11px]" strokeWidth={1.75} />
        </button>
      </div>
    </header>
  );
}
