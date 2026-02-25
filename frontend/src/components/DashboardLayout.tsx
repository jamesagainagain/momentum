import { ReactNode } from "react";
import { DashboardSidebar } from "./DashboardSidebar";
import { TopBar } from "./TopBar";

interface DashboardLayoutProps {
  children: ReactNode;
  title?: string;
  /** Current time window for TopBar (e.g. "Monthly" | "Weekly"). When set, TopBar syncs with page granularity. */
  timeWindow?: string;
  /** Called when user changes time window in TopBar. */
  onTimeWindowChange?: (window: string) => void;
  /** Time window buttons to show in TopBar (e.g. ["Weekly", "Monthly"] to hide Daily). */
  timeWindowOptions?: readonly string[];
}

export function DashboardLayout({
  children,
  title,
  timeWindow,
  onTimeWindowChange,
  timeWindowOptions,
}: DashboardLayoutProps) {
  return (
    <div className="flex h-screen w-full overflow-hidden bg-background">
      <DashboardSidebar />
      <div className="flex flex-1 flex-col overflow-hidden">
        <TopBar
          title={title}
          timeWindow={timeWindow}
          onTimeWindowChange={onTimeWindowChange}
          timeWindowOptions={timeWindowOptions}
        />
        <main className="flex-1 overflow-y-auto px-8 py-8 scrollbar-thin">
          {children}
        </main>
      </div>
    </div>
  );
}
