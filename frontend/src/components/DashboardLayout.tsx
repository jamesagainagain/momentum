import { ReactNode } from "react";
import { DashboardSidebar } from "./DashboardSidebar";
import { TopBar } from "./TopBar";

interface DashboardLayoutProps {
  children: ReactNode;
  /** Section title shown in the top bar (e.g. "Predict", "Trends"). Omit to hide the bar. */
  title?: string;
}

export function DashboardLayout({
  children,
  title,
}: DashboardLayoutProps) {
  return (
    <div className="flex h-screen w-full overflow-hidden bg-background">
      <DashboardSidebar />
      <div className="flex flex-1 flex-col overflow-hidden">
        <TopBar title={title} />
        <main className="flex-1 overflow-y-auto scrollbar-thin px-page-x py-page-y">
          {children}
        </main>
      </div>
    </div>
  );
}
