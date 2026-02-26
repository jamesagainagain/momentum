import { cn } from "@/lib/utils";

interface TopBarProps {
  /** Section title (e.g. "Predict", "Trends", "Topics"). Shown in the bar; omit to hide the bar. */
  title?: string;
}

export function TopBar({ title }: TopBarProps) {
  if (!title) return null;

  return (
    <header
      className={cn(
        "flex h-14 shrink-0 items-center border-b border-border glass-panel"
      )}
    >
      <div className="mx-auto w-full max-w-3xl px-page-x text-section-label">{title}</div>
    </header>
  );
}
