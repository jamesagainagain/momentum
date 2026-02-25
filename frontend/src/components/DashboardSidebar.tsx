import { useState } from "react";
import { NavLink } from "react-router-dom";
import {
  LayoutDashboard,
  Search,
  MessageSquare,
  FlaskConical,
  ChevronLeft,
  ChevronRight,
  Sparkles,
} from "lucide-react";
import { cn } from "@/lib/utils";

const navItems = [
  { title: "Predict", path: "/", icon: MessageSquare },
  { title: "Trends", path: "/dashboard", icon: LayoutDashboard },
  { title: "Topics", path: "/clusters", icon: Search },
  { title: "Scenarios", path: "/scenarios", icon: FlaskConical },
];

export function DashboardSidebar() {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <aside
      className={cn(
        "flex flex-col border-r border-border glass-panel transition-all duration-200",
        collapsed ? "w-[4.5rem]" : "w-[14rem]"
      )}
    >
      {/* Logo / brand — 8px grid */}
      <div className="flex h-16 items-center gap-3 border-b border-border px-5">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-foreground shrink-0">
          <Sparkles className="h-4 w-4 text-background" strokeWidth={2} />
        </div>
        {!collapsed && (
          <span className="font-display text-[15px] font-semibold tracking-tight text-foreground">
            Momentum
          </span>
        )}
      </div>

      {/* Nav — comfortable touch targets, 12px vertical rhythm */}
      <nav className="flex-1 space-y-0.5 px-3 pt-8">
        {navItems.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            end={item.path === "/"}
            className={({ isActive }) =>
              cn(
                "flex items-center gap-3 rounded-lg px-3 py-3 text-[13px] font-medium transition-colors min-h-[2.75rem]",
                isActive
                  ? "bg-foreground text-background shadow-sm"
                  : "text-muted-foreground hover:bg-accent hover:text-foreground"
              )
            }
          >
            <item.icon className="h-4 w-4 shrink-0" strokeWidth={1.75} />
            {!collapsed && <span>{item.title}</span>}
          </NavLink>
        ))}
      </nav>

      {/* Collapse */}
      <button
        onClick={() => setCollapsed(!collapsed)}
        className="flex h-14 items-center justify-center border-t border-border text-muted-foreground hover:text-foreground transition-colors"
      >
        {collapsed ? <ChevronRight className="h-4 w-4" strokeWidth={1.75} /> : <ChevronLeft className="h-4 w-4" strokeWidth={1.75} />}
      </button>
    </aside>
  );
}
