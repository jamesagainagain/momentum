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
        "flex flex-col border-r border-border bg-sidebar transition-all duration-200",
        collapsed ? "w-[56px]" : "w-[200px]"
      )}
    >
      {/* Logo / brand */}
      <div className="flex h-14 items-center gap-2.5 border-b border-border px-4">
        <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-foreground shrink-0">
          <Sparkles className="h-[14px] w-[14px] text-background" strokeWidth={2} />
        </div>
        {!collapsed && (
          <span className="font-display text-[15px] font-semibold tracking-tight text-foreground">
            Momentum
          </span>
        )}
      </div>

      {/* Nav — Predict first and visually primary */}
      <nav className="flex-1 space-y-0.5 px-2.5 pt-6">
        {navItems.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            end={item.path === "/"}
            className={({ isActive }) =>
              cn(
                "flex items-center gap-3 rounded-lg px-3 py-2.5 text-[12px] font-medium transition-colors",
                isActive
                  ? "bg-foreground text-background shadow-sm"
                  : "text-muted-foreground hover:bg-accent hover:text-foreground"
              )
            }
          >
            <item.icon className="h-[15px] w-[15px] shrink-0" strokeWidth={1.75} />
            {!collapsed && <span>{item.title}</span>}
          </NavLink>
        ))}
      </nav>

      {/* Collapse */}
      <button
        onClick={() => setCollapsed(!collapsed)}
        className="flex h-12 items-center justify-center border-t border-border text-muted-foreground hover:text-foreground transition-colors"
      >
        {collapsed ? <ChevronRight className="h-[14px] w-[14px]" strokeWidth={1.75} /> : <ChevronLeft className="h-[14px] w-[14px]" strokeWidth={1.75} />}
      </button>
    </aside>
  );
}
