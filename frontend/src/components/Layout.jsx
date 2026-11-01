import { NavLink, Outlet, useNavigate } from "react-router-dom";
import { BarChart3, GitCompare, Search, X } from "lucide-react";
import { useCompare } from "../lib/compareContext.jsx";
import { shortTitle } from "../lib/format.js";

function NavItem({ to, icon: Icon, children }) {
  return (
    <NavLink
      to={to}
      className={({ isActive }) =>
        `flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm font-medium transition ${
          isActive ? "bg-accent-soft text-accent" : "text-ink-2 hover:bg-surface-2"
        }`
      }
    >
      <Icon size={16} aria-hidden="true" />
      <span className="hidden sm:inline">{children}</span>
    </NavLink>
  );
}

function CompareTray() {
  const { items, toggle, clear, query } = useCompare();
  const navigate = useNavigate();
  if (items.length === 0) return null;
  const go = () => {
    const ids = items.map((p) => p.product_id).join(",");
    navigate(`/compare?ids=${ids}${query ? `&q=${encodeURIComponent(query)}` : ""}`);
  };
  return (
    <div className="fixed inset-x-0 bottom-0 z-20 border-t border-line bg-surface/95 backdrop-blur">
      <div className="mx-auto flex max-w-6xl flex-wrap items-center gap-2 px-4 py-3">
        <span className="text-sm font-medium text-ink-2">Compare ({items.length}/3):</span>
        {items.map((p) => (
          <span key={p.product_id} className="flex items-center gap-1 rounded-full bg-surface-2 px-3 py-1 text-xs">
            {shortTitle(p.title, 34)}
            <button onClick={() => toggle(p)} aria-label={`Remove ${p.title}`} className="text-muted hover:text-ink">
              <X size={12} />
            </button>
          </span>
        ))}
        <div className="ml-auto flex gap-2">
          <button onClick={clear} className="rounded-lg px-3 py-1.5 text-sm text-ink-2 hover:bg-surface-2">
            Clear
          </button>
          <button
            onClick={go}
            disabled={items.length < 2}
            className="rounded-lg bg-accent px-4 py-1.5 text-sm font-semibold text-accent-ink disabled:opacity-40"
          >
            Compare
          </button>
        </div>
      </div>
    </div>
  );
}

export default function Layout() {
  return (
    <div className="min-h-screen pb-24">
      <header className="sticky top-0 z-10 border-b border-line bg-page/90 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center gap-4 px-4 py-3">
          <NavLink to="/" className="flex items-center gap-2 font-semibold tracking-tight">
            <span className="grid h-8 w-8 place-items-center rounded-lg bg-accent text-accent-ink">
              <Search size={16} aria-hidden="true" />
            </span>
            ReviewLens
          </NavLink>
          <nav className="ml-auto flex gap-1">
            <NavItem to="/search" icon={Search}>Search</NavItem>
            <NavItem to="/compare" icon={GitCompare}>Compare</NavItem>
            <NavItem to="/insights" icon={BarChart3}>Insights</NavItem>
          </nav>
        </div>
      </header>
      <main className="mx-auto max-w-6xl px-4 py-6">
        <Outlet />
      </main>
      <CompareTray />
    </div>
  );
}
