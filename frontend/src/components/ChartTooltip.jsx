// Recharts tooltip styled with the app's tokens. Values use text ink; the
// coloured square next to each row carries series identity.
export default function ChartTooltip({ active, payload, label, valueFormatter = (v) => v, labelFormatter = (l) => l }) {
  if (!active || !payload || payload.length === 0) return null;
  return (
    <div className="rounded-lg border border-line bg-surface px-3 py-2 text-xs shadow-md">
      <div className="mb-1 font-semibold text-ink">{labelFormatter(label)}</div>
      <ul className="space-y-0.5">
        {payload.map((p) => (
          <li key={p.dataKey} className="flex items-center gap-2 text-ink-2">
            <span className="inline-block h-2 w-2 rounded-sm" style={{ background: p.color }} aria-hidden="true" />
            <span className="max-w-56 truncate">{p.name}</span>
            <span className="tabular ml-auto pl-3 font-semibold text-ink">{valueFormatter(p.value, p)}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

export const AXIS_PROPS = {
  stroke: "var(--axis)",
  tick: { fill: "var(--muted)", fontSize: 11 },
  tickLine: false,
};

export const SERIES = ["var(--series-1)", "var(--series-2)", "var(--series-3)"];
