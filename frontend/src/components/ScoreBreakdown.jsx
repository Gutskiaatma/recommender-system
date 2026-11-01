import { score100 } from "../lib/format.js";

// Shows how the final score is built: weight × component for each signal.
// Segments are the three categorical slots, always in the same order and
// always labelled, so identity never relies on colour alone.
const PARTS = [
  { key: "relevance", weight: "alpha", label: "Relevance", color: "var(--series-1)" },
  { key: "aspect", weight: "beta", label: "Aspect sentiment", color: "var(--series-2)" },
  { key: "quality", weight: "gamma", label: "Rating", color: "var(--series-3)" },
];

export default function ScoreBreakdown({ scores, weights }) {
  if (!weights) {
    return (
      <div className="text-xs text-ink-2">
        Similarity <span className="tabular font-semibold text-ink">{scores.relevance_raw.toFixed(3)}</span>{" "}
        (single-signal method)
      </div>
    );
  }
  const segments = PARTS.filter((p) => scores[p.key] !== null && weights[p.weight] > 0).map((p) => ({
    ...p,
    value: weights[p.weight] * scores[p.key],
    component: scores[p.key],
    w: weights[p.weight],
  }));
  return (
    <div className="space-y-1.5">
      <div className="flex h-2.5 gap-[2px] overflow-hidden rounded-full bg-surface-2" aria-hidden="true">
        {segments.map((s) => (
          <div key={s.key} style={{ width: `${s.value * 100}%`, background: s.color }} />
        ))}
      </div>
      <ul className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-ink-2">
        {segments.map((s) => (
          <li key={s.key} className="flex items-center gap-1.5">
            <span className="inline-block h-2 w-2 rounded-sm" style={{ background: s.color }} aria-hidden="true" />
            {s.label} {score100(s.component)} × {s.w.toFixed(2)}
          </li>
        ))}
      </ul>
    </div>
  );
}
