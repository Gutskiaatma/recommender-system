import { score100 } from "../lib/format.js";

// One aspect row: a 0-100 score bar with a tick at the catalogue average,
// plus the positive / neutral / negative mention split as a diverging strip.
export default function AspectBar({ aspect }) {
  const { label, score, catalog_average: avg, n_mentions: n, n_positive, n_negative, n_neutral, sufficient_evidence } = aspect;
  const total = Math.max(n, 1);
  return (
    <div className="space-y-1">
      <div className="flex items-baseline justify-between gap-2 text-sm">
        <span className="font-medium">{label}</span>
        <span className="tabular text-xs text-ink-2">
          {sufficient_evidence ? (
            <>
              <span className="text-sm font-semibold text-ink">{score100(score)}</span>/100 · {n} mentions
            </>
          ) : (
            <span className="text-warn">Not enough evidence ({n})</span>
          )}
        </span>
      </div>
      <div
        className="relative h-2 rounded-full bg-surface-2"
        role="img"
        aria-label={`${label}: score ${score100(score)} of 100, catalogue average ${score100(avg)}`}
      >
        <div
          className="h-2 rounded-full"
          style={{ width: `${score * 100}%`, background: sufficient_evidence ? "var(--series-1)" : "var(--axis)" }}
        />
        <div
          className="absolute -top-1 h-4 w-0.5 rounded bg-ink-2"
          style={{ left: `calc(${avg * 100}% - 1px)` }}
          title={`Catalogue average ${score100(avg)}`}
        />
      </div>
      {n > 0 && (
        <div className="flex items-center justify-end gap-2 text-[11px] text-muted">
          <span className="tabular">
            {n_positive} positive · {n_neutral} neutral · {n_negative} negative
          </span>
          <div className="flex h-1.5 w-20 gap-[2px] overflow-hidden rounded-full" aria-hidden="true">
            <div style={{ width: `${(n_positive / total) * 100}%`, background: "var(--pos)" }} />
            <div style={{ width: `${(n_neutral / total) * 100}%`, background: "var(--neu)" }} />
            <div style={{ width: `${(n_negative / total) * 100}%`, background: "var(--neg)" }} />
          </div>
        </div>
      )}
    </div>
  );
}
