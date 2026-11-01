// Small presentational components shared by several pages.
import { AlertTriangle, Loader2, Star } from "lucide-react";
import { highlightParts } from "../lib/format.js";

export function Stars({ value, size = 14 }) {
  const filled = Math.round(value);
  return (
    <span className="inline-flex items-center gap-0.5" aria-label={`${value.toFixed(2)} out of 5 stars`}>
      {[1, 2, 3, 4, 5].map((i) => (
        <Star
          key={i}
          size={size}
          aria-hidden="true"
          className="text-star"
          fill={i <= filled ? "currentColor" : "none"}
          strokeWidth={1.5}
        />
      ))}
    </span>
  );
}

export function Loading({ label = "Loading…" }) {
  return (
    <div className="flex items-center gap-2 py-10 text-ink-2" role="status">
      <Loader2 className="animate-spin" size={18} aria-hidden="true" /> {label}
    </div>
  );
}

export function ErrorBox({ error }) {
  if (!error) return null;
  return (
    <div role="alert" className="flex items-start gap-2 rounded-xl bg-warn-soft px-4 py-3 text-sm text-warn">
      <AlertTriangle size={18} className="mt-0.5 shrink-0" aria-hidden="true" />
      <span>{error.message || String(error)}</span>
    </div>
  );
}

export function Highlighted({ text, term }) {
  return (
    <>
      {highlightParts(text, term).map((p, i) =>
        p.match ? (
          <mark key={i} className="term">
            {p.text}
          </mark>
        ) : (
          <span key={i}>{p.text}</span>
        ),
      )}
    </>
  );
}

const POLARITY_STYLE = {
  positive: { label: "Positive", className: "text-pos", symbol: "+" },
  negative: { label: "Negative", className: "text-neg", symbol: "−" },
  neutral: { label: "Neutral", className: "text-muted", symbol: "•" },
};

export function ReviewSnippet({ snippet }) {
  const style = POLARITY_STYLE[snippet.polarity] ?? null;
  return (
    <figure className="rounded-lg bg-surface-2 px-3 py-2 text-sm">
      <blockquote className="text-ink">
        “<Highlighted text={snippet.text} term={snippet.term} />”
      </blockquote>
      <figcaption className="mt-1 flex flex-wrap gap-x-3 text-xs text-muted">
        {style && (
          <span className={`font-semibold ${style.className}`}>
            {style.symbol} {style.label}
          </span>
        )}
        <span>{snippet.rating}★ review</span>
        <span className="tabular">id {snippet.review_id}</span>
      </figcaption>
    </figure>
  );
}

export function StatTile({ label, value, hint }) {
  return (
    <div className="card px-4 py-3">
      <div className="text-xs font-medium uppercase tracking-wide text-muted">{label}</div>
      <div className="mt-1 text-2xl font-semibold">{value}</div>
      {hint && <div className="mt-0.5 text-xs text-ink-2">{hint}</div>}
    </div>
  );
}
