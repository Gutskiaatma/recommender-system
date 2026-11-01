import { useState } from "react";
import { Link } from "react-router-dom";
import { Check, ChevronDown, ChevronUp, Plus } from "lucide-react";
import AspectBar from "./AspectBar.jsx";
import ScoreBreakdown from "./ScoreBreakdown.jsx";
import { ReviewSnippet, Stars } from "./Bits.jsx";
import { useCompare } from "../lib/compareContext.jsx";
import { compactNumber, formatPrice, score100 } from "../lib/format.js";

export default function ProductCard({ product, weights, defaultOpen = false }) {
  const [open, setOpen] = useState(defaultOpen);
  const { toggle, has, full } = useCompare();
  const selected = has(product.product_id);
  const { scores } = product;

  return (
    <article className="card p-4 sm:p-5">
      <div className="flex gap-4">
        <div className="flex w-14 shrink-0 flex-col items-center">
          <span className="text-xs font-medium text-muted">#{product.rank}</span>
          <span className="tabular text-2xl font-semibold" title="Overall match score (0–100)">
            {weights ? score100(scores.final) : scores.relevance_raw.toFixed(2)}
          </span>
          <span className="text-[10px] uppercase tracking-wide text-muted">{weights ? "match" : "sim."}</span>
        </div>

        <div className="min-w-0 flex-1 space-y-3">
          <div className="flex flex-wrap items-start justify-between gap-2">
            <div className="min-w-0">
              <Link to={`/product/${product.product_id}`} className="font-semibold leading-snug hover:text-accent">
                {product.title}
              </Link>
              <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-sm text-ink-2">
                <span className="font-medium text-ink">{product.brand}</span>
                <span>{formatPrice(product.price)}</span>
                <span className="flex items-center gap-1">
                  <Stars value={product.mean_rating} />
                  <span className="tabular">
                    {product.mean_rating.toFixed(1)} ({compactNumber(product.n_reviews)})
                  </span>
                </span>
                {product.n_variants > 1 && <span className="text-xs text-muted">{product.n_variants} variants merged</span>}
              </div>
            </div>
            <button
              onClick={() => toggle(product)}
              disabled={!selected && full}
              className={`flex items-center gap-1 rounded-lg border px-2.5 py-1 text-xs font-medium transition disabled:opacity-40 ${
                selected ? "border-accent bg-accent-soft text-accent" : "border-line text-ink-2 hover:border-accent"
              }`}
            >
              {selected ? <Check size={14} aria-hidden="true" /> : <Plus size={14} aria-hidden="true" />}
              {selected ? "Comparing" : "Compare"}
            </button>
          </div>

          <ScoreBreakdown scores={scores} weights={weights} />

          {product.aspects.length > 0 && (
            <div className="grid gap-3 sm:grid-cols-2">
              {product.aspects.map((a) => (
                <AspectBar key={a.aspect} aspect={a} />
              ))}
            </div>
          )}

          <button
            onClick={() => setOpen((o) => !o)}
            aria-expanded={open}
            className="flex items-center gap-1 text-sm font-medium text-accent"
          >
            {open ? <ChevronUp size={16} aria-hidden="true" /> : <ChevronDown size={16} aria-hidden="true" />}
            {open ? "Hide" : "Why this product?"} — review evidence
          </button>

          {open && (
            <div className="space-y-4">
              <ul className="list-disc space-y-1 pl-5 text-sm text-ink-2">
                {product.reasons.map((r) => (
                  <li key={r}>{r}</li>
                ))}
              </ul>
              {product.aspects.map(
                (a) =>
                  a.snippets.length > 0 && (
                    <section key={a.aspect} className="space-y-2">
                      <h4 className="text-xs font-semibold uppercase tracking-wide text-muted">
                        What reviewers say about {a.label.toLowerCase()}
                      </h4>
                      <div className="grid gap-2">
                        {a.snippets.map((s) => (
                          <ReviewSnippet key={s.review_id + s.text} snippet={s} />
                        ))}
                      </div>
                    </section>
                  ),
              )}
              {product.evidence.length > 0 && (
                <section className="space-y-2">
                  <h4 className="text-xs font-semibold uppercase tracking-wide text-muted">
                    Review sentences most similar to your request
                  </h4>
                  <div className="grid gap-2">
                    {product.evidence.map((s) => (
                      <ReviewSnippet key={s.review_id + s.text} snippet={s} />
                    ))}
                  </div>
                </section>
              )}
            </div>
          )}
        </div>
      </div>
    </article>
  );
}
