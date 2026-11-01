import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { Check, Plus } from "lucide-react";
import AspectBar from "../components/AspectBar.jsx";
import { ErrorBox, Loading, ReviewSnippet, Stars } from "../components/Bits.jsx";
import { api } from "../api/client.js";
import { useCompare } from "../lib/compareContext.jsx";
import { compactNumber, formatPrice } from "../lib/format.js";

export default function ProductPage() {
  const { productId } = useParams();
  const [product, setProduct] = useState(null);
  const [error, setError] = useState(null);
  const { toggle, has, full } = useCompare();

  useEffect(() => {
    setProduct(null);
    setError(null);
    api.product(productId).then(setProduct).catch(setError);
  }, [productId]);

  if (error) return <ErrorBox error={error} />;
  if (!product) return <Loading />;
  const selected = has(product.product_id);
  const stars = ["5", "4", "3", "2", "1"];
  const maxStar = Math.max(...stars.map((s) => product.rating_distribution[s]));

  return (
    <div className="space-y-6">
      <div className="card space-y-3 p-5">
        <h1 className="text-xl font-semibold leading-snug">{product.title}</h1>
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-sm text-ink-2">
          <span className="font-medium text-ink">{product.brand}</span>
          <span>{formatPrice(product.price)}</span>
          <span className="flex items-center gap-1">
            <Stars value={product.mean_rating} /> {product.mean_rating.toFixed(2)} from {compactNumber(product.n_reviews)} reviews
          </span>
          <span>Adjusted rating {product.bayes_rating.toFixed(2)}</span>
          <button
            onClick={() => toggle(product)}
            disabled={!selected && full}
            className="ml-auto flex items-center gap-1 rounded-lg border border-line px-3 py-1 text-xs font-medium disabled:opacity-40"
          >
            {selected ? <Check size={14} /> : <Plus size={14} />} {selected ? "In comparison" : "Add to compare"}
          </button>
        </div>
        {product.n_variants > 1 && (
          <details className="text-xs text-ink-2">
            <summary className="cursor-pointer">{product.n_variants} listing variants merged (shared review pool)</summary>
            <ul className="mt-1 list-disc pl-5">
              {product.variant_titles.map((t) => (
                <li key={t}>{t}</li>
              ))}
            </ul>
          </details>
        )}
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <section className="card space-y-4 p-5 lg:col-span-2">
          <h2 className="font-semibold">Aspect sentiment from reviews</h2>
          <p className="text-xs text-ink-2">The tick on each bar marks the catalogue average for that aspect.</p>
          {product.aspects.map((a) => (
            <div key={a.aspect} className="space-y-2 border-t border-line pt-3 first:border-0 first:pt-0">
              <AspectBar aspect={a} />
              <div className="grid gap-2">
                {a.snippets.map((s) => (
                  <ReviewSnippet key={s.review_id + s.text} snippet={s} />
                ))}
              </div>
            </div>
          ))}
        </section>
        <section className="card h-fit space-y-2 p-5">
          <h2 className="font-semibold">Rating distribution</h2>
          <ul className="space-y-1.5">
            {stars.map((s) => (
              <li key={s} className="flex items-center gap-2 text-sm">
                <span className="w-6 text-ink-2">{s}★</span>
                <div className="h-2 flex-1 rounded-full bg-surface-2">
                  <div
                    className="h-2 rounded-full"
                    style={{ width: `${(product.rating_distribution[s] / maxStar) * 100}%`, background: "var(--series-1)" }}
                  />
                </div>
                <span className="tabular w-12 text-right text-xs text-muted">{product.rating_distribution[s]}</span>
              </li>
            ))}
          </ul>
        </section>
      </div>
    </div>
  );
}
