import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Check, Plus } from "lucide-react";
import ChartTooltip, { AXIS_PROPS, SERIES } from "../components/ChartTooltip.jsx";
import { ErrorBox, Loading, Stars } from "../components/Bits.jsx";
import { api } from "../api/client.js";
import { useCompare } from "../lib/compareContext.jsx";
import { compactNumber, formatPrice, pct, score100, shortTitle } from "../lib/format.js";

function ProductPicker() {
  const [search, setSearch] = useState("");
  const [items, setItems] = useState([]);
  const { toggle, has, full, items: selected, query } = useCompare();
  const navigate = useNavigate();
  useEffect(() => {
    const t = setTimeout(() => api.searchProducts(search, 12).then(setItems).catch(() => setItems([])), 200);
    return () => clearTimeout(t);
  }, [search]);
  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-2xl font-semibold">Compare phones</h1>
        <p className="text-sm text-ink-2">
          Pick 2–3 phones here, or tick “Compare” on search results. Comparisons use the same aspect scores as the ranking.
        </p>
      </div>
      <input
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        placeholder="Search by name or brand, e.g. Galaxy, iPhone, BLU"
        className="card w-full px-4 py-2.5 outline-none"
        aria-label="Search products"
      />
      <ul className="card divide-y divide-[var(--border)]">
        {items.map((p) => (
          <li key={p.product_id} className="flex items-center gap-3 px-4 py-2.5 text-sm">
            <div className="min-w-0 flex-1">
              <div className="truncate font-medium">{p.title}</div>
              <div className="text-xs text-ink-2">
                {p.brand} · {formatPrice(p.price)} · {p.mean_rating.toFixed(1)}★ ({compactNumber(p.n_reviews)})
              </div>
            </div>
            <button
              onClick={() => toggle(p)}
              disabled={!has(p.product_id) && full}
              className="flex items-center gap-1 rounded-lg border border-line px-2.5 py-1 text-xs disabled:opacity-40"
            >
              {has(p.product_id) ? <Check size={14} /> : <Plus size={14} />} {has(p.product_id) ? "Added" : "Add"}
            </button>
          </li>
        ))}
      </ul>
      <button
        disabled={selected.length < 2}
        onClick={() =>
          navigate(`/compare?ids=${selected.map((p) => p.product_id).join(",")}${query ? `&q=${encodeURIComponent(query)}` : ""}`)
        }
        className="rounded-lg bg-accent px-4 py-2 text-sm font-semibold text-accent-ink disabled:opacity-40"
      >
        Compare {selected.length} selected
      </button>
    </div>
  );
}

export default function ComparePage() {
  const [params] = useSearchParams();
  const ids = useMemo(() => (params.get("ids") ?? "").split(",").filter(Boolean), [params]);
  const query = params.get("q") ?? "";
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (ids.length < 2) return;
    setData(null);
    setError(null);
    api.compare(ids, query).then(setData).catch(setError);
  }, [ids, query]);

  if (ids.length < 2) return <ProductPicker />;
  if (error) return <ErrorBox error={error} />;
  if (!data) return <Loading label="Comparing…" />;

  const names = data.products.map((p, i) => `${i + 1}. ${shortTitle(p.title, 28)}`);
  const aspectRows = data.aspects.map((a, ai) => {
    const row = { aspect: a.label };
    data.products.forEach((p, pi) => {
      row[`p${pi}`] = Math.round(p.aspects[ai].score * 100);
      row[`n${pi}`] = p.aspects[ai].n_mentions;
    });
    return row;
  });
  const starRows = ["5", "4", "3", "2", "1"].map((s) => {
    const row = { star: `${s}★` };
    data.products.forEach((p, pi) => {
      row[`p${pi}`] = Math.round((p.rating_distribution[s] / p.n_reviews) * 1000) / 10;
    });
    return row;
  });

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">Comparison</h1>
        {query && (
          <p className="text-sm text-ink-2">
            Match scores computed for: <span className="font-medium text-ink">“{query}”</span>
          </p>
        )}
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        {data.products.map((p, i) => (
          <div key={p.product_id} className="card space-y-2 p-4">
            <div className="flex items-center gap-2 text-xs font-semibold text-ink-2">
              <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: SERIES[i] }} aria-hidden="true" />
              Phone {i + 1}
            </div>
            <Link to={`/product/${p.product_id}`} className="block font-semibold leading-snug hover:text-accent">
              {p.title}
            </Link>
            <div className="text-sm text-ink-2">
              {p.brand} · {formatPrice(p.price)}
            </div>
            <div className="flex items-center gap-1 text-sm">
              <Stars value={p.mean_rating} /> <span className="tabular">{p.mean_rating.toFixed(2)}</span>
              <span className="text-xs text-muted">({compactNumber(p.n_reviews)} reviews)</span>
            </div>
            {p.match && (
              <div className="tabular text-sm">
                Match score <span className="text-xl font-semibold">{score100(p.match.final)}</span>
                <span className="text-xs text-muted">/100</span>
              </div>
            )}
          </div>
        ))}
      </div>

      <section className="card p-4">
        <h2 className="mb-1 font-semibold">Aspect scores (0–100)</h2>
        <p className="mb-3 text-xs text-ink-2">Bayesian-smoothed share of positive review mentions; 50 = neutral.</p>
        <div className="h-72">
          <ResponsiveContainer>
            <BarChart data={aspectRows} barGap={2} margin={{ top: 4, right: 8, left: -16, bottom: 4 }}>
              <CartesianGrid vertical={false} stroke="var(--grid)" />
              <XAxis dataKey="aspect" {...AXIS_PROPS} interval={0} />
              <YAxis domain={[0, 100]} {...AXIS_PROPS} axisLine={false} />
              <Tooltip
                cursor={{ fill: "var(--surface-2)" }}
                content={
                  <ChartTooltip
                    valueFormatter={(v, item) => `${v} (${item.payload[`n${item.dataKey.slice(1)}`]} mentions)`}
                  />
                }
              />
              <Legend wrapperStyle={{ fontSize: 12, color: "var(--ink-2)" }} />
              {data.products.map((p, i) => (
                <Bar key={p.product_id} dataKey={`p${i}`} name={names[i]} fill={SERIES[i]} radius={[4, 4, 0, 0]} maxBarSize={28} />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>

      <section className="card overflow-x-auto p-4">
        <h2 className="mb-3 font-semibold">Side by side</h2>
        <table className="w-full min-w-[640px] text-sm">
          <thead>
            <tr className="border-b border-line text-left text-xs uppercase tracking-wide text-muted">
              <th className="py-2 pr-3 font-medium">Aspect</th>
              {data.products.map((p, i) => (
                <th key={p.product_id} className="py-2 pr-3 font-medium">
                  Phone {i + 1}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="tabular">
            {data.aspects.map((a, ai) => (
              <tr key={a.aspect} className="border-b border-line last:border-0">
                <td className="py-2 pr-3 font-medium">{a.label}</td>
                {data.products.map((p) => {
                  const s = p.aspects[ai];
                  const best = data.best_by_aspect[a.aspect] === p.product_id;
                  return (
                    <td key={p.product_id} className="py-2 pr-3">
                      {s.sufficient_evidence ? (
                        <span className={best ? "font-semibold text-ink" : "text-ink-2"}>
                          {best && <Check size={13} className="mr-1 inline text-good" aria-label="best" />}
                          {score100(s.score)} · {pct(s.positive_share)} pos ({s.n_mentions})
                        </span>
                      ) : (
                        <span className="text-muted">insufficient ({s.n_mentions})</span>
                      )}
                    </td>
                  );
                })}
              </tr>
            ))}
            <tr>
              <td className="py-2 pr-3 font-medium">Adjusted rating</td>
              {data.products.map((p) => (
                <td key={p.product_id} className="py-2 pr-3 text-ink-2">
                  {p.bayes_rating.toFixed(2)} / 5
                </td>
              ))}
            </tr>
          </tbody>
        </table>
        <p className="mt-2 text-xs text-muted">✓ marks the best phone per aspect among those with enough evidence (≥ 3 mentions).</p>
      </section>

      <section className="card p-4">
        <h2 className="mb-3 font-semibold">Star-rating distribution (% of reviews)</h2>
        <div className="h-64">
          <ResponsiveContainer>
            <BarChart data={starRows} barGap={2} margin={{ top: 4, right: 8, left: -16, bottom: 4 }}>
              <CartesianGrid vertical={false} stroke="var(--grid)" />
              <XAxis dataKey="star" {...AXIS_PROPS} />
              <YAxis unit="%" {...AXIS_PROPS} axisLine={false} />
              <Tooltip cursor={{ fill: "var(--surface-2)" }} content={<ChartTooltip valueFormatter={(v) => `${v}%`} />} />
              <Legend wrapperStyle={{ fontSize: 12, color: "var(--ink-2)" }} />
              {data.products.map((p, i) => (
                <Bar key={p.product_id} dataKey={`p${i}`} name={names[i]} fill={SERIES[i]} radius={[4, 4, 0, 0]} maxBarSize={28} />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>
    </div>
  );
}
