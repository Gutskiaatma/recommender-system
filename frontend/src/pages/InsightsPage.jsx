import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import ChartTooltip, { AXIS_PROPS, SERIES } from "../components/ChartTooltip.jsx";
import { ErrorBox, Loading, StatTile } from "../components/Bits.jsx";
import { api } from "../api/client.js";
import { METHOD_LABELS, compactNumber, pct } from "../lib/format.js";

function ChartCard({ title, subtitle, children, height = "h-64" }) {
  return (
    <section className="card p-4">
      <h2 className="font-semibold">{title}</h2>
      {subtitle && <p className="text-xs text-ink-2">{subtitle}</p>}
      <div className={`${height} mt-3`}>{children}</div>
    </section>
  );
}

function SingleBar({ data, x, y, name, unit = "", formatter = (v) => v, layout = "horizontal" }) {
  const vertical = layout === "vertical";
  return (
    <ResponsiveContainer>
      <BarChart data={data} layout={layout} margin={{ top: 4, right: 16, left: vertical ? 24 : -12, bottom: 4 }}>
        <CartesianGrid vertical={vertical} horizontal={!vertical} stroke="var(--grid)" />
        {vertical ? (
          <>
            <XAxis type="number" {...AXIS_PROPS} unit={unit} />
            <YAxis type="category" dataKey={x} width={110} {...AXIS_PROPS} axisLine={false} />
          </>
        ) : (
          <>
            <XAxis dataKey={x} {...AXIS_PROPS} />
            <YAxis {...AXIS_PROPS} axisLine={false} unit={unit} />
          </>
        )}
        <Tooltip cursor={{ fill: "var(--surface-2)" }} content={<ChartTooltip valueFormatter={formatter} />} />
        <Bar dataKey={y} name={name} fill="var(--series-1)" radius={vertical ? [0, 4, 4, 0] : [4, 4, 0, 0]} maxBarSize={32} />
      </BarChart>
    </ResponsiveContainer>
  );
}

const METRICS = ["P@5", "R@10", "NDCG@10", "MRR"];

function RankingEvaluation({ ranking }) {
  const methods = Object.keys(ranking.methods);
  const rows = METRICS.map((m) => {
    const row = { metric: m };
    methods.forEach((meth) => (row[meth] = ranking.methods[meth][m]));
    return row;
  });
  return (
    <section className="card space-y-4 p-4">
      <div>
        <h2 className="font-semibold">Ranking quality — {ranking.split} queries (n = {ranking.n_queries})</h2>
        <p className="text-xs text-ink-2">{ranking.labels_note}</p>
      </div>
      <div className="h-64">
        <ResponsiveContainer>
          <BarChart data={rows} barGap={2} margin={{ top: 4, right: 8, left: -16, bottom: 4 }}>
            <CartesianGrid vertical={false} stroke="var(--grid)" />
            <XAxis dataKey="metric" {...AXIS_PROPS} />
            <YAxis domain={[0, 1]} {...AXIS_PROPS} axisLine={false} />
            <Tooltip cursor={{ fill: "var(--surface-2)" }} content={<ChartTooltip valueFormatter={(v) => v.toFixed(3)} />} />
            <Legend wrapperStyle={{ fontSize: 12, color: "var(--ink-2)" }} />
            {methods.map((m, i) => (
              <Bar key={m} dataKey={m} name={METHOD_LABELS[m] ?? m} fill={SERIES[i]} radius={[4, 4, 0, 0]} maxBarSize={28} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
      <table className="tabular w-full text-sm">
        <thead>
          <tr className="border-b border-line text-left text-xs uppercase tracking-wide text-muted">
            <th className="py-1.5 font-medium">Method</th>
            {METRICS.map((m) => (
              <th key={m} className="py-1.5 font-medium">
                {m}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {[...methods, ...Object.keys(ranking.ablations ?? {})].map((m) => {
            const r = ranking.methods[m] ?? ranking.ablations[m];
            return (
              <tr key={m} className="border-b border-line last:border-0">
                <td className="py-1.5 pr-2">{METHOD_LABELS[m] ?? m}</td>
                {METRICS.map((k) => (
                  <td key={k} className="py-1.5">
                    {r[k].toFixed(3)}
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </section>
  );
}

function AbsaEvaluation({ absa }) {
  return (
    <section className="card space-y-3 p-4">
      <div>
        <h2 className="font-semibold">Aspect sentiment accuracy (n = {absa.n} labelled mentions)</h2>
        <p className="text-xs text-ink-2">{absa.labels_note}</p>
      </div>
      <table className="tabular w-full text-sm">
        <thead>
          <tr className="border-b border-line text-left text-xs uppercase tracking-wide text-muted">
            <th className="py-1.5 font-medium">Model</th>
            <th className="py-1.5 font-medium">Accuracy</th>
            <th className="py-1.5 font-medium">Macro-F1</th>
            <th className="py-1.5 font-medium">F1 neg</th>
            <th className="py-1.5 font-medium">F1 pos</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(absa.backends).map(([name, r]) => (
            <tr key={name} className="border-b border-line last:border-0">
              <td className="py-1.5">{name}</td>
              <td className="py-1.5">{r.accuracy.toFixed(3)}</td>
              <td className="py-1.5">{r.macro_f1.toFixed(3)}</td>
              <td className="py-1.5">{r.per_class.negative.f1.toFixed(3)}</td>
              <td className="py-1.5">{r.per_class.positive.f1.toFixed(3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

export default function InsightsPage() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  useEffect(() => {
    api.stats().then(setData).catch(setError);
  }, []);
  if (error) return <ErrorBox error={error} />;
  if (!data) return <Loading />;
  const d = data.dataset;
  const ratingRows = ["1", "2", "3", "4", "5"].map((s) => ({ star: `${s}★`, reviews: d.rating_distribution[s] }));
  const aspectRows = [...d.aspects].sort((a, b) => b.mentions - a.mentions);
  const sentimentRows = aspectRows.map((a) => ({ label: a.label, positive: Math.round(a.positive_share * 1000) / 10 }));
  const ev = data.evaluation ?? {};

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">Dataset & evaluation insights</h1>
        <p className="text-sm text-ink-2">
          Everything on this page is computed by the pipeline and evaluation scripts from the real dataset.
        </p>
      </div>
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <StatTile label="Phones" value={d.n_products.toLocaleString()} hint={`${d.n_brands} brands`} />
        <StatTile label="Reviews" value={compactNumber(d.n_reviews)} hint={`${compactNumber(d.n_reviews_nlp)} used for NLP`} />
        <StatTile label="Mean rating" value={`${d.mean_rating.toFixed(2)}★`} hint="review-weighted" />
        <StatTile label="Aspect mentions" value={compactNumber(data.manifest.n_mentions)} hint={`sentiment: ${data.manifest.sentiment_backend}`} />
      </div>

      {ev.ranking ? <RankingEvaluation ranking={ev.ranking} /> : (
        <p className="card p-4 text-sm text-ink-2">No evaluation results yet. Run <code>python scripts/evaluate.py</code>.</p>
      )}
      {ev.absa && <AbsaEvaluation absa={ev.absa} />}

      <div className="grid gap-4 lg:grid-cols-2">
        <ChartCard title="Aspect mentions" subtitle="Review sentences mentioning each aspect">
          <SingleBar data={aspectRows} x="label" y="mentions" name="Mentions" layout="vertical" formatter={(v) => v.toLocaleString()} />
        </ChartCard>
        <ChartCard title="Positive share by aspect" subtitle="Share of mentions classified positive">
          <SingleBar data={sentimentRows} x="label" y="positive" name="Positive" unit="%" layout="vertical" formatter={(v) => `${v}%`} />
        </ChartCard>
        <ChartCard title="Star ratings" subtitle="All de-duplicated reviews of included phones">
          <SingleBar data={ratingRows} x="star" y="reviews" name="Reviews" formatter={(v) => v.toLocaleString()} />
        </ChartCard>
        <ChartCard title="Listed price" subtitle={`Price known for ${pct(d.price.known_share)} of phones · median $${d.price.median}`}>
          <SingleBar data={d.price.histogram} x="bucket" y="count" name="Phones" />
        </ChartCard>
      </div>

      {data.data_quality && (
        <section className="card p-4">
          <h2 className="mb-2 font-semibold">Data-quality report</h2>
          <dl className="tabular grid gap-x-6 gap-y-1 text-sm sm:grid-cols-2">
            {Object.entries(data.data_quality).map(([k, v]) => (
              <div key={k} className="flex justify-between border-b border-line py-1">
                <dt className="text-ink-2">{k.replaceAll("_", " ")}</dt>
                <dd className="font-medium">{typeof v === "number" ? v.toLocaleString() : String(v)}</dd>
              </div>
            ))}
          </dl>
        </section>
      )}
    </div>
  );
}
