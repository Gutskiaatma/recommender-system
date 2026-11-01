import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import SearchBox from "../components/SearchBox.jsx";
import ProductCard from "../components/ProductCard.jsx";
import ParsedQuery from "../components/ParsedQuery.jsx";
import { ErrorBox, Loading } from "../components/Bits.jsx";
import { api } from "../api/client.js";
import { useCompare } from "../lib/compareContext.jsx";
import { METHOD_LABELS } from "../lib/format.js";

export default function SearchPage() {
  const [params] = useSearchParams();
  const query = params.get("q") ?? "";
  const method = params.get("method") ?? "hybrid";
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [aspectLabels, setAspectLabels] = useState({});
  const { setQuery } = useCompare();

  useEffect(() => {
    api
      .aspects()
      .then((list) => setAspectLabels(Object.fromEntries(list.map((a) => [a.aspect, a.label]))))
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (!query) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    setQuery(query);
    api
      .recommend(query, method, 10)
      .then((d) => !cancelled && setData(d))
      .catch((e) => !cancelled && setError(e))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [query, method, setQuery]);

  return (
    <div className="space-y-6">
      <SearchBox initialQuery={query} initialMethod={method} showExamples={!query} />
      {!query && <p className="text-sm text-ink-2">Describe what you want in plain English, or pick an example.</p>}
      {loading && <Loading label="Ranking phones and gathering review evidence…" />}
      <ErrorBox error={error} />
      {data && !loading && (
        <div className="space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <ParsedQuery parsed={data.parsed} aspectLabels={aspectLabels} />
            <span className="tabular text-xs text-muted">
              {METHOD_LABELS[data.method]} · {data.n_candidates.toLocaleString()} eligible phones · {data.elapsed_ms} ms
            </span>
          </div>
          {data.notes.map((n) => (
            <p key={n} className="text-xs text-ink-2">
              {n}
            </p>
          ))}
          {data.results.length === 0 ? (
            <p className="card p-6 text-sm text-ink-2">No phones match these constraints. Try relaxing the price or brand.</p>
          ) : (
            data.results.map((p, i) => <ProductCard key={p.product_id} product={p} weights={data.weights} defaultOpen={i === 0} />)
          )}
        </div>
      )}
    </div>
  );
}
