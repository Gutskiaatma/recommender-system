import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { ArrowRight } from "lucide-react";
import { api } from "../api/client.js";
import { METHOD_LABELS } from "../lib/format.js";

const METHOD_HELP = {
  hybrid: "Relevance + aspect sentiment + rating",
  semantic: "Sentence-embedding similarity only",
  tfidf: "Keyword (TF-IDF) similarity only",
};

export default function SearchBox({ initialQuery = "", initialMethod = "hybrid", showExamples = true, large = false }) {
  const [query, setQuery] = useState(initialQuery);
  const [method, setMethod] = useState(initialMethod);
  const [examples, setExamples] = useState([]);
  const navigate = useNavigate();

  useEffect(() => setQuery(initialQuery), [initialQuery]);
  useEffect(() => setMethod(initialMethod), [initialMethod]);
  useEffect(() => {
    if (showExamples) api.examples().then(setExamples).catch(() => setExamples([]));
  }, [showExamples]);

  const submit = (q = query) => {
    const text = q.trim();
    if (text.length < 3) return;
    navigate(`/search?q=${encodeURIComponent(text)}&method=${method}`);
  };

  return (
    <div className="space-y-3">
      <form
        onSubmit={(e) => {
          e.preventDefault();
          submit();
        }}
        className="card p-2 shadow-sm"
      >
        <label htmlFor="query" className="sr-only">
          Describe the phone you want
        </label>
        <textarea
          id="query"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              submit();
            }
          }}
          rows={large ? 3 : 2}
          maxLength={500}
          placeholder="e.g. A phone with an excellent camera, long battery life and smooth gaming performance under $400"
          className={`w-full resize-none bg-transparent px-3 py-2 outline-none placeholder:text-muted ${large ? "text-lg" : "text-base"}`}
        />
        <div className="flex flex-wrap items-center gap-2 border-t border-line px-2 pt-2">
          <div role="radiogroup" aria-label="Ranking method" className="flex rounded-lg bg-surface-2 p-0.5">
            {Object.entries(METHOD_LABELS).map(([key, label]) => (
              <button
                key={key}
                type="button"
                role="radio"
                aria-checked={method === key}
                title={METHOD_HELP[key]}
                onClick={() => setMethod(key)}
                className={`rounded-md px-3 py-1 text-xs font-medium transition ${
                  method === key ? "bg-surface text-ink shadow-sm" : "text-ink-2 hover:text-ink"
                }`}
              >
                {label}
              </button>
            ))}
          </div>
          <span className="hidden text-xs text-muted md:inline">{METHOD_HELP[method]}</span>
          <button
            type="submit"
            disabled={query.trim().length < 3}
            className="ml-auto flex items-center gap-1.5 rounded-lg bg-accent px-4 py-2 text-sm font-semibold text-accent-ink transition hover:opacity-90 disabled:opacity-40"
          >
            Recommend <ArrowRight size={16} aria-hidden="true" />
          </button>
        </div>
      </form>
      {showExamples && examples.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {examples.map((ex) => (
            <button
              key={ex}
              onClick={() => {
                setQuery(ex);
                submit(ex);
              }}
              className="rounded-full border border-line bg-surface px-3 py-1 text-left text-xs text-ink-2 transition hover:border-accent hover:text-ink"
            >
              {ex}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
