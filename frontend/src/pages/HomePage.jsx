import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { MessageSquareText, ScanSearch, Scale } from "lucide-react";
import SearchBox from "../components/SearchBox.jsx";
import { api } from "../api/client.js";
import { compactNumber } from "../lib/format.js";

const STEPS = [
  {
    icon: ScanSearch,
    title: "Understand your request",
    text: "Your sentence is parsed for aspects (camera, battery, …), emphasis, brands and price limits, and embedded with a Sentence Transformer.",
  },
  {
    icon: MessageSquareText,
    title: "Read the reviews",
    text: "Every review sentence that mentions an aspect gets an aspect-level sentiment score, aggregated per phone with Bayesian smoothing.",
  },
  {
    icon: Scale,
    title: "Rank and explain",
    text: "Phones are ranked by relevance, sentiment on the aspects you care about and an adjusted star rating, with real review quotes as evidence.",
  },
];

export default function HomePage() {
  const [health, setHealth] = useState(null);
  const [stats, setStats] = useState(null);
  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth({ status: "offline" }));
    api.stats().then((s) => setStats(s.dataset)).catch(() => {});
  }, []);

  return (
    <div className="space-y-10">
      <section className="mx-auto max-w-3xl space-y-5 pt-6 text-center">
        <h1 className="text-3xl font-semibold tracking-tight sm:text-4xl">Find the right phone by describing it.</h1>
        <p className="text-ink-2">
          Recommendations from{" "}
          {stats ? (
            <>
              <strong className="text-ink">{compactNumber(stats.n_reviews)}</strong> real Amazon reviews of{" "}
              <strong className="text-ink">{stats.n_products.toLocaleString()}</strong> unlocked phones
            </>
          ) : (
            "real Amazon customer reviews"
          )}
          , explained with what owners actually wrote.
        </p>
      </section>

      <section className="mx-auto max-w-3xl text-left">
        {health && health.status !== "ok" && (
          <div role="alert" className="mb-3 rounded-xl bg-warn-soft px-4 py-3 text-sm text-warn">
            {health.status === "offline"
              ? "The API is not reachable. Start it with: uvicorn backend.app.main:app --port 8000"
              : `The API is running but has no data yet: ${health.detail ?? "run python scripts/run_pipeline.py"}`}
          </div>
        )}
        <SearchBox large />
      </section>

      <section className="grid gap-4 sm:grid-cols-3">
        {STEPS.map(({ icon: Icon, title, text }) => (
          <div key={title} className="card space-y-2 p-5">
            <Icon size={20} className="text-accent" aria-hidden="true" />
            <h2 className="font-semibold">{title}</h2>
            <p className="text-sm text-ink-2">{text}</p>
          </div>
        ))}
      </section>

      <p className="text-center text-sm text-ink-2">
        Curious how well it works? See the{" "}
        <Link to="/insights" className="font-medium text-accent">
          dataset and evaluation insights
        </Link>
        .
      </p>
    </div>
  );
}
