"""Side-by-side comparison of up to three products."""

from __future__ import annotations

from typing import Any

import numpy as np

from reviewrec.explain.evidence import aspect_summary
from reviewrec.ranking.recommender import Recommender


def compare_products(
    rec: Recommender, product_ids: list[str], query: str | None = None, aspects: list[str] | None = None
) -> dict[str, Any]:
    store = rec.store
    unknown = [pid for pid in product_ids if pid not in store.product_row]
    if unknown:
        raise KeyError(f"Unknown product ids: {unknown}")
    idxs = [store.product_row[pid] for pid in product_ids]

    parsed = rec.parse(query) if query else None
    if aspects:
        shown = [a for a in aspects if a in store.aspect_names]
    elif parsed and parsed.aspects:
        shown = [a for a in parsed.aspects if a in store.aspect_names]
    else:
        shown = list(store.aspect_names)
    match_scores: dict[str, dict[str, float]] = {}
    if query:
        s = rec.score_all(query, "hybrid", parsed=parsed)
        for pid, i in zip(product_ids, idxs):
            # Products outside the candidate pool still get a comparable score.
            w = rec.weights.normalised(bool(s["query_aspects"]))
            rel = float(np.clip(s["relevance_norm"][i], 0, 1)) if np.isfinite(s["final"][i]) else 0.0
            match_scores[pid] = {
                "final": round(w.alpha * rel + w.beta * float(s["aspect"][i]) + w.gamma * float(s["quality"][i]), 4),
                "relevance": round(rel, 4),
                "aspect": round(float(s["aspect"][i]), 4) if s["query_aspects"] else None,
                "quality": round(float(s["quality"][i]), 4),
            }

    products = []
    for pid, i in zip(product_ids, idxs):
        row = store.products.iloc[i]
        summaries = []
        for a in shown:
            summary = aspect_summary(store, i, a, rec.min_mentions)
            summary["label"] = rec.lexicon.labels.get(a, a)
            summaries.append(summary)
        products.append(
            {
                "product_id": pid,
                "title": row["title"],
                "brand": row["brand"],
                "price": None if np.isnan(row["price"]) else float(row["price"]),
                "n_reviews": int(row["n_reviews"]),
                "mean_rating": round(float(row["mean_rating"]), 3),
                "bayes_rating": round(float(row["bayes_rating"]), 3),
                "rating_distribution": {str(k): int(row[f"stars_{k}"]) for k in range(1, 6)},
                "aspects": summaries,
                "match": match_scores.get(pid),
            }
        )

    # Best product per aspect, only among products with sufficient evidence.
    best: dict[str, str | None] = {}
    for a_pos, a in enumerate(shown):
        candidates = [(p["aspects"][a_pos]["score"], p["product_id"]) for p in products if p["aspects"][a_pos]["sufficient_evidence"]]
        best[a] = max(candidates)[1] if candidates else None
    return {
        "query": query,
        "aspects": [{"aspect": a, "label": rec.lexicon.labels.get(a, a)} for a in shown],
        "products": products,
        "best_by_aspect": best,
    }
