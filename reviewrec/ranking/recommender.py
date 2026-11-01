"""The recommendation engine used by the API and the evaluation.

Methods
-------
tfidf     baseline: rank by TF-IDF cosine similarity only
semantic  rank by Sentence-Transformer relevance only (title + review evidence)
hybrid    review-aware: take the top-N semantic candidates, re-rank with
          S = alpha*R + beta*A + gamma*Q (see ranking/scoring.py)

Price and brand constraints parsed from the query are applied as hard filters
for every method, so the comparison between methods stays fair.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from reviewrec.aspects.lexicon import AspectLexicon, get_lexicon
from reviewrec.config import Settings, get_settings
from reviewrec.explain.evidence import aspect_snippets, aspect_summary, build_reasons, query_snippets
from reviewrec.ranking.query_parser import ParsedQuery, parse_query
from reviewrec.ranking.scoring import Weights, aspect_component, combine, minmax
from reviewrec.retrieval.semantic import Embedder, make_embedder
from reviewrec.store import ArtifactStore

METHODS = ("tfidf", "semantic", "hybrid")


def _clean(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return [_clean(v) for v in value.tolist()]
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


class Recommender:
    def __init__(self, store: ArtifactStore, embedder: Embedder, settings: Settings | None = None, lexicon: AspectLexicon | None = None):
        self.store = store
        self.embedder = embedder
        self.settings = settings or get_settings()
        self.lexicon = lexicon or get_lexicon()
        self.weights = Weights(
            float(self.settings.get("scoring.alpha", 0.55)),
            float(self.settings.get("scoring.beta", 0.30)),
            float(self.settings.get("scoring.gamma", 0.15)),
        )
        self.neutral_band = float(self.settings.get("sentiment.neutral_band", 0.3))
        self.min_mentions = int(self.settings.get("aggregation.min_mentions_for_claim", 3))
        self.candidate_pool = int(self.settings.get("retrieval.candidate_pool", 100))
        self.brands = sorted(self.store.products["brand"].dropna().unique().tolist())
        self.quality = self.store.products["quality"].to_numpy(dtype=np.float64)

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "Recommender":
        settings = settings or get_settings()
        store = ArtifactStore.load(settings)
        embedder = make_embedder(settings.embedding_model, device=settings.get("embeddings.device", "cpu"))
        return cls(store, embedder, settings)

    # ------------------------------------------------------------------ parsing
    def parse(self, query: str) -> ParsedQuery:
        return parse_query(
            query,
            self.lexicon,
            known_brands=self.brands,
            default_weight=float(self.settings.get("scoring.default_aspect_weight", 1.0)),
            emphasis_weight=float(self.settings.get("scoring.emphasis_weight", 1.5)),
        )

    def _filter_mask(self, parsed: ParsedQuery) -> tuple[np.ndarray, list[str]]:
        p = self.store.products
        mask = np.ones(len(p), dtype=bool)
        notes: list[str] = []
        if parsed.price_min is not None or parsed.price_max is not None:
            price = p["price"].to_numpy(dtype=np.float64)
            known = ~np.isnan(price)
            ok = known.copy()
            if parsed.price_min is not None:
                ok &= price >= parsed.price_min
            if parsed.price_max is not None:
                ok &= price <= parsed.price_max
            mask &= ok
            notes.append(f"Price filter applied; {int((~known).sum())} products without a listed price were excluded.")
        if parsed.brands:
            mask &= p["brand"].isin(parsed.brands).to_numpy()
        if parsed.excluded_brands:
            mask &= ~p["brand"].isin(parsed.excluded_brands).to_numpy()
        return mask, notes

    # ------------------------------------------------------------------ scoring
    def score_all(self, query: str, method: str = "hybrid", weights: Weights | None = None, parsed: ParsedQuery | None = None) -> dict[str, Any]:
        """Return raw score arrays for every product (used by evaluation)."""
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        parsed = parsed or self.parse(query)
        mask, notes = self._filter_mask(parsed)
        query_emb = self.embedder.encode_query(query)
        sem = self.store.semantic.score(query_emb)
        n = len(self.store.products)
        query_aspects = [a for a in parsed.aspects if a in self.store.aspect_names]
        aw = np.array([parsed.aspects[a] for a in query_aspects], dtype=np.float64)
        cols = [self.store.aspect_names.index(a) for a in query_aspects]
        aspect_all = aspect_component(self.store.aspect_matrices["score"][:, cols], aw) if cols else np.zeros(n)

        relevance_raw = self.store.tfidf.score(query) if method == "tfidf" else sem["relevance"]
        final = np.full(n, -np.inf)
        relevance_norm = np.zeros(n)
        used_weights = None
        eligible = np.flatnonzero(mask)
        if method in ("tfidf", "semantic"):
            final[eligible] = relevance_raw[eligible]
            if len(eligible):
                relevance_norm[eligible] = minmax(relevance_raw[eligible])
        else:
            w = (weights or self.weights).normalised(bool(cols))
            used_weights = w
            pool = eligible[np.argsort(-relevance_raw[eligible])[: self.candidate_pool]]
            if len(pool):
                relevance_norm[pool] = minmax(relevance_raw[pool])
                final[pool] = combine(relevance_norm[pool], aspect_all[pool], self.quality[pool], w)
        return {
            "parsed": parsed,
            "notes": notes,
            "query_emb": query_emb,
            "semantic": sem,
            "relevance_raw": relevance_raw,
            "relevance_norm": relevance_norm,
            "aspect": aspect_all,
            "quality": self.quality,
            "final": final,
            "query_aspects": query_aspects,
            "weights": used_weights,
            "n_eligible": int(mask.sum()),
        }

    def rank(self, query: str, method: str = "hybrid", top_k: int = 10, weights: Weights | None = None) -> list[str]:
        scores = self.score_all(query, method, weights)
        final = scores["final"]
        order = np.argsort(-final, kind="stable")
        order = [i for i in order if np.isfinite(final[i])][:top_k]
        return [self.store.products["product_id"].iat[i] for i in order]

    # ------------------------------------------------------------------ explain
    def product_card(self, idx: int, query_aspects: list[str], query_emb: np.ndarray | None, sentence_sims: np.ndarray | None) -> dict[str, Any]:
        row = self.store.products.iloc[idx]
        product = {
            "product_id": row["product_id"],
            "title": row["title"],
            "brand": row["brand"],
            "price": _clean(row["price"]),
            "n_reviews": int(row["n_reviews"]),
            "mean_rating": round(float(row["mean_rating"]), 3),
            "bayes_rating": round(float(row["bayes_rating"]), 3),
            "rating_distribution": {str(s): int(row[f"stars_{s}"]) for s in range(1, 6)},
            "n_variants": int(row["n_variants"]),
            "variant_titles": list(row["variant_titles"])[:8],
        }
        shown_aspects = query_aspects or self.top_aspects(idx, 3)
        summaries = [aspect_summary(self.store, idx, a, self.min_mentions) for a in shown_aspects]
        for s in summaries:
            s["label"] = self.lexicon.labels.get(s["aspect"], s["aspect"])
            s["snippets"] = aspect_snippets(self.store, product["product_id"], s["aspect"], query_emb, self.neutral_band)
        product["aspects"] = summaries
        product["reasons"] = build_reasons(product, summaries, self.lexicon.labels, bool(query_aspects))
        shown = {snip["text"].strip().lower() for s in summaries for snip in s["snippets"]}
        product["evidence"] = (
            query_snippets(self.store, product["product_id"], sentence_sims, exclude_texts=shown) if sentence_sims is not None else []
        )
        return product

    def top_aspects(self, idx: int, n: int = 3) -> list[str]:
        """Most-discussed aspects of a product (used when the query names none)."""
        counts = self.store.aspect_matrices["n_mentions"][idx]
        order = np.argsort(-counts)[:n]
        return [self.store.aspect_names[j] for j in order if counts[j] > 0]

    def recommend(self, query: str, method: str = "hybrid", top_k: int = 10) -> dict[str, Any]:
        t0 = time.perf_counter()
        scores = self.score_all(query, method)
        final = scores["final"]
        order = [i for i in np.argsort(-final, kind="stable") if np.isfinite(final[i])][:top_k]
        results = []
        for rank, idx in enumerate(order, start=1):
            card = self.product_card(idx, scores["query_aspects"], scores["query_emb"], scores["semantic"]["sentence_sims"])
            card["rank"] = rank
            card["scores"] = {
                "final": round(float(final[idx]), 4),
                "relevance": round(float(scores["relevance_norm"][idx]), 4),
                "relevance_raw": round(float(scores["relevance_raw"][idx]), 4),
                "aspect": round(float(scores["aspect"][idx]), 4) if scores["query_aspects"] else None,
                "quality": round(float(scores["quality"][idx]), 4),
            }
            results.append(card)
        w = scores["weights"]
        return {
            "query": query,
            "method": method,
            "parsed": scores["parsed"].to_dict(),
            "weights": None if w is None else {"alpha": round(w.alpha, 4), "beta": round(w.beta, 4), "gamma": round(w.gamma, 4)},
            "notes": scores["notes"],
            "n_candidates": scores["n_eligible"],
            "results": results,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
        }
