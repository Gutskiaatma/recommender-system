"""Evidence selection for explanations.

Rules that keep explanations honest:
  * Every snippet is a real sentence from the dataset with its review id.
  * For each requested aspect we show the strongest *positive and negative*
    mentions (if both exist), not only praise, plus the full counts.
  * When an aspect has fewer than ``min_mentions`` mentions we say so instead
    of making a claim.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from reviewrec.aspects.sentiment import polarity_label
from reviewrec.store import ArtifactStore


def _snippet(row: Any, sim: float | None, neutral_band: float) -> dict[str, Any]:
    return {
        "text": row.sentence,
        "clause": row.clause,
        "term": row.term,
        "polarity": polarity_label(float(row.v), neutral_band),
        "sentiment": round(float(row.v), 3),
        "rating": int(row.rating),
        "review_id": row.review_id,
        "query_similarity": None if sim is None else round(float(sim), 3),
    }


def aspect_snippets(
    store: ArtifactStore,
    product_id: str,
    aspect: str,
    query_emb: np.ndarray | None,
    neutral_band: float,
    n_positive: int = 2,
    n_negative: int = 1,
) -> list[dict[str, Any]]:
    rows = store.mention_rows_by_product.get(product_id)
    if rows is None or len(rows) == 0:
        return []
    sub = store.mentions.iloc[rows]
    keep = (sub["aspect"] == aspect).to_numpy()
    if not keep.any():
        return []
    rows = rows[keep]
    sub = sub[keep]
    if query_emb is not None:
        sims = np.asarray(store.mention_embeddings[rows], dtype=np.float32) @ query_emb.astype(np.float32)
    else:
        sims = np.zeros(len(rows), dtype=np.float32)
    # Prefer informative clauses: similarity to the query plus a small bonus for strong polarity.
    v = sub["v"].to_numpy()
    rank_score = sims + 0.25 * np.abs(v)
    order = np.argsort(-rank_score)
    chosen: list[dict[str, Any]] = []
    pos = [i for i in order if v[i] > neutral_band][:n_positive]
    neg = [i for i in order if v[i] < -neutral_band][:n_negative]
    picked = pos + neg
    if not picked:
        picked = list(order[: n_positive + n_negative])
    seen_text: set[str] = set()
    for i in picked:
        row = sub.iloc[i]
        key = row.sentence.strip().lower()
        if key in seen_text:
            continue
        seen_text.add(key)
        chosen.append(_snippet(row, float(sims[i]) if query_emb is not None else None, neutral_band))
    return chosen


def query_snippets(
    store: ArtifactStore, product_id: str, sentence_sims: np.ndarray, n: int = 3, exclude_texts: set[str] | None = None
) -> list[dict[str, Any]]:
    """Top review sentences by similarity to the query, skipping duplicates and
    sentences already shown as aspect evidence."""
    rows = store.sentence_rows_by_product.get(product_id)
    if rows is None or len(rows) == 0:
        return []
    sims = sentence_sims[rows]
    order = np.argsort(-sims)
    out = []
    seen: set[str] = set(exclude_texts or ())
    for i in order:
        if len(out) >= n:
            break
        row = store.sentences.iloc[rows[i]]
        key = row.sentence.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "text": row.sentence,
                "rating": int(row.rating),
                "review_id": row.review_id,
                "query_similarity": round(float(sims[i]), 3),
            }
        )
    return out


def aspect_summary(store: ArtifactStore, product_idx: int, aspect: str, min_mentions: int) -> dict[str, Any]:
    j = store.aspect_names.index(aspect)
    m = store.aspect_matrices
    n = int(m["n_mentions"][product_idx, j])
    prior = (store.aspect_priors[aspect] + 1) / 2
    score = float(m["score"][product_idx, j])
    return {
        "aspect": aspect,
        "score": round(score, 4),
        "catalog_average": round(prior, 4),
        "n_mentions": n,
        "n_positive": int(m["n_positive"][product_idx, j]),
        "n_negative": int(m["n_negative"][product_idx, j]),
        "n_neutral": int(m["n_neutral"][product_idx, j]),
        "positive_share": round(float(m["n_positive"][product_idx, j]) / n, 4) if n else None,
        "confidence": round(float(m["confidence"][product_idx, j]), 4),
        "sufficient_evidence": n >= min_mentions,
    }


def build_reasons(
    product: dict[str, Any], aspect_summaries: list[dict[str, Any]], labels: dict[str, str], has_query_aspects: bool
) -> list[str]:
    reasons: list[str] = []
    for s in aspect_summaries:
        label = labels.get(s["aspect"], s["aspect"])
        if not s["sufficient_evidence"]:
            reasons.append(f"{label}: not enough evidence ({s['n_mentions']} mention(s)), treated as average.")
            continue
        share = s["positive_share"] or 0.0
        diff = s["score"] - s["catalog_average"]
        direction = "above" if diff > 0.02 else "below" if diff < -0.02 else "close to"
        reasons.append(
            f"{label}: {share:.0%} of {s['n_mentions']} mentions are positive "
            f"({s['n_negative']} negative); {direction} the catalogue average."
        )
    if not has_query_aspects:
        reasons.append("No specific aspect recognised in the query, so ranking uses relevance and overall rating.")
    reasons.append(
        f"Rated {product['mean_rating']:.2f}/5 from {product['n_reviews']:,} reviews "
        f"(Bayesian-adjusted {product['bayes_rating']:.2f})."
    )
    return reasons
