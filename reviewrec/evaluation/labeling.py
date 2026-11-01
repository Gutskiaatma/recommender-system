"""Build *blind* labelling sets.

Ranking relevance (TREC-style pooling)
    For each test query we take the top-k products from every method, merge
    them, shuffle them with a fixed seed and hide which method returned them.
    The annotator sees the product title, price, rating and a random sample
    of raw review sentences (no model scores) and assigns a grade:
        2 = highly relevant: right kind of phone and the reviews clearly support
            the requested aspects/constraints
        1 = partially relevant: matches some requirements or evidence is mixed
        0 = not relevant: violates a constraint or reviews contradict the request

ABSA (aspect sentiment)
    A stratified random sample of aspect mentions. The annotator labels the
    sentiment the clause expresses *about that aspect* (positive / negative /
    neutral) and whether the aspect was detected correctly (lexicon precision).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from reviewrec.ranking.recommender import METHODS, Recommender

RUBRIC = (
    "2 = highly relevant (right phone type and reviews clearly support the requested aspects/constraints); "
    "1 = partially relevant (some requirements met or mixed evidence); "
    "0 = not relevant (violates a constraint or reviews contradict the request)."
)


def evidence_for_annotator(rec: Recommender, product_id: str, aspects: list[str], rng: np.random.Generator, per_aspect: int = 4) -> str:
    """Random raw review sentences (no model scores) for the annotator."""
    store = rec.store
    lines: list[str] = []
    rows = store.mention_rows_by_product.get(product_id, np.array([], dtype=int))
    mentions = store.mentions.iloc[rows] if len(rows) else store.mentions.iloc[[]]
    for aspect in aspects:
        sub = mentions[mentions["aspect"] == aspect]
        lines.append(f"[{aspect}: {len(sub)} mentions]")
        if len(sub):
            pick = sub.iloc[rng.choice(len(sub), size=min(per_aspect, len(sub)), replace=False)]
            lines.extend(f"  ({r.rating}*) {r.sentence}" for r in pick.itertuples())
    srows = store.sentence_rows_by_product.get(product_id, np.array([], dtype=int))
    if len(srows):
        pick = store.sentences.iloc[rng.choice(srows, size=min(4, len(srows)), replace=False)]
        lines.append("[random review sentences]")
        lines.extend(f"  ({r.rating}*) {r.sentence}" for r in pick.itertuples())
    return "\n".join(lines)


def build_relevance_pool(rec: Recommender, queries: pd.DataFrame, k: int = 10, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    products = rec.store.products.set_index("product_id")
    for q in queries.itertuples(index=False):
        parsed = rec.parse(q.query)
        pooled: list[str] = []
        for method in METHODS:
            for pid in rec.rank(q.query, method, top_k=k):
                if pid not in pooled:
                    pooled.append(pid)
        order = rng.permutation(len(pooled))
        for i in order:
            pid = pooled[i]
            p = products.loc[pid]
            rows.append(
                {
                    "query_id": q.query_id,
                    "split": q.split,
                    "query": q.query,
                    "product_id": pid,
                    "title": p["title"],
                    "brand": p["brand"],
                    "price": None if pd.isna(p["price"]) else round(float(p["price"]), 2),
                    "mean_rating": round(float(p["mean_rating"]), 2),
                    "n_reviews": int(p["n_reviews"]),
                    "evidence": evidence_for_annotator(rec, pid, list(parsed.aspects), rng),
                    "grade": "",
                }
            )
    return pd.DataFrame(rows)


def build_absa_sample(mentions: pd.DataFrame, n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Stratified by aspect (equal share per aspect, capped by availability)."""
    rng = np.random.default_rng(seed)
    aspects = sorted(mentions["aspect"].unique())
    per = max(1, n // len(aspects))
    parts = []
    for a in aspects:
        sub = mentions[mentions["aspect"] == a]
        idx = rng.choice(len(sub), size=min(per, len(sub)), replace=False)
        parts.append(sub.iloc[idx])
    sample = pd.concat(parts).sample(frac=1.0, random_state=seed)
    out = sample[["mention_id", "aspect", "term", "clause", "sentence"]].copy()
    out["label"] = ""          # positive / negative / neutral
    out["aspect_correct"] = ""  # yes / no
    return out.reset_index(drop=True)
