"""End-to-end offline pipeline: raw CSV -> cleaned tables -> NLP -> artifacts.

Each step reads the previous step's files from disk, so you can rerun one
step without repeating the expensive ones:

    prepare    load + clean + product families          -> data/processed
    aspects    sentences + aspect mentions               -> data/interim
    embed      title / sentence / clause embeddings      -> data/interim
    sentiment  aspect-level polarity per mention         -> data/interim
    index      TF-IDF, aggregates, final artifacts       -> artifacts/
"""

from __future__ import annotations

import json
import logging
import platform
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from reviewrec.aspects.aggregate import aggregate_product_aspects, bayesian_rating
from reviewrec.aspects.extractor import build_sentences, extract_mentions
from reviewrec.aspects.lexicon import get_lexicon
from reviewrec.aspects.sentiment import (
    TransformerABSAScorer,
    VaderScorer,
    WeakSupervisedScorer,
    make_weak_labels,
    polarity_labels,
)
from reviewrec.config import Settings, get_settings
from reviewrec.data.clean import clean_reviews
from reviewrec.data.load import load_raw
from reviewrec.data.products import build_products
from reviewrec.ranking.scoring import quality_score
from reviewrec.retrieval.semantic import make_embedder
from reviewrec.retrieval.tfidf import TfidfIndex, build_product_documents

log = logging.getLogger("reviewrec.pipeline")
STEPS = ("prepare", "aspects", "embed", "sentiment", "index")


def _write_json(path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")


def step_prepare(s: Settings, nrows: int | None = None) -> dict[str, Any]:
    raw = load_raw(s.path("paths.raw_csv"), nrows=nrows)
    log.info("Loaded %d raw rows", len(raw))
    clean, report = clean_reviews(raw, s.get("data"))
    products, reviews, product_report = build_products(clean, s.get("data"), seed=int(s.get("seed", 42)))
    report.update(product_report)
    out = s.processed_dir
    out.mkdir(parents=True, exist_ok=True)
    products.to_parquet(out / "products.parquet", index=False)
    reviews.to_parquet(out / "reviews.parquet", index=False)
    _write_json(out / "data_quality_report.json", report)
    log.info("Prepared %d products and %d reviews for NLP", len(products), len(reviews))
    return report


def step_aspects(s: Settings) -> dict[str, Any]:
    reviews = pd.read_parquet(s.processed_dir / "reviews.parquet")
    sentences = build_sentences(
        reviews,
        min_words=int(s.get("text.min_sentence_words", 3)),
        max_per_product=int(s.get("text.max_sentences_per_product", 400)),
        seed=int(s.get("seed", 42)),
    )
    mentions = extract_mentions(sentences, get_lexicon())
    s.interim_dir.mkdir(parents=True, exist_ok=True)
    sentences.to_parquet(s.interim_dir / "sentences.parquet", index=False)
    mentions.to_parquet(s.interim_dir / "mentions_raw.parquet", index=False)
    info = {
        "sentences": int(len(sentences)),
        "sentences_indexed": int(sentences["in_index"].sum()),
        "mentions": int(len(mentions)),
        "mentions_per_aspect": mentions["aspect"].value_counts().to_dict(),
    }
    log.info("Aspects: %s", info)
    return info


def step_embed(s: Settings) -> dict[str, Any]:
    products = pd.read_parquet(s.processed_dir / "products.parquet")
    sentences = pd.read_parquet(s.interim_dir / "sentences.parquet")
    mentions = pd.read_parquet(s.interim_dir / "mentions_raw.parquet")
    embedder = make_embedder(s.embedding_model, device=s.get("embeddings.device", "cpu"), batch_size=int(s.get("embeddings.batch_size", 128)))
    t0 = time.time()
    titles = (products["title"] + " " + products["brand"]).tolist()
    title_emb = embedder.encode(titles, show_progress=True)
    indexed = sentences[sentences["in_index"]]
    sent_emb = embedder.encode(indexed["sentence"].tolist(), show_progress=True).astype(np.float16)
    clause_emb = embedder.encode(mentions["clause"].tolist(), show_progress=True).astype(np.float16)
    d = s.interim_dir
    np.save(d / "title_embeddings.npy", title_emb)
    np.save(d / "sentence_embeddings.npy", sent_emb)
    np.save(d / "mention_embeddings.npy", clause_emb)
    info = {"model": s.embedding_model, "dim": int(title_emb.shape[1]), "seconds": round(time.time() - t0, 1),
            "titles": len(titles), "sentences": int(len(indexed)), "clauses": int(len(mentions))}
    log.info("Embeddings: %s", info)
    return info


def step_sentiment(s: Settings) -> dict[str, Any]:
    mentions = pd.read_parquet(s.interim_dir / "mentions_raw.parquet")
    backend = s.sentiment_backend
    vader = VaderScorer()
    mentions["vader_v"] = vader.score(mentions["clause"].tolist())
    info: dict[str, Any] = {"backend": backend}
    if backend == "weak_supervised":
        emb = np.load(s.interim_dir / "mention_embeddings.npy").astype(np.float32)
        labels = make_weak_labels(
            mentions["rating"], int(s.get("sentiment.positive_min_rating", 4)), int(s.get("sentiment.negative_max_rating", 2))
        )
        scorer = WeakSupervisedScorer.train(
            emb, labels, mentions["product_id"], int(s.get("sentiment.max_train_examples", 60000)), int(s.get("seed", 42))
        )
        mentions["v"] = scorer.score_embeddings(emb)
        s.artifacts_dir.mkdir(parents=True, exist_ok=True)
        scorer.save(s.artifacts_dir / "sentiment_model.joblib")
        info["training"] = scorer.metadata
    elif backend == "vader":
        mentions["v"] = mentions["vader_v"]
    elif backend == "transformer":
        scorer = TransformerABSAScorer(s.get("sentiment.transformer_model"))
        mentions["v"] = scorer.score(mentions["clause"].tolist(), mentions["aspect"].tolist())
    else:
        raise ValueError(f"Unknown sentiment backend: {backend}")
    band = float(s.get("sentiment.vader_neutral_band" if backend == "vader" else "sentiment.neutral_band", 0.3))
    mentions["label"] = polarity_labels(mentions["v"].to_numpy(), band)
    mentions.to_parquet(s.interim_dir / "mentions.parquet", index=False)
    info["label_distribution"] = mentions["label"].value_counts(normalize=True).round(4).to_dict()
    log.info("Sentiment: %s", info)
    return info


def step_index(s: Settings, extra_manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    products = pd.read_parquet(s.processed_dir / "products.parquet")
    sentences = pd.read_parquet(s.interim_dir / "sentences.parquet")
    mentions = pd.read_parquet(s.interim_dir / "mentions.parquet")
    adir = s.artifacts_dir
    adir.mkdir(parents=True, exist_ok=True)
    band = float(s.get("sentiment.neutral_band", 0.3)) if s.sentiment_backend != "vader" else float(s.get("sentiment.vader_neutral_band", 0.05))

    # Product-level quality signal.
    total_ratings = products["n_reviews"].sum()
    global_mean = float((products["mean_rating"] * products["n_reviews"]).sum() / total_ratings)
    products["bayes_rating"] = bayesian_rating(
        products["mean_rating"], products["n_reviews"], global_mean, float(s.get("aggregation.rating_prior_strength", 20))
    )
    products["quality"] = quality_score(products["bayes_rating"].to_numpy())
    products.to_parquet(adir / "products.parquet", index=False)

    # Semantic index files (sentences must stay grouped by product, in product order).
    row_of = {pid: i for i, pid in enumerate(products["product_id"])}
    indexed = sentences[sentences["in_index"]].copy()
    indexed["product_idx"] = indexed["product_id"].map(row_of)
    sent_emb = np.load(s.interim_dir / "sentence_embeddings.npy")
    order = np.argsort(indexed["product_idx"].to_numpy(), kind="stable")
    indexed = indexed.iloc[order].reset_index(drop=True)
    np.save(adir / "sentence_embeddings.npy", sent_emb[order])
    np.save(adir / "sentence_product_idx.npy", indexed["product_idx"].to_numpy(dtype=np.int64))
    indexed.drop(columns=["in_index"]).to_parquet(adir / "sentences_index.parquet", index=False)
    np.save(adir / "title_embeddings.npy", np.load(s.interim_dir / "title_embeddings.npy"))
    np.save(adir / "mention_embeddings.npy", np.load(s.interim_dir / "mention_embeddings.npy"))
    mentions.to_parquet(adir / "mentions.parquet", index=False)

    tfidf = TfidfIndex.fit(build_product_documents(products, sentences), int(s.get("retrieval.tfidf_max_features", 50000)))
    tfidf.save(adir)

    table, priors = aggregate_product_aspects(mentions, float(s.get("aggregation.aspect_prior_strength", 5)), band)
    table.to_parquet(adir / "product_aspects.parquet", index=False)
    _write_json(adir / "aspect_priors.json", priors)

    stats = build_stats(products, mentions, table, priors, s)
    _write_json(adir / "stats.json", stats)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "embedding_model": s.get("embeddings.model_name"),
        "sentiment_backend": s.sentiment_backend,
        "n_products": int(len(products)),
        "n_indexed_sentences": int(len(indexed)),
        "n_mentions": int(len(mentions)),
        "tfidf_vocabulary": int(len(tfidf.vectorizer.vocabulary_)),
        "python": platform.python_version(),
        **(extra_manifest or {}),
    }
    _write_json(adir / "manifest.json", manifest)
    log.info("Index built: %s", manifest)
    return manifest


def build_stats(products: pd.DataFrame, mentions: pd.DataFrame, table: pd.DataFrame, priors: dict[str, float], s: Settings) -> dict[str, Any]:
    stars = {str(k): int(products[f"stars_{k}"].sum()) for k in range(1, 6)}
    brands = products.groupby("brand").agg(products=("product_id", "count"), reviews=("n_reviews", "sum")).sort_values("reviews", ascending=False)
    lex = get_lexicon()
    aspects = []
    for a in sorted(priors):
        sub = mentions[mentions["aspect"] == a]
        aspects.append(
            {
                "aspect": a,
                "label": lex.labels.get(a, a),
                "mentions": int(len(sub)),
                "positive_share": round(float((sub["label"] == "positive").mean()), 4) if len(sub) else None,
                "negative_share": round(float((sub["label"] == "negative").mean()), 4) if len(sub) else None,
                "catalog_mean_score": round((priors[a] + 1) / 2, 4),
                "products_with_enough_evidence": int((table[table["aspect"] == a]["n_mentions"] >= int(s.get("aggregation.min_mentions_for_claim", 3))).sum()),
            }
        )
    price = products["price"].dropna()
    return {
        "n_products": int(len(products)),
        "n_reviews": int(products["n_reviews"].sum()),
        "n_reviews_nlp": int(products["n_text_reviews"].clip(upper=int(s.get("data.max_reviews_per_product", 250))).sum()),
        "n_brands": int(products["brand"].nunique()),
        "rating_distribution": stars,
        "mean_rating": round(float((products["mean_rating"] * products["n_reviews"]).sum() / products["n_reviews"].sum()), 4),
        "price": {
            "known_share": round(float(products["price"].notna().mean()), 4),
            "median": round(float(price.median()), 2) if len(price) else None,
            "histogram": _histogram(price, [0, 50, 100, 150, 200, 300, 400, 600, 800, 3000]),
        },
        "reviews_per_product": _histogram(products["n_reviews"], [15, 25, 50, 100, 250, 500, 1000, 100000]),
        "top_brands": [{"brand": b, **{k: int(v) for k, v in r.items()}} for b, r in brands.head(12).iterrows()],
        "aspects": aspects,
    }


def _histogram(values: pd.Series, edges: list[float]) -> list[dict[str, Any]]:
    counts, _ = np.histogram(values, bins=edges)
    out = []
    for lo, hi, c in zip(edges[:-1], edges[1:], counts):
        label = f"{int(lo)}+" if hi >= 3000 else f"{int(lo)}-{int(hi)}"
        out.append({"bucket": label, "count": int(c)})
    return out


def run(steps: list[str] | None = None, nrows: int | None = None, settings: Settings | None = None) -> dict[str, Any]:
    s = settings or get_settings()
    steps = steps or list(STEPS)
    summary: dict[str, Any] = {}
    for step in STEPS:
        if step not in steps:
            continue
        t0 = time.time()
        log.info("=== step: %s ===", step)
        if step == "prepare":
            summary[step] = step_prepare(s, nrows=nrows)
        elif step == "aspects":
            summary[step] = step_aspects(s)
        elif step == "embed":
            summary[step] = step_embed(s)
        elif step == "sentiment":
            summary[step] = step_sentiment(s)
        elif step == "index":
            summary[step] = step_index(s)
        summary[f"{step}_seconds"] = round(time.time() - t0, 1)
    _write_json(s.interim_dir / "pipeline_summary.json", summary)
    return summary
