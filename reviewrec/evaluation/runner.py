"""Evaluation runner used by scripts/evaluate.py and scripts/tune_weights.py."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from reviewrec.evaluation.metrics import (
    classification_report,
    cohen_kappa,
    judged_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from reviewrec.ranking.recommender import METHODS, Recommender
from reviewrec.ranking.scoring import Weights


def labels_by_query(labels: pd.DataFrame) -> dict[str, dict[str, int]]:
    """If several annotators labelled a pair, use the rounded mean grade."""
    g = labels.assign(grade=labels["grade"].astype(float)).groupby(["query_id", "product_id"])["grade"].mean().round()
    out: dict[str, dict[str, int]] = {}
    for (qid, pid), grade in g.items():
        out.setdefault(qid, {})[pid] = int(grade)
    return out


def query_metrics(ranked: list[str], labels: dict[str, int]) -> dict[str, float]:
    return {
        "P@5": precision_at_k(ranked, labels, 5),
        "P@10": precision_at_k(ranked, labels, 10),
        "R@10": recall_at_k(ranked, labels, 10),
        "NDCG@10": ndcg_at_k(ranked, labels, 10),
        "MRR": reciprocal_rank(ranked, labels),
        "HighlyRel@5": precision_at_k(ranked, labels, 5, threshold=2),
        "judged@10": judged_at_k(ranked, labels, 10),
    }


def evaluate_rankings(
    rec: Recommender,
    queries: pd.DataFrame,
    qrels: dict[str, dict[str, int]],
    configs: dict[str, tuple[str, Weights | None]],
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    rows = []
    for q in queries.itertuples(index=False):
        labels = qrels.get(q.query_id, {})
        if not labels:
            continue
        for name, (method, weights) in configs.items():
            t0 = time.perf_counter()
            ranked = rec.rank(q.query, method, top_k=10, weights=weights)
            ms = (time.perf_counter() - t0) * 1000
            rows.append({"query_id": q.query_id, "config": name, "latency_ms": ms, **query_metrics(ranked, labels)})
    per_query = pd.DataFrame(rows)
    summary = {
        name: {k: round(float(v), 4) for k, v in grp.drop(columns=["query_id", "config"]).mean(numeric_only=True).items()}
        for name, grp in per_query.groupby("config", sort=False)
    }
    return per_query, summary


def paired_bootstrap(per_query: pd.DataFrame, a: str, b: str, metric: str = "NDCG@10", n: int = 10000, seed: int = 42) -> dict[str, float]:
    """95% bootstrap CI of mean(metric[a] - metric[b]) over queries."""
    pivot = per_query.pivot(index="query_id", columns="config", values=metric).dropna(subset=[a, b])
    diff = (pivot[a] - pivot[b]).to_numpy()
    rng = np.random.default_rng(seed)
    boots = rng.choice(diff, size=(n, len(diff)), replace=True).mean(axis=1)
    return {
        "mean_diff": round(float(diff.mean()), 4),
        "ci95_low": round(float(np.percentile(boots, 2.5)), 4),
        "ci95_high": round(float(np.percentile(boots, 97.5)), 4),
        "share_bootstrap_gt_0": round(float((boots > 0).mean()), 4),
        "wins": int((diff > 0).sum()),
        "ties": int((diff == 0).sum()),
        "losses": int((diff < 0).sum()),
    }


def default_configs(rec: Recommender) -> dict[str, tuple[str, Weights | None]]:
    w = rec.weights
    configs: dict[str, tuple[str, Weights | None]] = {m: (m, None) for m in METHODS}
    configs["hybrid"] = ("hybrid", w)
    return configs


def ablation_configs(rec: Recommender) -> dict[str, tuple[str, Weights | None]]:
    w = rec.weights
    return {
        "hybrid_without_aspect_sentiment": ("hybrid", Weights(w.alpha, 0.0, w.gamma)),
        "hybrid_without_rating": ("hybrid", Weights(w.alpha, w.beta, 0.0)),
    }


def evaluate_absa(mentions: pd.DataFrame, labels: pd.DataFrame, neutral_band: float, vader_band: float) -> dict[str, Any]:
    merged = labels.merge(mentions[["mention_id", "v", "vader_v"]], on="mention_id", how="inner")
    lexicon = {
        "n": int(len(merged)),
        "precision": round(float((merged["aspect_correct"] == "yes").mean()), 4),
        "per_aspect": {a: round(float((g["aspect_correct"] == "yes").mean()), 4) for a, g in merged.groupby("aspect")},
    }
    valid = merged[merged["aspect_correct"] == "yes"]
    classes = ["negative", "neutral", "positive"]

    def to_label(v: np.ndarray, band: float) -> np.ndarray:
        return np.where(v > band, "positive", np.where(v < -band, "negative", "neutral"))

    backends = {
        "VADER (lexicon baseline)": classification_report(valid["label"], to_label(valid["vader_v"].to_numpy(), vader_band), classes),
        "Weak-supervised LR on embeddings": classification_report(valid["label"], to_label(valid["v"].to_numpy(), neutral_band), classes),
    }
    return {
        "n": int(len(valid)),
        "label_distribution": valid["label"].value_counts().to_dict(),
        "backends": backends,
        "lexicon": lexicon,
    }


def relevance_agreement(labels: pd.DataFrame) -> dict[str, Any] | None:
    """Cohen's kappa between the first two annotators on pairs both labelled."""
    annotators = labels["annotator"].dropna().unique().tolist()
    if len(annotators) < 2:
        return None
    a, b = annotators[:2]
    la = labels[labels["annotator"] == a].set_index(["query_id", "product_id"])["grade"].astype(int)
    lb = labels[labels["annotator"] == b].set_index(["query_id", "product_id"])["grade"].astype(int)
    common = la.index.intersection(lb.index)
    if len(common) == 0:
        return None
    return {
        "annotators": [a, b],
        "n_pairs": int(len(common)),
        "cohen_kappa_3_grades": round(cohen_kappa(la[common].tolist(), lb[common].tolist()), 4),
        "cohen_kappa_binary": round(cohen_kappa((la[common] >= 1).astype(int).tolist(), (lb[common] >= 1).astype(int).tolist()), 4),
        "exact_agreement": round(float((la[common] == lb[common]).mean()), 4),
    }


def signal_correlation(rec: Recommender, min_mentions: int = 10) -> dict[str, float]:
    """Pearson correlation between each aspect score and the quality signal Q,
    over products with enough mentions. Used to discuss double counting."""
    out = {}
    mats = rec.store.aspect_matrices
    for j, a in enumerate(rec.store.aspect_names):
        mask = mats["n_mentions"][:, j] >= min_mentions
        if mask.sum() >= 10:
            out[a] = round(float(np.corrcoef(mats["score"][mask, j], rec.quality[mask])[0, 1]), 4)
    return out
