"""Ranking and classification metrics (implemented from scratch so they are
easy to explain in an interview; sklearn is only used to cross-check in tests).

Relevance labels are graded: 0 = not relevant, 1 = partially, 2 = highly.
For binary metrics (precision, recall, MRR) a product counts as relevant when
its grade >= ``threshold`` (default 1).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np


def precision_at_k(ranked: Sequence[str], labels: Mapping[str, int], k: int, threshold: int = 1) -> float:
    top = ranked[:k]
    if k == 0:
        return 0.0
    return sum(labels.get(pid, 0) >= threshold for pid in top) / k


def recall_at_k(ranked: Sequence[str], labels: Mapping[str, int], k: int, threshold: int = 1) -> float:
    """Recall relative to all *labelled* relevant items (pooled judgements)."""
    relevant = {pid for pid, g in labels.items() if g >= threshold}
    if not relevant:
        return float("nan")
    return len(relevant & set(ranked[:k])) / len(relevant)


def dcg(gains: Sequence[float]) -> float:
    return sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked: Sequence[str], labels: Mapping[str, int], k: int) -> float:
    gains = [labels.get(pid, 0) for pid in ranked[:k]]
    ideal = sorted(labels.values(), reverse=True)[:k]
    ideal_dcg = dcg(ideal)
    if ideal_dcg == 0:
        return float("nan")
    return dcg(gains) / ideal_dcg


def reciprocal_rank(ranked: Sequence[str], labels: Mapping[str, int], threshold: int = 1) -> float:
    for i, pid in enumerate(ranked):
        if labels.get(pid, 0) >= threshold:
            return 1.0 / (i + 1)
    return 0.0


def judged_at_k(ranked: Sequence[str], labels: Mapping[str, int], k: int) -> float:
    """Share of the top-k that has a human/assistant judgement (pool coverage)."""
    top = ranked[:k]
    return sum(pid in labels for pid in top) / max(len(top), 1)


def cohen_kappa(a: Sequence[int], b: Sequence[int]) -> float:
    a_arr, b_arr = np.asarray(a), np.asarray(b)
    if len(a_arr) == 0:
        return float("nan")
    cats = np.union1d(a_arr, b_arr)
    po = float(np.mean(a_arr == b_arr))
    pe = float(sum(np.mean(a_arr == c) * np.mean(b_arr == c) for c in cats))
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def classification_report(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    per_class = {}
    for c in labels:
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        per_class[c] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4), "support": int(np.sum(y_true == c))}
    return {
        "accuracy": round(float(np.mean(y_true == y_pred)), 4) if len(y_true) else float("nan"),
        "macro_f1": round(float(np.mean([v["f1"] for v in per_class.values()])), 4),
        "per_class": per_class,
        "n": int(len(y_true)),
    }
