import math

import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score, ndcg_score

from reviewrec.evaluation.metrics import (
    classification_report,
    cohen_kappa,
    judged_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

LABELS = {"a": 2, "b": 0, "c": 1, "d": 2}


def test_precision_recall():
    ranked = ["a", "b", "c", "x"]
    assert precision_at_k(ranked, LABELS, 2) == 0.5
    assert precision_at_k(ranked, LABELS, 4) == 0.5
    assert precision_at_k(ranked, LABELS, 4, threshold=2) == 0.25
    assert recall_at_k(ranked, LABELS, 4) == pytest.approx(2 / 3)
    assert math.isnan(recall_at_k(ranked, {"a": 0}, 4))


def test_reciprocal_rank_and_judged():
    assert reciprocal_rank(["b", "x", "c"], LABELS) == pytest.approx(1 / 3)
    assert reciprocal_rank(["b", "x"], LABELS) == 0.0
    assert judged_at_k(["a", "x", "b", "y"], LABELS, 4) == 0.5


def test_ndcg_matches_sklearn_exponential_gain():
    ranked = ["b", "c", "a", "d"]
    ours = ndcg_at_k(ranked, LABELS, 4)
    # sklearn uses linear gains; convert our exponential gain (2^g - 1) to compare.
    true = np.array([[2**LABELS[p] - 1 for p in ranked]])
    scores = np.array([[4, 3, 2, 1]])
    assert ours == pytest.approx(ndcg_score(true, scores, k=4))
    assert ndcg_at_k(["a", "d", "c", "b"], LABELS, 4) == pytest.approx(1.0)


def test_cohen_kappa_matches_sklearn():
    a = [0, 1, 2, 2, 1, 0, 2, 1]
    b = [0, 1, 1, 2, 1, 0, 2, 2]
    assert cohen_kappa(a, b) == pytest.approx(cohen_kappa_score(a, b))


def test_classification_report():
    rep = classification_report(["positive", "negative", "neutral", "positive"], ["positive", "negative", "positive", "positive"],
                                ["negative", "neutral", "positive"])
    assert rep["accuracy"] == 0.75
    assert rep["per_class"]["neutral"]["f1"] == 0.0
    assert rep["per_class"]["positive"]["recall"] == 1.0
