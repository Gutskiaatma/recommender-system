"""Review-aware scoring.

    S(p) = alpha * R(p) + beta * A(p) + gamma * Q(p),   alpha + beta + gamma = 1

R  relevance        semantic (or TF-IDF) similarity, min-max scaled inside the
                    candidate pool so it spans [0, 1] for every query.
A  aspect sentiment weighted mean of Bayesian-smoothed aspect scores, only for
                    the aspects the user asked for. Products with no mentions
                    of an aspect get the catalogue prior (neutral, not zero).
Q  quality          Bayesian-average star rating scaled from [1, 5] to [0, 1].

Double counting: A and Q both come from reviews and are positively correlated.
They are kept because they answer different questions - A: "are the aspects
*you* asked about praised?", Q: "are owners satisfied overall?" (which also
catches defects the lexicon has no word for). Q therefore gets the smallest
weight, and review *count* is never added as its own popularity term: it only
controls how strongly the smoothing trusts a product's own data.
If the query has no recognised aspect, beta is redistributed to alpha and gamma.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Weights:
    alpha: float = 0.55
    beta: float = 0.30
    gamma: float = 0.15

    def normalised(self, has_aspects: bool) -> "Weights":
        a, b, g = self.alpha, (self.beta if has_aspects else 0.0), self.gamma
        total = a + b + g
        if total <= 0:
            return Weights(1.0, 0.0, 0.0)
        return Weights(a / total, b / total, g / total)


def minmax(values: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    lo, hi = values.min(), values.max()
    if hi - lo < eps:
        return np.ones_like(values)
    return (values - lo) / (hi - lo)


def quality_score(bayes_rating: np.ndarray) -> np.ndarray:
    return np.clip((np.asarray(bayes_rating, dtype=np.float64) - 1.0) / 4.0, 0.0, 1.0)


def aspect_component(aspect_scores: np.ndarray, aspect_weights: np.ndarray) -> np.ndarray:
    """aspect_scores: [n_candidates, n_query_aspects] already in [0, 1]
    (prior-filled where a product has no mentions); aspect_weights: [n_query_aspects]."""
    if aspect_scores.size == 0 or aspect_weights.sum() <= 0:
        return np.zeros(aspect_scores.shape[0])
    return aspect_scores @ aspect_weights / aspect_weights.sum()


def combine(relevance: np.ndarray, aspect: np.ndarray, quality: np.ndarray, weights: Weights) -> np.ndarray:
    return weights.alpha * relevance + weights.beta * aspect + weights.gamma * quality
