"""Aggregate mention-level sentiment into product-level aspect scores.

Bayesian smoothing (shrinkage toward the aspect's catalogue mean mu_a):

    s~(p, a) = (sum of v over the n mentions + k * mu_a) / (n + k)
    score(p, a) = (s~ + 1) / 2                      -> [0, 1]
    confidence(p, a) = n / (n + k)                   -> [0, 1)

With k = 5, one glowing mention moves a product only 1/6 of the way from the
catalogue average, while 50 consistent mentions move it ~91% of the way.
This prevents a product with a single review from ranking first.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def aspect_priors(mentions: pd.DataFrame) -> dict[str, float]:
    return mentions.groupby("aspect")["v"].mean().to_dict()


def aggregate_product_aspects(
    mentions: pd.DataFrame,
    prior_strength: float = 5.0,
    neutral_band: float = 0.3,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """mentions needs columns product_id, aspect, v. Returns (table, priors)."""
    priors = aspect_priors(mentions)
    m = mentions.copy()
    m["is_pos"] = m["v"] > neutral_band
    m["is_neg"] = m["v"] < -neutral_band
    g = m.groupby(["product_id", "aspect"])
    table = pd.DataFrame(
        {
            "n_mentions": g.size(),
            "n_positive": g["is_pos"].sum().astype(int),
            "n_negative": g["is_neg"].sum().astype(int),
            "sum_v": g["v"].sum(),
            "mean_v": g["v"].mean(),
        }
    ).reset_index()
    table["n_neutral"] = table["n_mentions"] - table["n_positive"] - table["n_negative"]
    mu = table["aspect"].map(priors)
    k = float(prior_strength)
    table["smoothed_v"] = (table["sum_v"] + k * mu) / (table["n_mentions"] + k)
    table["score"] = (table["smoothed_v"] + 1) / 2
    table["confidence"] = table["n_mentions"] / (table["n_mentions"] + k)
    table = table.drop(columns=["sum_v"])
    return table, {a: float(v) for a, v in priors.items()}


def prior_score(priors: dict[str, float], aspect: str) -> float:
    """Score of a product with zero mentions of an aspect (= catalogue mean)."""
    return (priors.get(aspect, 0.0) + 1) / 2


def bayesian_rating(mean_rating: np.ndarray, n_ratings: np.ndarray, global_mean: float, m: float) -> np.ndarray:
    """IMDb-style weighted rating: shrinks low-count products toward the mean."""
    n = np.asarray(n_ratings, dtype=float)
    return (n * np.asarray(mean_rating, dtype=float) + m * global_mean) / (n + m)
