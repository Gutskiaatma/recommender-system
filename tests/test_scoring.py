import numpy as np
import pandas as pd
import pytest

from reviewrec.aspects.aggregate import aggregate_product_aspects, bayesian_rating
from reviewrec.ranking.scoring import Weights, aspect_component, combine, minmax, quality_score


def test_minmax_scales_and_handles_constant_input():
    assert minmax(np.array([2.0, 4.0, 3.0])).tolist() == [0.0, 1.0, 0.5]
    assert minmax(np.array([5.0, 5.0])).tolist() == [1.0, 1.0]


def test_weights_redistribute_when_no_aspects():
    w = Weights(0.55, 0.30, 0.15).normalised(has_aspects=False)
    assert w.beta == 0
    assert w.alpha + w.gamma == pytest.approx(1.0)
    assert w.alpha / w.gamma == pytest.approx(0.55 / 0.15)


def test_combine_is_weighted_sum():
    s = combine(np.array([1.0]), np.array([0.5]), np.array([0.0]), Weights(0.5, 0.3, 0.2))
    assert s[0] == pytest.approx(0.65)


def test_aspect_component_weighted_mean():
    scores = np.array([[1.0, 0.0], [0.5, 0.5]])
    out = aspect_component(scores, np.array([1.5, 1.0]))
    assert out.tolist() == pytest.approx([0.6, 0.5])


def test_bayesian_rating_shrinks_small_samples():
    r = bayesian_rating(np.array([5.0, 5.0]), np.array([1, 1000]), global_mean=4.0, m=20)
    assert r[0] == pytest.approx((5 + 80) / 21)
    assert r[1] > 4.95
    assert quality_score(np.array([1.0, 5.0, 3.0])).tolist() == [0.0, 1.0, 0.5]


def test_aspect_smoothing_one_review_is_not_definitive():
    mentions = pd.DataFrame(
        {
            "product_id": ["p1"] + ["p2"] * 50 + ["p3"] * 50,
            "aspect": ["camera"] * 101,
            "v": [1.0] + [0.9] * 50 + [-0.2] * 50,
        }
    )
    table, priors = aggregate_product_aspects(mentions, prior_strength=5, neutral_band=0.3)
    t = table.set_index("product_id")
    # One perfect mention ranks below 50 consistently good ones.
    assert t.loc["p1", "score"] < t.loc["p2", "score"]
    assert t.loc["p1", "confidence"] == pytest.approx(1 / 6)
    assert t.loc["p2", "n_positive"] == 50
    assert t.loc["p3", "n_neutral"] == 50
    assert 0 <= t["score"].min() and t["score"].max() <= 1
