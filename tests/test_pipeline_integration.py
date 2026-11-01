"""Integration tests: run the real pipeline on the tiny synthetic dataset."""

from __future__ import annotations

import json

import pandas as pd


def test_data_quality_report(built_artifacts):
    s = built_artifacts["settings"]
    report = json.loads((s.processed_dir / "data_quality_report.json").read_text())
    assert report["removed_missing_title"] == 1
    assert report["removed_invalid_rating"] == 1
    assert report["removed_accessory_products"] == 1
    assert report["flagged_non_english"] >= 1
    assert report["removed_products_below_min_reviews"] == 1  # "Omega Tiny Phone"


def test_variants_are_merged_into_one_family(built_artifacts):
    s = built_artifacts["settings"]
    products = pd.read_parquet(s.processed_dir / "products.parquet")
    alpha = products[products["title"].str.startswith("Alpha")]
    assert len(alpha) == 1
    assert alpha["n_variants"].iat[0] == 2
    # Shared reviews are counted once, not twice.
    assert alpha["n_reviews"].iat[0] == 30
    assert set(products["brand"]) == {"Alpha", "Beta", "Gamma", "Delta"}


def test_artifacts_are_consistent(built_artifacts):
    s = built_artifacts["settings"]
    adir = s.artifacts_dir
    import numpy as np

    products = pd.read_parquet(adir / "products.parquet")
    title_emb = np.load(adir / "title_embeddings.npy")
    sent_idx = np.load(adir / "sentence_product_idx.npy")
    sentences = pd.read_parquet(adir / "sentences_index.parquet")
    mentions = pd.read_parquet(adir / "mentions.parquet")
    mention_emb = np.load(adir / "mention_embeddings.npy")
    assert title_emb.shape[0] == len(products)
    assert len(sent_idx) == len(sentences)
    assert (np.diff(sent_idx) >= 0).all(), "sentences must be grouped by product"
    assert len(mention_emb) == len(mentions)
    assert mentions["v"].between(-1, 1).all()
    assert products["quality"].between(0, 1).all()
