"""Product table construction.

Key data-quality issue in Amazon data: colour/storage *variants* of the same
phone ("iPhone 6, Gold, 64GB" and "iPhone 6, Silver, 64GB") are listed as
separate titles but share the same review pool. Treating them as different
products would (a) show near-identical recommendations and (b) count each
review several times. We merge titles whose review sets overlap strongly into
one *product family* (similar to Amazon's parent ASIN) using union-find.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from reviewrec.text.normalize import stable_id


class _UnionFind:
    def __init__(self, items: list[str]):
        self.parent = {i: i for i in items}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def assign_families(df: pd.DataFrame, overlap_threshold: float = 0.5, min_shared: int = 5) -> pd.Series:
    """Map every product_title to a family key.

    Two titles are merged when they share at least ``min_shared`` substantial
    reviews and the shared reviews make up at least ``overlap_threshold`` of
    the smaller title's reviews.
    """
    substantial = df[df["n_words"] >= 8]
    fp_sets = substantial.groupby("product_title")["review_fp"].apply(set).to_dict()
    titles = df["product_title"].unique().tolist()
    uf = _UnionFind(titles)

    # Inverted index: fingerprint -> titles containing it; only compare titles
    # that share at least one review (avoids O(n^2) over all titles).
    index: dict[str, list[str]] = defaultdict(list)
    for title, fps in fp_sets.items():
        for fp in fps:
            index[fp].append(title)
    shared_counts: dict[tuple[str, str], int] = defaultdict(int)
    for owners in index.values():
        if 1 < len(owners) <= 50:
            owners = sorted(owners)
            for i in range(len(owners)):
                for j in range(i + 1, len(owners)):
                    shared_counts[(owners[i], owners[j])] += 1
    for (a, b), shared in shared_counts.items():
        smaller = min(len(fp_sets[a]), len(fp_sets[b]))
        if shared >= min_shared and shared / smaller >= overlap_threshold:
            uf.union(a, b)
    return df["product_title"].map(uf.find)


def build_products(df: pd.DataFrame, cfg: dict[str, Any], seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return (products, reviews, report).

    * products: one row per product family with product-level attributes and
      rating statistics computed from **all** of its de-duplicated reviews.
    * reviews: review-level rows used for NLP, capped per product.
    """
    report: dict[str, Any] = {}
    df = df.copy()
    df["family_key"] = assign_families(
        df, float(cfg.get("family_overlap_threshold", 0.5)), int(cfg.get("family_min_shared", 5))
    )

    # Representative title = the variant with the most reviews.
    title_sizes = df.groupby(["family_key", "product_title"]).size().rename("n").reset_index()
    rep = title_sizes.sort_values(["family_key", "n"], ascending=[True, False]).drop_duplicates("family_key")
    rep_title = dict(zip(rep["family_key"], rep["product_title"]))
    variants = title_sizes.groupby("family_key")["product_title"].apply(list).to_dict()
    df["product_id"] = df["family_key"].map(lambda k: stable_id(rep_title[k], prefix="p_"))

    before = len(df)
    empty = df["review_text"].eq("")
    dup_in_family = df.duplicated(subset=["product_id", "review_fp"], keep="first") & ~empty
    df = df[~dup_in_family].copy()
    report["titles_before_merge"] = int(len(title_sizes))
    report["product_families"] = int(df["product_id"].nunique())
    report["removed_cross_variant_duplicates"] = int(before - len(df))

    min_reviews = int(cfg.get("min_reviews_per_product", 15))
    counts = df.groupby("product_id").size()
    keep_ids = counts[counts >= min_reviews].index
    report["removed_products_below_min_reviews"] = int((counts < min_reviews).sum())
    report["removed_reviews_of_small_products"] = int(counts[counts < min_reviews].sum())
    df = df[df["product_id"].isin(keep_ids)].copy()

    grouped = df.groupby("product_id")
    rating_counts = (
        df.pivot_table(index="product_id", columns="rating", values="source_row", aggfunc="count", fill_value=0)
        .reindex(columns=[1, 2, 3, 4, 5], fill_value=0)
    )
    products = pd.DataFrame(
        {
            "family_key": grouped["family_key"].first(),
            "brand": grouped["brand"].agg(lambda s: s.mode().iat[0] if not s.mode().empty else ""),
            "price": grouped["price"].median(),
            "n_reviews": grouped.size(),
            "n_text_reviews": grouped["has_text"].sum().astype(int),
            "mean_rating": grouped["rating"].mean(),
        }
    )
    products["title"] = products["family_key"].map(rep_title)
    products["variant_titles"] = products["family_key"].map(variants)
    products["n_variants"] = products["variant_titles"].map(len)
    for star in range(1, 6):
        products[f"stars_{star}"] = rating_counts[star].reindex(products.index).fillna(0).astype(int)
    products = products.reset_index().drop(columns=["family_key"])
    products = products.sort_values("n_reviews", ascending=False).reset_index(drop=True)

    # Review-level table for NLP: only usable text, capped per product.
    # Prefer reviews with helpful votes (more informative), break ties randomly
    # with a fixed seed so the pipeline is reproducible.
    max_per_product = int(cfg.get("max_reviews_per_product", 250))
    text_reviews = df[df["has_text"]].copy()
    rng = np.random.default_rng(seed)
    text_reviews["_rand"] = rng.random(len(text_reviews))
    text_reviews["review_votes"] = text_reviews["review_votes"].fillna(0)
    text_reviews = (
        text_reviews.sort_values(["product_id", "review_votes", "_rand"], ascending=[True, False, True])
        .groupby("product_id", group_keys=False)
        .head(max_per_product)
        .drop(columns=["_rand"])
    )
    text_reviews["review_id"] = [f"r_{i}" for i in text_reviews["source_row"]]
    reviews = text_reviews[
        ["review_id", "product_id", "source_row", "product_title", "rating", "review_text", "review_votes", "n_words"]
    ].reset_index(drop=True)
    report["products_final"] = int(len(products))
    report["reviews_for_rating_stats"] = int(products["n_reviews"].sum())
    report["reviews_for_nlp"] = int(len(reviews))
    return products, reviews, report
