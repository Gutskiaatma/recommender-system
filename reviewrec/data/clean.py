"""Review-level cleaning and data-quality checks.

Every step records how many rows it removed or flagged, so the pipeline
produces a data-quality report (``data/processed/data_quality_report.json``)
that you can quote in the README and in interviews.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from reviewrec.text.normalize import clean_text, fingerprint, is_probably_english, word_count

_PHONE_HINTS = re.compile(
    r"\b(phone|smartphone|cell|unlocked|gsm|cdma|4g|lte|3g|dual sim|galaxy|iphone|blackberry|"
    r"lumia|moto|nexus|pixel|xperia|android)\b",
    flags=re.IGNORECASE,
)


def _accessory_pattern(keywords: list[str]) -> re.Pattern[str]:
    escaped = sorted((re.escape(k) for k in keywords), key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(escaped) + r")\b", flags=re.IGNORECASE)


def is_accessory_title(title: str, accessory_re: re.Pattern[str]) -> bool:
    """A title is an accessory when it names an accessory and nothing phone-like,
    or when it *starts* with an accessory word ("Case for Galaxy S7")."""
    if not accessory_re.search(title):
        return False
    first_words = " ".join(title.lower().split()[:3])
    if accessory_re.search(first_words):
        return True
    return not _PHONE_HINTS.search(title)


def infer_brand(title: str) -> str:
    first = title.split()[0] if title.split() else ""
    return re.sub(r"[^A-Za-z0-9-]", "", first)


def clean_reviews(raw: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean the standard review table.

    Returns the cleaned frame and a report with counts for every step.
    Rows are only *removed* when they cannot be used at all; reviews that are
    too short or not English are *flagged* (``has_text=False``) because their
    star rating is still valid evidence for the overall quality signal.
    """
    report: dict[str, Any] = {"raw_rows": int(len(raw))}
    df = raw.copy()

    df["product_title"] = df["product_title"].map(clean_text)
    df["brand"] = df["brand"].map(clean_text)
    df["review_text"] = df["review_text"].map(clean_text)

    missing_title = df["product_title"].eq("")
    bad_rating = ~df["rating"].between(1, 5)
    report["removed_missing_title"] = int(missing_title.sum())
    report["removed_invalid_rating"] = int((bad_rating & ~missing_title).sum())
    df = df[~missing_title & ~bad_rating].copy()
    df["rating"] = df["rating"].round().astype(int)

    no_brand = df["brand"].eq("")
    df.loc[no_brand, "brand"] = df.loc[no_brand, "product_title"].map(infer_brand)
    report["brand_inferred_from_title"] = int(no_brand.sum())
    # Normalise brand casing so "SAMSUNG" and "Samsung" are one brand.
    df["brand"] = df["brand"].str.strip().str.title()

    max_price = float(cfg.get("max_price_usd", 2500))
    bad_price = df["price"].notna() & ((df["price"] <= 0) | (df["price"] > max_price))
    report["price_set_to_missing_outlier"] = int(bad_price.sum())
    df.loc[bad_price, "price"] = np.nan
    report["price_missing_rows"] = int(df["price"].isna().sum())

    accessory_re = _accessory_pattern(cfg.get("accessory_keywords", []))
    titles = pd.Series(df["product_title"].unique())
    accessory_titles = set(titles[titles.map(lambda t: is_accessory_title(t, accessory_re))])
    is_acc = df["product_title"].isin(accessory_titles)
    report["removed_accessory_rows"] = int(is_acc.sum())
    report["removed_accessory_products"] = len(accessory_titles)
    df = df[~is_acc].copy()

    df["review_fp"] = df["review_text"].map(fingerprint)
    empty_text = df["review_text"].eq("")
    dup_mask = df.duplicated(subset=["product_title", "review_fp", "rating"], keep="first") & ~empty_text
    report["removed_exact_duplicates"] = int(dup_mask.sum())
    df = df[~dup_mask].copy()

    min_words = int(cfg.get("min_review_words", 3))
    min_en = float(cfg.get("min_english_ratio", 0.12))
    df["n_words"] = df["review_text"].map(word_count)
    long_enough = df["n_words"] >= min_words
    english = df["review_text"].map(lambda t: is_probably_english(t, min_en))
    df["has_text"] = long_enough & english
    report["flagged_too_short"] = int((~long_enough).sum())
    report["flagged_non_english"] = int((long_enough & ~english).sum())
    report["clean_rows"] = int(len(df))
    report["clean_rows_with_text"] = int(df["has_text"].sum())
    return df.reset_index(drop=True), report
