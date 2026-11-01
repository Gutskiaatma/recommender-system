"""Raw dataset loading.

The loader converts a source-specific file into a *standard review schema* so
that everything downstream is dataset-agnostic. To add another dataset
(e.g. Amazon Reviews 2023), write another ``load_*`` function returning the
same columns and register it in ``LOADERS``.

Standard review schema (one row per review):
    source_row     int    row number in the raw file (traceability)
    product_title  str    product-level attribute
    brand          str    product-level attribute (may be empty)
    price          float  product-level attribute in USD (may be NaN)
    rating         float  review-level, 1..5 stars
    review_text    str    review-level
    review_votes   float  review-level helpful votes (may be NaN)
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd

STANDARD_COLUMNS = [
    "source_row",
    "product_title",
    "brand",
    "price",
    "rating",
    "review_text",
    "review_votes",
]

# Column names in the Kaggle "Amazon Reviews: Unlocked Mobile Phones" CSV
# (PromptCloud). Matching is case/space-insensitive to tolerate re-saved files.
_UNLOCKED_COLUMN_MAP = {
    "productname": "product_title",
    "brandname": "brand",
    "price": "price",
    "rating": "rating",
    "reviews": "review_text",
    "reviewvotes": "review_votes",
}


def _canon(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def load_unlocked_phones_csv(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. Download 'Amazon_Unlocked_Mobile.csv' from "
            "https://www.kaggle.com/datasets/PromptCloudHQ/amazon-reviews-unlocked-mobile-phones "
            "and place it in data/raw/."
        )
    df = pd.read_csv(path, nrows=nrows, encoding="utf-8", encoding_errors="replace", low_memory=False)
    rename = {}
    for col in df.columns:
        key = _canon(col)
        if key in _UNLOCKED_COLUMN_MAP:
            rename[col] = _UNLOCKED_COLUMN_MAP[key]
    df = df.rename(columns=rename)
    missing = {"product_title", "rating", "review_text"} - set(df.columns)
    if missing:
        raise ValueError(f"Unexpected CSV format, missing columns: {sorted(missing)}. Found: {list(df.columns)}")
    for optional in ("brand", "price", "review_votes"):
        if optional not in df.columns:
            df[optional] = pd.NA
    df["source_row"] = range(len(df))
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["review_votes"] = pd.to_numeric(df["review_votes"], errors="coerce")
    for col in ("product_title", "brand", "review_text"):
        df[col] = df[col].astype("string")
    return df[STANDARD_COLUMNS]


LOADERS: dict[str, Callable[..., pd.DataFrame]] = {
    "unlocked_phones": load_unlocked_phones_csv,
}


def load_raw(path: str | Path, source: str = "unlocked_phones", nrows: int | None = None) -> pd.DataFrame:
    if source not in LOADERS:
        raise ValueError(f"Unknown source '{source}'. Available: {sorted(LOADERS)}")
    return LOADERS[source](path, nrows=nrows)
