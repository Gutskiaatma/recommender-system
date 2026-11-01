"""Shared test fixtures.

The integration fixtures build a *tiny synthetic* dataset (clearly fake,
written only for tests) and run the real pipeline on it with the offline
hashing embedder, so the whole suite runs in seconds without internet.
Real results come from the real dataset via scripts/run_pipeline.py.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]

SYNTHETIC_PRODUCTS = {
    "Alpha Phone X 64GB Unlocked Smartphone": {
        "brand": "Alpha",
        "price": 320.0,
        "good": ["The camera takes stunning photos even in low light.", "Selfies look sharp and the camera is excellent.",
                 "Great screen, bright and crisp display."],
        "bad": ["The battery drains in a few hours, terrible battery life.", "Battery is awful and charging takes forever."],
        "rating_good": 5,
        "rating_bad": 2,
        "n": 30,
    },
    "Beta Phone Y 32GB Unlocked Dual SIM": {
        "brand": "Beta",
        "price": 150.0,
        "good": ["Battery life is amazing, it lasts two full days.", "The battery is the best I have ever had on a phone.",
                 "Good value for the price."],
        "bad": ["The camera is blurry and photos look washed out.", "Pictures are grainy, the camera is poor."],
        "rating_good": 5,
        "rating_bad": 2,
        "n": 30,
    },
    "Gamma Pro 128GB Octa Core Unlocked Phone": {
        "brand": "Gamma",
        "price": 610.0,
        "good": ["Performance is incredibly smooth, games run without any lag.", "Gaming is fantastic and the processor is fast.",
                 "The camera is very good as well."],
        "bad": ["It is expensive and overpriced for what you get."],
        "rating_good": 5,
        "rating_bad": 3,
        "n": 30,
    },
    "Delta Lite 8GB Unlocked Cell Phone": {
        "brand": "Delta",
        "price": 89.0,
        "good": ["Very cheap and a great deal for the money.", "Affordable phone that works for calls."],
        "bad": ["The screen is dim and the display scratches easily.", "It lags and freezes when opening apps."],
        "rating_good": 4,
        "rating_bad": 1,
        "n": 30,
    },
}


def _synthetic_rows(seed: int = 7) -> list[dict]:
    rnd = random.Random(seed)
    rows = []
    for title, spec in SYNTHETIC_PRODUCTS.items():
        for i in range(spec["n"]):
            if i % 3 == 2:
                text = f"{rnd.choice(spec['bad'])} Review number {i} for this phone."
                rating = spec["rating_bad"]
            else:
                text = f"{rnd.choice(spec['good'])} I have used this phone for {i + 1} weeks."
                rating = spec["rating_good"]
            rows.append({"Product Name": title, "Brand Name": spec["brand"], "Price": spec["price"], "Rating": rating,
                         "Reviews": text, "Review Votes": i % 4})
    # Colour variant sharing the same reviews as Alpha -> must be merged into one product family.
    alpha = [r for r in rows if r["Product Name"].startswith("Alpha")]
    for r in alpha:
        rows.append({**r, "Product Name": "Alpha Phone X 64GB Unlocked Smartphone - Black"})
    # Accessory listing -> must be removed.
    for i in range(20):
        rows.append({"Product Name": "Case Cover for Alpha Phone X", "Brand Name": "CaseCo", "Price": 9.99, "Rating": 4,
                     "Reviews": f"Nice case number {i}, fits well.", "Review Votes": 0})
    # Product with too few reviews -> excluded (cold start).
    for i in range(3):
        rows.append({"Product Name": "Omega Tiny Phone", "Brand Name": "Omega", "Price": 50, "Rating": 5,
                     "Reviews": f"Perfect camera and battery {i}.", "Review Votes": 0})
    # Non-English, empty and invalid rows.
    rows.append({"Product Name": "Beta Phone Y 32GB Unlocked Dual SIM", "Brand Name": "Beta", "Price": 150, "Rating": 5,
                 "Reviews": "Es un excelente telefono, la bateria dura mucho y la camara es muy buena para el precio", "Review Votes": 0})
    rows.append({"Product Name": "Beta Phone Y 32GB Unlocked Dual SIM", "Brand Name": "Beta", "Price": 150, "Rating": 9,
                 "Reviews": "Invalid rating row", "Review Votes": 0})
    rows.append({"Product Name": None, "Brand Name": "Beta", "Price": 150, "Rating": 4, "Reviews": "No title", "Review Votes": 0})
    return rows


@pytest.fixture(scope="session")
def synthetic_project(tmp_path_factory) -> Path:
    base = tmp_path_factory.mktemp("rr_project")
    raw = base / "raw.csv"
    pd.DataFrame(_synthetic_rows()).to_csv(raw, index=False)
    settings = yaml.safe_load((ROOT / "config" / "settings.yaml").read_text(encoding="utf-8"))
    settings["paths"].update(
        {
            "raw_csv": str(raw),
            "interim_dir": str(base / "interim"),
            "processed_dir": str(base / "processed"),
            "artifacts_dir": str(base / "artifacts"),
            "aspects_file": str(ROOT / "config" / "aspects.yaml"),
        }
    )
    settings["data"]["min_reviews_per_product"] = 10
    settings["sentiment"]["max_train_examples"] = 10000
    settings["retrieval"]["candidate_pool"] = 10
    settings_file = base / "settings.yaml"
    settings_file.write_text(yaml.safe_dump(settings), encoding="utf-8")
    return settings_file


@pytest.fixture(scope="session")
def built_artifacts(synthetic_project: Path):
    old = {k: os.environ.get(k) for k in ("RR_SETTINGS_FILE", "RR_EMBEDDING_MODEL", "RR_ARTIFACTS_DIR", "RR_SENTIMENT_BACKEND")}
    os.environ["RR_SETTINGS_FILE"] = str(synthetic_project)
    os.environ["RR_EMBEDDING_MODEL"] = "hashing"
    os.environ.pop("RR_ARTIFACTS_DIR", None)
    os.environ.pop("RR_SENTIMENT_BACKEND", None)
    from reviewrec import config
    from reviewrec.aspects import lexicon

    config.get_settings.cache_clear()
    config.get_aspect_config.cache_clear()
    lexicon.get_lexicon.cache_clear()
    from reviewrec.pipeline import run

    summary = run()
    yield {"settings_file": synthetic_project, "summary": summary, "settings": config.get_settings()}
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    config.get_settings.cache_clear()
    config.get_aspect_config.cache_clear()
    lexicon.get_lexicon.cache_clear()


@pytest.fixture(scope="session")
def recommender(built_artifacts):
    from reviewrec.ranking.recommender import Recommender

    return Recommender.from_settings(built_artifacts["settings"])
