"""Load all precomputed artifacts produced by ``scripts/run_pipeline.py``.

The API and the evaluation load this once; nothing here recomputes
embeddings or sentiment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from reviewrec.config import Settings, get_settings
from reviewrec.retrieval.semantic import SemanticIndex
from reviewrec.retrieval.tfidf import TfidfIndex

REQUIRED_FILES = [
    "products.parquet",
    "sentences_index.parquet",
    "mentions.parquet",
    "mention_embeddings.npy",
    "product_aspects.parquet",
    "aspect_priors.json",
    "title_embeddings.npy",
    "sentence_embeddings.npy",
    "sentence_product_idx.npy",
    "tfidf_vectorizer.joblib",
    "tfidf_matrix.npz",
    "manifest.json",
]


class ArtifactsMissingError(RuntimeError):
    pass


@dataclass
class ArtifactStore:
    products: pd.DataFrame
    sentences: pd.DataFrame
    mentions: pd.DataFrame
    mention_embeddings: np.ndarray
    product_aspects: pd.DataFrame
    aspect_priors: dict[str, float]
    tfidf: TfidfIndex
    semantic: SemanticIndex
    manifest: dict[str, Any]
    stats: dict[str, Any]
    aspect_matrices: dict[str, np.ndarray]
    aspect_names: list[str]
    product_row: dict[str, int]
    mention_rows_by_product: dict[str, np.ndarray]
    sentence_rows_by_product: dict[str, np.ndarray]

    @classmethod
    def load(cls, settings: Settings | None = None, artifacts_dir: Path | None = None) -> "ArtifactStore":
        settings = settings or get_settings()
        adir = artifacts_dir or settings.artifacts_dir
        missing = [f for f in REQUIRED_FILES if not (adir / f).exists()]
        if missing:
            raise ArtifactsMissingError(
                f"Artifacts missing in {adir}: {missing}. Run `python scripts/run_pipeline.py` first."
            )
        products = pd.read_parquet(adir / "products.parquet")
        sentences = pd.read_parquet(adir / "sentences_index.parquet")
        mentions = pd.read_parquet(adir / "mentions.parquet")
        product_aspects = pd.read_parquet(adir / "product_aspects.parquet")
        priors = json.loads((adir / "aspect_priors.json").read_text(encoding="utf-8"))
        manifest = json.loads((adir / "manifest.json").read_text(encoding="utf-8"))
        stats_path = adir / "stats.json"
        stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}

        semantic = SemanticIndex.load(
            adir, float(settings.get("retrieval.title_weight", 0.4)), int(settings.get("retrieval.evidence_top_k", 5))
        )
        tfidf = TfidfIndex.load(adir)

        product_row = {pid: i for i, pid in enumerate(products["product_id"])}
        aspect_names = sorted(priors.keys())
        n_p, n_a = len(products), len(aspect_names)
        a_idx = {a: j for j, a in enumerate(aspect_names)}
        mats = {name: np.zeros((n_p, n_a)) for name in ("score", "n_mentions", "n_positive", "n_negative", "n_neutral", "confidence")}
        for j, a in enumerate(aspect_names):
            mats["score"][:, j] = (priors[a] + 1) / 2  # prior for products without mentions
        rows = product_aspects["product_id"].map(product_row).to_numpy()
        cols = product_aspects["aspect"].map(a_idx).to_numpy()
        for name in mats:
            mats[name][rows, cols] = product_aspects[name].to_numpy()

        mention_rows = {pid: np.asarray(v) for pid, v in mentions.groupby("product_id").indices.items()}
        sentence_rows = {pid: np.asarray(v) for pid, v in sentences.groupby("product_id").indices.items()}
        return cls(
            products=products,
            sentences=sentences,
            mentions=mentions,
            mention_embeddings=np.load(adir / "mention_embeddings.npy", mmap_mode="r"),
            product_aspects=product_aspects,
            aspect_priors=priors,
            tfidf=tfidf,
            semantic=semantic,
            manifest=manifest,
            stats=stats,
            aspect_matrices=mats,
            aspect_names=aspect_names,
            product_row=product_row,
            mention_rows_by_product=mention_rows,
            sentence_rows_by_product=sentence_rows,
        )
