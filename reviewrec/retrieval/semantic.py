"""Semantic retrieval with Sentence Transformers.

Relevance of a product to a query combines two cosine similarities:

* title similarity     - query vs the product's title + brand (product-level text)
* evidence similarity  - mean of the top-k most similar *review sentences*
                         of that product (review-level text)

    R_sem(p) = w * title_sim(p) + (1 - w) * mean(top_k sentence sims of p)

Why the review part: titles in this dataset rarely say anything about camera
or battery quality, but reviews do. Using the *top-k mean* instead of the max
rewards products with several matching sentences and reduces the influence of
one lucky sentence. Note that this is a *topical* signal: "battery dies in 3
hours" is very similar to "long battery life". Polarity is handled separately
by aspect sentiment in the review-aware ranker.

All product and sentence embeddings are computed once by the pipeline and
loaded from disk; only the query is embedded at request time.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np


class Embedder:
    """Thin wrapper around SentenceTransformer with lazy loading."""

    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 128):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # heavy import, done lazily

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        emb = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return emb.astype(np.float32)

    @lru_cache(maxsize=512)
    def encode_query(self, query: str) -> np.ndarray:
        return self.encode([query])[0]


class HashingEmbedder:
    """Deterministic bag-of-words hashing embedder.

    Not a semantic model. It exists so the automated tests run offline and
    fast without downloading a transformer. Select it with
    ``RR_EMBEDDING_MODEL=hashing``.
    """

    def __init__(self, dim: int = 256):
        self.model_name = "hashing"
        self.dim = dim

    def encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        import re
        import zlib

        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in re.findall(r"[a-z0-9]+", text.lower()):
                out[i, zlib.crc32(tok.encode()) % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-9)

    def encode_query(self, query: str) -> np.ndarray:
        return self.encode([query])[0]


def make_embedder(model_name: str, device: str = "cpu", batch_size: int = 128):
    if model_name == "hashing":
        return HashingEmbedder()
    return Embedder(model_name, device=device, batch_size=batch_size)


class SemanticIndex:
    """Holds precomputed, L2-normalised embeddings.

    sentence_embeddings are sorted by product so per-product scores can be
    computed with a padded matrix instead of a slow Python loop.
    """

    def __init__(
        self,
        title_embeddings: np.ndarray,
        sentence_embeddings: np.ndarray,
        sentence_product_idx: np.ndarray,
        title_weight: float = 0.4,
        top_k: int = 5,
    ):
        self.title_embeddings = title_embeddings.astype(np.float32)
        self.sentence_embeddings = sentence_embeddings  # float16 on disk/in memory to save RAM
        self.sentence_product_idx = sentence_product_idx.astype(np.int64)
        self.n_products = title_embeddings.shape[0]
        self.title_weight = title_weight
        self.top_k = top_k
        counts = np.bincount(self.sentence_product_idx, minlength=self.n_products)
        self._max_sent = int(counts.max()) if len(counts) else 0
        starts = np.zeros(self.n_products, dtype=np.int64)
        starts[1:] = np.cumsum(counts)[:-1]
        self._col = np.arange(len(self.sentence_product_idx)) - starts[self.sentence_product_idx]
        self._counts = counts

    def sentence_similarities(self, query_emb: np.ndarray) -> np.ndarray:
        sims = np.empty(len(self.sentence_embeddings), dtype=np.float32)
        q = query_emb.astype(np.float32)
        chunk = 100_000
        for i in range(0, len(sims), chunk):
            sims[i : i + chunk] = self.sentence_embeddings[i : i + chunk].astype(np.float32) @ q
        return sims

    def evidence_scores(self, sentence_sims: np.ndarray) -> np.ndarray:
        """Mean of the top-k sentence similarities for every product."""
        padded = np.full((self.n_products, max(self._max_sent, 1)), -1.0, dtype=np.float32)
        padded[self.sentence_product_idx, self._col] = sentence_sims
        k = min(self.top_k, padded.shape[1])
        top = -np.partition(-padded, k - 1, axis=1)[:, :k]
        valid = np.minimum(self._counts, k)
        top = np.where(np.arange(k)[None, :] < valid[:, None], top, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            scores = np.where(valid > 0, top.sum(axis=1) / np.maximum(valid, 1), 0.0)
        return scores.astype(np.float32)

    def score(self, query_emb: np.ndarray) -> dict[str, np.ndarray]:
        title_sim = self.title_embeddings @ query_emb.astype(np.float32)
        sentence_sims = self.sentence_similarities(query_emb)
        evidence = self.evidence_scores(sentence_sims)
        relevance = self.title_weight * title_sim + (1 - self.title_weight) * evidence
        return {"relevance": relevance, "title_sim": title_sim, "evidence_sim": evidence, "sentence_sims": sentence_sims}

    @classmethod
    def load(cls, artifacts_dir: Path, title_weight: float, top_k: int) -> "SemanticIndex":
        return cls(
            np.load(artifacts_dir / "title_embeddings.npy"),
            np.load(artifacts_dir / "sentence_embeddings.npy", mmap_mode="r"),
            np.load(artifacts_dir / "sentence_product_idx.npy"),
            title_weight=title_weight,
            top_k=top_k,
        )
