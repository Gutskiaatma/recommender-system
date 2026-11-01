"""TF-IDF + cosine similarity baseline.

Each product is one document: title + brand + the same review sentences the
semantic index uses. This keeps the comparison fair: both methods see exactly
the same text and differ only in how they represent it (sparse word counts vs
dense sentence embeddings).

Settings: unigrams+bigrams ("battery life"), sublinear TF (a word repeated
50 times is not 50x more important), English stop words removed, terms that
appear in fewer than 2 products or more than 90% of products dropped.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


def build_product_documents(products: pd.DataFrame, sentences: pd.DataFrame) -> list[str]:
    text_by_product = (
        sentences[sentences["in_index"]].groupby("product_id")["sentence"].apply(lambda s: " ".join(s)).to_dict()
    )
    return [
        f"{row.title} {row.brand} {text_by_product.get(row.product_id, '')}"
        for row in products[["product_id", "title", "brand"]].itertuples(index=False)
    ]


class TfidfIndex:
    def __init__(self, vectorizer: TfidfVectorizer, matrix: sparse.csr_matrix):
        self.vectorizer = vectorizer
        self.matrix = matrix

    @classmethod
    def fit(cls, documents: list[str], max_features: int = 50000) -> "TfidfIndex":
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
            max_df=0.9,
            max_features=max_features,
            dtype=np.float32,
        )
        matrix = vectorizer.fit_transform(documents).tocsr()
        return cls(vectorizer, matrix)

    def score(self, query: str) -> np.ndarray:
        q = self.vectorizer.transform([query])
        # Rows are L2-normalised by TfidfVectorizer, so the dot product is cosine similarity.
        return np.asarray((self.matrix @ q.T).toarray()).ravel().astype(np.float32)

    def top_terms(self, product_idx: int, n: int = 10) -> list[str]:
        row = self.matrix.getrow(product_idx)
        if row.nnz == 0:
            return []
        order = np.argsort(-row.data)[:n]
        vocab = self.vectorizer.get_feature_names_out()
        return [vocab[row.indices[i]] for i in order]

    def save(self, artifacts_dir: Path) -> None:
        joblib.dump(self.vectorizer, artifacts_dir / "tfidf_vectorizer.joblib")
        sparse.save_npz(artifacts_dir / "tfidf_matrix.npz", self.matrix)

    @classmethod
    def load(cls, artifacts_dir: Path) -> "TfidfIndex":
        return cls(
            joblib.load(artifacts_dir / "tfidf_vectorizer.joblib"),
            sparse.load_npz(artifacts_dir / "tfidf_matrix.npz").tocsr(),
        )
