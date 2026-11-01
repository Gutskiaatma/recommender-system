"""Aspect-level sentiment.

Each aspect mention has a *clause* (the part of the sentence around the aspect
term). We map every clause to a polarity value v in [-1, 1].

Three interchangeable backends (selected in settings.yaml -> sentiment.backend):

1. ``vader``  - VADER lexicon/rule model applied to the clause. No training,
   instant, handles negation and intensifiers ("not good", "very bad") but
   misses domain wording ("battery dies by noon").

2. ``weak_supervised`` (default) - logistic regression on Sentence-Transformer
   embeddings of the clause, trained with *weak labels*: clauses from 4-5 star
   reviews are labelled positive, from 1-2 star reviews negative, 3-star
   reviews are skipped. No manual labels are needed, it learns domain phrases,
   and it runs on CPU in seconds. Weak labels are noisy (a 1-star review can
   still praise the camera), so we measure accuracy on hand-labelled clauses
   in the evaluation instead of trusting validation accuracy on weak labels.

3. ``transformer`` (optional) - a pretrained aspect-based sentiment model
   (yangheng/deberta-v3-base-absa-v1.1) that classifies (clause, aspect)
   pairs. Strongest in principle, but ~180M parameters, slow on CPU and needs
   a Hugging Face download.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


def polarity_label(v: float, neutral_band: float) -> str:
    if v > neutral_band:
        return "positive"
    if v < -neutral_band:
        return "negative"
    return "neutral"


def polarity_labels(values: np.ndarray, neutral_band: float) -> np.ndarray:
    return np.where(values > neutral_band, "positive", np.where(values < -neutral_band, "negative", "neutral"))


class VaderScorer:
    name = "vader"

    def __init__(self):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        self._analyzer = SentimentIntensityAnalyzer()

    def score(self, clauses: list[str]) -> np.ndarray:
        return np.array([self._analyzer.polarity_scores(c)["compound"] for c in clauses], dtype=np.float32)


def make_weak_labels(ratings: pd.Series, positive_min: int = 4, negative_max: int = 2) -> pd.Series:
    """1 = positive, 0 = negative, NaN = not used for training (3 stars)."""
    labels = pd.Series(np.nan, index=ratings.index)
    labels[ratings >= positive_min] = 1.0
    labels[ratings <= negative_max] = 0.0
    return labels


@dataclass
class WeakSupervisedScorer:
    model: Any
    metadata: dict[str, Any]
    name: str = "weak_supervised"

    @classmethod
    def train(
        cls,
        embeddings: np.ndarray,
        labels: pd.Series,
        groups: pd.Series,
        max_examples: int = 60000,
        seed: int = 42,
    ) -> "WeakSupervisedScorer":
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        from sklearn.model_selection import GroupShuffleSplit

        mask = labels.notna().to_numpy()
        X, y, g = embeddings[mask], labels[mask].astype(int).to_numpy(), groups[mask].to_numpy()
        rng = np.random.default_rng(seed)
        if len(y) > max_examples:
            idx = rng.choice(len(y), size=max_examples, replace=False)
            X, y, g = X[idx], y[idx], g[idx]

        # Split by product so validation measures generalisation to unseen phones.
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_idx, val_idx = next(splitter.split(X, y, g))
        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
        clf.fit(X[train_idx], y[train_idx])
        val_prob = clf.predict_proba(X[val_idx])[:, 1]
        val_pred = (val_prob >= 0.5).astype(int)
        metadata = {
            "train_examples": int(len(train_idx)),
            "val_examples": int(len(val_idx)),
            "positive_share": float(y.mean()),
            "val_accuracy_vs_weak_labels": float(accuracy_score(y[val_idx], val_pred)),
            "val_macro_f1_vs_weak_labels": float(f1_score(y[val_idx], val_pred, average="macro")),
            "val_roc_auc_vs_weak_labels": float(roc_auc_score(y[val_idx], val_prob)),
            "note": "Validation is against noisy star-rating labels; see evaluation/absa for hand-labelled accuracy.",
        }
        # Refit on all available weak labels for the final model.
        clf_full = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
        clf_full.fit(X, y)
        return cls(model=clf_full, metadata=metadata)

    def score_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        p = self.model.predict_proba(embeddings)[:, 1]
        return (2 * p - 1).astype(np.float32)

    def save(self, path: Path) -> None:
        joblib.dump({"model": self.model, "metadata": self.metadata}, path)

    @classmethod
    def load(cls, path: Path) -> "WeakSupervisedScorer":
        obj = joblib.load(path)
        return cls(model=obj["model"], metadata=obj["metadata"])


class TransformerABSAScorer:
    """Optional pretrained ABSA model (needs internet access to Hugging Face once).

    The model receives the clause as text and the aspect label as text_pair and
    returns Negative / Neutral / Positive probabilities. v = P(pos) - P(neg).
    """

    name = "transformer"

    def __init__(self, model_name: str, batch_size: int = 32, device: int = -1):
        from transformers import pipeline

        self.model_name = model_name
        self._pipe = pipeline("text-classification", model=model_name, top_k=None, device=device)
        self.batch_size = batch_size

    def score(self, clauses: list[str], aspects: list[str]) -> np.ndarray:
        inputs = [{"text": c, "text_pair": a} for c, a in zip(clauses, aspects)]
        values = np.zeros(len(inputs), dtype=np.float32)
        for start in range(0, len(inputs), self.batch_size):
            outputs = self._pipe(inputs[start : start + self.batch_size], truncation=True)
            for j, scores in enumerate(outputs):
                probs = {d["label"].lower(): d["score"] for d in scores}
                values[start + j] = probs.get("positive", 0.0) - probs.get("negative", 0.0)
        return values
