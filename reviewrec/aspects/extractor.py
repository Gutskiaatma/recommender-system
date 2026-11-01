"""Turn reviews into sentences and aspect mentions.

Output tables
-------------
sentences: one row per review sentence
    sentence_id, review_id, product_id, rating, sentence, in_index
mentions: one row per (sentence, aspect) pair
    mention_id, sentence_id, review_id, product_id, aspect, term, clause, sentence, rating

``in_index`` marks the (capped, seeded) sample of sentences that is embedded
for semantic retrieval. Aspect mentions use *all* sentences of the selected
reviews, so sentiment statistics are not affected by the cap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from reviewrec.aspects.lexicon import AspectLexicon
from reviewrec.text.sentences import clause_for_span, split_sentences


def build_sentences(reviews: pd.DataFrame, min_words: int = 3, max_per_product: int = 400, seed: int = 42) -> pd.DataFrame:
    rows = []
    for review_id, product_id, rating, text in reviews[["review_id", "product_id", "rating", "review_text"]].itertuples(index=False):
        for i, sent in enumerate(split_sentences(text, min_words=min_words)):
            rows.append((f"{review_id}_s{i}", review_id, product_id, int(rating), sent))
    sentences = pd.DataFrame(rows, columns=["sentence_id", "review_id", "product_id", "rating", "sentence"])
    rng = np.random.default_rng(seed)
    sentences["_rand"] = rng.random(len(sentences))
    rank = sentences.groupby("product_id")["_rand"].rank(method="first")
    sentences["in_index"] = rank <= max_per_product
    sentences = sentences.drop(columns=["_rand"])
    # Keep sentences grouped by product: the semantic index relies on it.
    return sentences.sort_values(["product_id", "sentence_id"], kind="stable").reset_index(drop=True)


def extract_mentions(sentences: pd.DataFrame, lexicon: AspectLexicon) -> pd.DataFrame:
    rows = []
    for sentence_id, review_id, product_id, rating, sentence in sentences[
        ["sentence_id", "review_id", "product_id", "rating", "sentence"]
    ].itertuples(index=False):
        seen: set[str] = set()
        for match in lexicon.find_in_review(sentence):
            if match.aspect in seen:
                continue  # one mention per aspect per sentence
            seen.add(match.aspect)
            clause = clause_for_span(sentence, match.start)
            rows.append(
                (f"{sentence_id}_{match.aspect}", sentence_id, review_id, product_id, match.aspect, match.term, clause, sentence, rating)
            )
    return pd.DataFrame(
        rows,
        columns=["mention_id", "sentence_id", "review_id", "product_id", "aspect", "term", "clause", "sentence", "rating"],
    )
