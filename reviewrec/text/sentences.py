"""Sentence and clause splitting.

We use a rule-based splitter instead of spaCy/NLTK: review text is informal
("love it!!! battery lasts 2 days...camera meh"), the rules are fast, have no
model downloads, and are easy to unit-test. Clause splitting on contrastive
words matters for aspect sentiment: in "camera is great but battery is awful"
each aspect should only see its own clause.
"""

from __future__ import annotations

import re

# Split after ., !, ? (one or more) or ellipses when followed by whitespace,
# and on line breaks. Avoid splitting decimals like "5.5 inch" or "Android 6.0".
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])(?<!\d\.)\s+|\n+|(?<=\.\.\.)(?=\S)")
_CLAUSE_BOUNDARY = re.compile(
    r"\s*(?:[;]|,?\s+\b(?:but|however|although|though|except|whereas|yet|while|otherwise)\b)\s*",
    flags=re.IGNORECASE,
)


def split_sentences(text: str, min_words: int = 1) -> list[str]:
    if not text:
        return []
    parts = (p.strip(" -*•") for p in _SENTENCE_BOUNDARY.split(text))
    return [p for p in parts if p and len(p.split()) >= min_words]


def split_clauses(sentence: str) -> list[tuple[int, int, str]]:
    """Return clauses as (start, end, text) character spans of the sentence."""
    clauses: list[tuple[int, int, str]] = []
    start = 0
    for match in _CLAUSE_BOUNDARY.finditer(sentence):
        if match.start() > start:
            clauses.append((start, match.start(), sentence[start:match.start()]))
        start = match.end()
    if start < len(sentence):
        clauses.append((start, len(sentence), sentence[start:]))
    return clauses or [(0, len(sentence), sentence)]


def clause_for_span(sentence: str, span_start: int, min_words: int = 3) -> str:
    """Clause containing a character position; falls back to the whole sentence
    when the clause is too short to carry an opinion on its own."""
    for start, end, text in split_clauses(sentence):
        if start <= span_start < end:
            text = text.strip()
            return text if len(text.split()) >= min_words else sentence.strip()
    return sentence.strip()
