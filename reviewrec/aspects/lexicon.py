"""Lexicon-based aspect detection.

Why a lexicon and not a trained aspect extractor?
  * There are no aspect annotations for this dataset; training an extractor
    (e.g. BIO tagging) would need thousands of hand-labelled sentences.
  * Phone aspects are a small, stable vocabulary ("battery", "screen", ...),
    so a curated lexicon reaches high precision and is fully transparent.
  * Its precision is measured on hand-labelled samples in the evaluation.

Overlapping matches are resolved globally by preferring the longest phrase,
so "dropped calls" counts as *connectivity* and not as *durability* ("dropped").
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from reviewrec.config import get_aspect_config


@dataclass(frozen=True)
class AspectMatch:
    aspect: str
    start: int
    end: int
    term: str


def _compile(terms: list[str]) -> re.Pattern[str]:
    alts = sorted({t.strip().lower() for t in terms if t and t.strip()}, key=len, reverse=True)
    escaped = [re.escape(t).replace(r"\ ", r"\s+") for t in alts]
    return re.compile(r"(?<![\w-])(" + "|".join(escaped) + r")(?![\w-])", flags=re.IGNORECASE)


class AspectLexicon:
    def __init__(self, config: dict[str, Any]):
        aspects = config["aspects"]
        self.aspects: list[str] = list(aspects.keys())
        self.labels = {a: aspects[a].get("label", a.title()) for a in self.aspects}
        self.descriptions = {a: aspects[a].get("description", a) for a in self.aspects}
        self.terms = {a: list(aspects[a].get("terms", [])) for a in self.aspects}
        self.query_terms = {a: list(aspects[a].get("query_terms", [])) for a in self.aspects}
        self._review_patterns = {a: _compile(self.terms[a]) for a in self.aspects}
        self._query_patterns = {a: _compile(self.terms[a] + self.query_terms[a]) for a in self.aspects}
        modifiers = config.get("query_modifiers", {})
        self.emphasis_words: list[str] = list(modifiers.get("emphasis", []))
        self.ignore_phrases: list[str] = list(modifiers.get("ignore", []))

    def _find(self, text: str, patterns: dict[str, re.Pattern[str]]) -> list[AspectMatch]:
        found: list[AspectMatch] = []
        for aspect, pattern in patterns.items():
            for m in pattern.finditer(text):
                found.append(AspectMatch(aspect, m.start(), m.end(), m.group(0).lower()))
        # Longest match wins when spans overlap across aspects.
        found.sort(key=lambda m: (-(m.end - m.start), m.start))
        kept: list[AspectMatch] = []
        for m in found:
            if all(m.end <= k.start or m.start >= k.end for k in kept):
                kept.append(m)
        return sorted(kept, key=lambda m: m.start)

    def find_in_review(self, text: str) -> list[AspectMatch]:
        return self._find(text, self._review_patterns)

    def find_in_query(self, text: str) -> list[AspectMatch]:
        return self._find(text, self._query_patterns)


@lru_cache(maxsize=1)
def get_lexicon() -> AspectLexicon:
    return AspectLexicon(get_aspect_config())
