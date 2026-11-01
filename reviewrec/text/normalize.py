"""Text normalisation helpers shared by the pipeline and the API.

Design note: transformer embedders and VADER both use casing, punctuation and
emoticons, so we keep the *display* text close to the original and only do
light cleaning. Lower-casing / stop-word removal is done inside the TF-IDF
vectoriser, which is the only model that benefits from it.
"""

from __future__ import annotations

import hashlib
import html
import re
import unicodedata

_WS = re.compile(r"\s+")
_HTML_TAG = re.compile(r"<[^>]+>")
_URL = re.compile(r"https?://\S+|www\.\S+")
_REPEAT_PUNCT = re.compile(r"([!?.]){4,}")
_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")

# Very common English function words. Used for a cheap language check:
# the dataset contains some Spanish reviews that our English models handle badly.
_ENGLISH_HINTS = frozenset(
    "the a an and or but is are was were it this that i my me you to of in on for with "
    "not have has had be very so its it's phone just do does did".split()
)


def clean_text(text: object) -> str:
    """Light cleaning that keeps the review readable for display and for models."""
    if text is None:
        return ""
    s = str(text)
    if s.lower() == "nan":
        return ""
    s = html.unescape(s)
    s = unicodedata.normalize("NFKC", s)
    s = _HTML_TAG.sub(" ", s)
    s = _URL.sub(" ", s)
    s = "".join(ch for ch in s if ch.isprintable() or ch in "\n\t")
    s = _REPEAT_PUNCT.sub(r"\1\1\1", s)
    return _WS.sub(" ", s).strip()


def normalize_for_matching(text: str) -> str:
    """Aggressive normalisation for duplicate detection only."""
    s = text.lower()
    s = _NON_ALNUM.sub(" ", s)
    return _WS.sub(" ", s).strip()


def fingerprint(text: str) -> str:
    """Stable hash of a review's normalised text (used to find duplicates)."""
    return hashlib.sha1(normalize_for_matching(text).encode("utf-8")).hexdigest()[:16]


def word_count(text: str) -> int:
    return len(text.split())


def english_ratio(text: str) -> float:
    tokens = normalize_for_matching(text).split()
    if not tokens:
        return 0.0
    return sum(t in _ENGLISH_HINTS for t in tokens) / len(tokens)


def is_probably_english(text: str, min_ratio: float = 0.12) -> bool:
    """Heuristic language check.

    Very short texts ("Excellent!") carry no signal, so they pass as long as
    they are ASCII. Longer texts must contain enough English function words.
    """
    tokens = text.split()
    if len(tokens) < 6:
        return all(ord(c) < 128 for c in text)
    return english_ratio(text) >= min_ratio


def stable_id(text: str, prefix: str = "") -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}{digest}"
