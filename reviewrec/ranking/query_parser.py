"""Parse a free-text preference query into structured signals.

"I want a Samsung phone with an excellent camera and long battery life under $400,
 don't care about storage"
  -> aspects   {camera: 1.5, battery: 1.0}
  -> ignored   [storage]
  -> brands    [Samsung]
  -> price_max 400

The full query text is still used for semantic/TF-IDF relevance; this parser
only adds the structure needed for aspect-aware ranking and hard filters.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from reviewrec.aspects.lexicon import AspectLexicon

_NOT_A_PRICE = r"(?!\s*(?:gb|mb|tb|mp|mah|inch|inches|in\b|\"|hours?|hrs?|days?|%|g\b|k\b|core|years?))"
_NUM = r"\$?\s*(\d[\d,]*(?:\.\d+)?)(?![\d,])"
_PRICE_MAX = re.compile(
    r"\b(?:under|below|less than|cheaper than|within|max(?:imum)?|up to|upto|no more than|budget of|at most)\s*"
    + _NUM + _NOT_A_PRICE,
    re.IGNORECASE,
)
_PRICE_MIN = re.compile(r"\b(?:over|above|more than|at least|min(?:imum)?|starting at)\s*" + _NUM + _NOT_A_PRICE, re.IGNORECASE)
_PRICE_RANGE = re.compile(r"\bbetween\s*" + _NUM + r"\s*(?:and|-|to)\s*" + _NUM + _NOT_A_PRICE, re.IGNORECASE)
_DOLLAR_RANGE = re.compile(r"\$\s*(\d[\d,]*)\s*(?:-|to)\s*\$?\s*(\d[\d,]*)", re.IGNORECASE)
_SEGMENT_SPLIT = re.compile(r"[,;.!?]|\band\b|\bbut\b|\bwith\b|\balso\b|\bplus\b", re.IGNORECASE)
_NEGATORS = re.compile(r"\b(?:no|not|except|avoid|without|exclude|excluding|other than)\s+$", re.IGNORECASE)

BRAND_ALIASES = {
    "iphone": "Apple",
    "galaxy": "Samsung",
    "moto": "Motorola",
    "xperia": "Sony",
    "lumia": "Nokia",
    "pixel": "Google",
    "nexus": "Google",
}


@dataclass
class ParsedQuery:
    text: str
    aspects: dict[str, float] = field(default_factory=dict)
    matched_terms: dict[str, list[str]] = field(default_factory=dict)
    ignored_aspects: list[str] = field(default_factory=list)
    brands: list[str] = field(default_factory=list)
    excluded_brands: list[str] = field(default_factory=list)
    price_min: float | None = None
    price_max: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _to_float(s: str) -> float:
    return float(s.replace(",", ""))


def _segments(text: str) -> list[tuple[int, int]]:
    spans, start = [], 0
    for m in _SEGMENT_SPLIT.finditer(text):
        spans.append((start, m.start()))
        start = m.end()
    spans.append((start, len(text)))
    return spans


def parse_prices(text: str) -> tuple[float | None, float | None]:
    lo = hi = None
    if m := _PRICE_RANGE.search(text):
        lo, hi = sorted((_to_float(m.group(1)), _to_float(m.group(2))))
        return lo, hi
    if m := _DOLLAR_RANGE.search(text):
        lo, hi = sorted((_to_float(m.group(1)), _to_float(m.group(2))))
        return lo, hi
    if m := _PRICE_MAX.search(text):
        hi = _to_float(m.group(1))
    if m := _PRICE_MIN.search(text):
        lo = _to_float(m.group(1))
    return lo, hi


def parse_brands(text: str, known_brands: list[str]) -> tuple[list[str], list[str]]:
    lowered = text.lower()
    wanted, excluded = [], []
    candidates: dict[str, str] = {b.lower(): b for b in known_brands if len(b) >= 2}
    for alias, brand in BRAND_ALIASES.items():
        if brand.lower() in candidates:
            candidates.setdefault(alias, candidates[brand.lower()])
    for token, brand in candidates.items():
        for m in re.finditer(r"(?<![\w-])" + re.escape(token) + r"(?![\w-])", lowered):
            target = excluded if _NEGATORS.search(lowered[: m.start()]) else wanted
            if brand not in target:
                target.append(brand)
    wanted = [b for b in wanted if b not in excluded]
    return wanted, excluded


def parse_query(
    text: str,
    lexicon: AspectLexicon,
    known_brands: list[str] | None = None,
    default_weight: float = 1.0,
    emphasis_weight: float = 1.5,
) -> ParsedQuery:
    parsed = ParsedQuery(text=text.strip())
    lowered = text.lower()
    segments = _segments(lowered)
    emphasis_re = re.compile(
        r"(?<![\w-])(" + "|".join(re.escape(w) for w in sorted(lexicon.emphasis_words, key=len, reverse=True)) + r")(?![\w-])"
    ) if lexicon.emphasis_words else None

    for match in lexicon.find_in_query(text):
        seg_start, seg_end = next(((s, e) for s, e in segments if s <= match.start < e), (0, len(lowered)))
        # Ignore phrases ("don't care about the camera") come just before the term.
        before = lowered[max(0, match.start - 40) : match.start]
        if any(p in before for p in lexicon.ignore_phrases):
            if match.aspect not in parsed.ignored_aspects:
                parsed.ignored_aspects.append(match.aspect)
            continue
        weight = default_weight
        if emphasis_re and emphasis_re.search(lowered[seg_start:seg_end]):
            weight = emphasis_weight
        parsed.aspects[match.aspect] = max(parsed.aspects.get(match.aspect, 0.0), weight)
        parsed.matched_terms.setdefault(match.aspect, [])
        if match.term not in parsed.matched_terms[match.aspect]:
            parsed.matched_terms[match.aspect].append(match.term)

    for aspect in parsed.ignored_aspects:
        parsed.aspects.pop(aspect, None)
        parsed.matched_terms.pop(aspect, None)
    parsed.price_min, parsed.price_max = parse_prices(text)
    if known_brands:
        parsed.brands, parsed.excluded_brands = parse_brands(text, known_brands)
    return parsed
