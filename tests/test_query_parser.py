import pytest

from reviewrec.aspects.lexicon import get_lexicon
from reviewrec.ranking.query_parser import parse_prices, parse_query

BRANDS = ["Samsung", "Apple", "Motorola", "Blu"]


def parse(q):
    return parse_query(q, get_lexicon(), BRANDS)


def test_example_query_from_brief():
    p = parse("I want a smartphone with an excellent camera, long battery life, and good performance for gaming.")
    assert p.aspects == {"camera": 1.5, "battery": 1.0, "performance": 1.0}
    assert p.price_max is None and p.brands == []


def test_ignore_phrase_and_brand_and_price():
    p = parse("Samsung phone under $300 with a big screen, don't care about the camera")
    assert "camera" not in p.aspects and p.ignored_aspects == ["camera"]
    assert p.aspects == {"display": 1.0}
    assert p.brands == ["Samsung"]
    assert p.price_max == 300


@pytest.mark.parametrize(
    "text,expected",
    [
        ("under $250", (None, 250.0)),
        ("less than 1,200 dollars", (None, 1200.0)),
        ("between 100 and 300", (100.0, 300.0)),
        ("$150 - $400 range", (150.0, 400.0)),
        ("at least 64gb storage", (None, None)),
        ("over $500", (500.0, None)),
        ("5.5 inch screen under 6 hours", (None, None)),
    ],
)
def test_price_parsing(text, expected):
    assert parse_prices(text) == expected


def test_brand_aliases_and_exclusion():
    assert parse("an iPhone with good battery").brands == ["Apple"]
    p = parse("good camera phone, not samsung")
    assert p.excluded_brands == ["Samsung"] and p.brands == []


def test_query_only_terms_map_to_aspects():
    assert "performance" in parse("phone for a gamer").aspects
    assert "camera" in parse("phone for photography").aspects
