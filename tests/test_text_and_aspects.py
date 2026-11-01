import pandas as pd

from reviewrec.aspects.extractor import build_sentences, extract_mentions
from reviewrec.aspects.lexicon import get_lexicon
from reviewrec.aspects.sentiment import VaderScorer, make_weak_labels, polarity_label
from reviewrec.data.clean import infer_brand, is_accessory_title, _accessory_pattern
from reviewrec.text.normalize import clean_text, fingerprint, is_probably_english
from reviewrec.text.sentences import clause_for_span, split_clauses, split_sentences


def test_clean_text():
    assert clean_text("  Great &amp; cheap<br/>phone!!!!!! http://x.com ") == "Great & cheap phone!!!"
    assert clean_text(None) == ""
    assert clean_text(float("nan")) == ""


def test_fingerprint_ignores_case_and_punctuation():
    assert fingerprint("Great phone!!") == fingerprint("great   PHONE")


def test_language_heuristic():
    assert is_probably_english("The battery is great and the camera is good for the price")
    assert not is_probably_english("Es un excelente telefono, la bateria dura mucho y la camara es muy buena")
    assert is_probably_english("Excellent!")


def test_sentence_split_keeps_decimals():
    s = split_sentences("Screen is 5.5 inches. Android 6.0 works well! Love it")
    assert s == ["Screen is 5.5 inches.", "Android 6.0 works well!", "Love it"]


def test_clause_isolates_contrast():
    sent = "The camera is great but the battery is terrible"
    assert [c[2] for c in split_clauses(sent)] == ["The camera is great", "the battery is terrible"]
    assert clause_for_span(sent, sent.index("battery")) == "the battery is terrible"
    # Too-short clause falls back to the full sentence.
    assert clause_for_span("Camera, wow", 0) == "Camera, wow"


def test_lexicon_longest_match_wins():
    lex = get_lexicon()
    found = [(m.aspect, m.term) for m in lex.find_in_review("I had dropped calls but battery life is fine")]
    assert ("connectivity", "dropped calls") in found
    assert ("battery", "battery life") in found
    assert all(a != "durability" for a, _ in found)


def test_mentions_one_per_aspect_per_sentence():
    reviews = pd.DataFrame(
        {"review_id": ["r_1"], "product_id": ["p"], "rating": [4],
         "review_text": ["The camera and the photos are great, but the battery drains fast. Screen ok."]}
    )
    sentences = build_sentences(reviews, min_words=2)
    mentions = extract_mentions(sentences, get_lexicon())
    first = mentions[mentions["sentence_id"] == "r_1_s0"]
    assert sorted(first["aspect"]) == ["battery", "camera"]
    battery = first[first["aspect"] == "battery"].iloc[0]
    assert "camera" not in battery["clause"]


def test_accessory_filter():
    pat = _accessory_pattern(["case", "screen protector", "charger"])
    assert is_accessory_title("Case for Samsung Galaxy S7", pat)
    assert is_accessory_title("Premium Screen Protector 2-pack", pat)
    assert not is_accessory_title("Samsung Galaxy S7 Unlocked Phone with free case", pat)
    assert infer_brand("BLU Studio X 5.5") == "BLU"


def test_sentiment_helpers():
    labels = make_weak_labels(pd.Series([1, 2, 3, 4, 5]))
    assert labels.tolist()[:2] == [0.0, 0.0] and pd.isna(labels.iloc[2]) and labels.tolist()[3:] == [1.0, 1.0]
    assert polarity_label(0.5, 0.3) == "positive" and polarity_label(-0.31, 0.3) == "negative" and polarity_label(0.1, 0.3) == "neutral"
    v = VaderScorer().score(["the battery is terrible", "the battery is great"])
    assert v[0] < 0 < v[1]
