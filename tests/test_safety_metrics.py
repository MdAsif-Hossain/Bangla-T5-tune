"""Unit tests: the safety metrics must catch the dangerous failure modes."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agri_mt.safety_metrics import (  # noqa: E402
    number_match,
    negation_faithful,
    term_accuracy,
    numbers_in,
)

GLOSS = [
    {"en": "Tricyclazole", "bn": "ট্রাইসাইক্লাজল", "en_aliases": ["tricyclazole"], "bn_aliases": []},
    {"en": "Rice", "bn": "ধান", "en_aliases": ["rice", "paddy"], "bn_aliases": ["চাল"]},
]


def test_bengali_digit_normalization():
    assert numbers_in("১.১২ লিটার ও ৫০ কেজি") == ["1.12", "50"]


def test_number_match_preserved():
    # Bengali-digit dose that survives -> perfect.
    score, d = number_match("Apply 1.12 liter per acre.", "একরে ১.১২ লিটার প্রয়োগ করুন।")
    assert score == 1.0 and d["missing"] == []


def test_number_match_dropped_dose():
    # Dose corrupted 1.12 -> 1.2 : must be flagged.
    score, d = number_match("Apply 1.12 liter.", "১.২ লিটার প্রয়োগ করুন।")
    assert score < 1.0 and "1.12" in d["missing"]


def test_number_match_none_when_no_numbers():
    assert number_match("Drain the field.", "জমির পানি বের করুন।")[0] is None


def test_negation_preserved():
    faithful, _ = negation_faithful("Do not drain the field.", "জমির পানি নিষ্কাশন করবেন না।")
    assert faithful is True


def test_negation_flip_detected():
    # The dangerous flip: "do not drain" -> "drain" (negation dropped).
    faithful, d = negation_faithful("Do not drain the field.", "জমির পানি নিষ্কাশন করুন।")
    assert faithful is False and d["src_neg"] and not d["hyp_neg"]


def test_negation_none_when_no_negation():
    assert negation_faithful("Drain the field.", "জমির পানি বের করুন।")[0] is None


def test_term_accuracy_correct():
    acc, hits = term_accuracy(
        "Apply Tricyclazole to control rice blast.",
        "ধানের ব্লাস্ট দমনে ট্রাইসাইক্লাজল প্রয়োগ করুন।",
        GLOSS,
    )
    assert acc == 1.0 and len(hits) == 2  # Tricyclazole + Rice both applicable & rendered


def test_term_accuracy_wrong_term():
    acc, hits = term_accuracy(
        "Apply Tricyclazole to rice.",
        "ধানে একটি ছত্রাকনাশক প্রয়োগ করুন।",  # 'Tricyclazole' not rendered
        GLOSS,
    )
    # Rice rendered (ধান), Tricyclazole missing -> 0.5
    assert acc == 0.5


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
