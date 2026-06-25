"""The numeric-protection guarantee must hold even when the 'translation'
reorders, drops, or garbles content."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agri_mt.constrained_decode import (  # noqa: E402
    protect_numbers,
    restore_numbers,
    annotate_source_with_terms,
    preprocess_source,
    postprocess_target,
    mask_training_pair,
)
from agri_mt.safety_metrics import number_match  # noqa: E402

GLOSS = [
    {"en": "Tricyclazole", "bn": "ট্রাইসাইক্লাজল", "en_aliases": [], "bn_aliases": []},
    {"en": "Rice", "bn": "ধান", "en_aliases": ["paddy"], "bn_aliases": []},
]


def test_protect_then_restore_roundtrip():
    masked, orig = protect_numbers("Apply 1.12 liter and 50 kg per acre.")
    assert orig == ["1.12", "50"]
    assert "1.12" not in masked and "Ⓝ0" in masked and "Ⓝ1" in masked


def test_guarantee_survives_reordering_and_drop():
    # Simulate a bad MT that REORDERS placeholders and DROPS surrounding words.
    masked, orig = protect_numbers("Mix 1.12 liter in 50 liter water.")
    fake_translation = f"পানিতে Ⓝ1 মিশিয়ে Ⓝ0 প্রয়োগ করুন"  # reordered placeholders
    out = restore_numbers(fake_translation, orig, to_bengali=True)
    # Both numbers present (as Bengali digits): NumberMatch must be perfect.
    score, _ = number_match("Mix 1.12 liter in 50 liter water.", out)
    assert score == 1.0


def test_dropped_placeholder_is_surfaced_not_lost():
    masked, orig = protect_numbers("Apply 0.6 g per liter.")
    fake = "প্রতি লিটারে প্রয়োগ করুন"  # model dropped the placeholder entirely
    out = restore_numbers(fake, orig, to_bengali=False)
    assert "0.6" in out  # surfaced at the tail rather than silently lost


def test_glossary_inline_annotation():
    ann = annotate_source_with_terms("Apply Tricyclazole to rice.", GLOSS)
    assert "⟦ট্রাইসাইক্লাজল⟧" in ann and "⟦ধান⟧" in ann


def test_full_pre_post_pipeline():
    # Final design: glossary term hints only; numbers are LEFT IN PLACE (the
    # placeholder approach corrupted them in BanglaT5).
    src = "Spray Tricyclazole at 1.12 liter. Do not apply 2 days before harvest."
    prepared, orig = preprocess_source(src, GLOSS)
    assert "⟦ট্রাইসাইক্লাজল⟧" in prepared and orig == []
    assert "1.12" in prepared and "2" in prepared      # numbers not masked
    model_out = "ট্রাইসাইক্লাজল ১.১২ লিটার স্প্রে করুন। সংগ্রহের ২ দিন আগে প্রয়োগ করবেন না।"
    final = postprocess_target(model_out, orig)
    score, _ = number_match(src, final)
    assert score == 1.0 and "⟦" not in final


def test_mask_training_pair_hints_only():
    # Training prep annotates the term hint and keeps numbers intact on both sides.
    en = "Apply Tricyclazole at 0.6 gram per liter."
    bn = "প্রতি লিটারে ০.৬ গ্রাম ট্রাইসাইক্লাজল প্রয়োগ করুন।"
    src, tgt = mask_training_pair(en, bn, GLOSS)
    assert "⟦ট্রাইসাইক্লাজল⟧" in src    # term hint added to source
    assert "0.6" in src                  # number left in place (not masked)
    assert tgt == bn                     # target unchanged


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
