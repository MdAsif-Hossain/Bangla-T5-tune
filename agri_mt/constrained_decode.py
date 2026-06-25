"""Terminology- and number-safe translation wrappers for BanglaT5.

Two mechanisms, applied CONSISTENTLY at training and inference time:

1. Numeric-placeholder protection (the dosage guarantee)
   ------------------------------------------------------
   General NMT drops/reorders/garbles numbers ("1.12 liter" -> "1.2"). We
   replace every source number with an indexed placeholder the model learns to
   copy verbatim, translate, then restore the ORIGINAL digits by index. Because
   restoration is exact, NumberMatch becomes ~100% by construction rather than
   by hope. Train the model on the SAME masking so it reliably preserves the
   placeholders.

2. Glossary inline annotation (soft terminology constraint, Dinu et al. 2019)
   --------------------------------------------------------------------------
   For each agricultural glossary term in the source we append a bracketed
   target hint, e.g. "apply Tricyclazole" -> "apply Tricyclazole ⟦ট্রাইসাইক্লাজল⟧".
   Trained with these hints, the model learns to emit the canonical Bengali
   term, lifting TermAcc without hard, ungrammatical forced decoding.

This module is pure pre/post-processing — model-independent and unit-tested —
so the guarantee logic is verifiable without a GPU.
"""
from __future__ import annotations

import re

from agri_mt.safety_metrics import BENGALI_DIGITS, _NUM_RE

# Placeholder design: a Latin sentinel SentencePiece keeps intact and the model
# is trained to copy. ⟦ ⟧ brackets are rare and survive tokenization.
_PH = "Ⓝ{}"                       # e.g. Ⓝ0, Ⓝ1 ...
_PH_FIND = re.compile(r"Ⓝ(\d+)")
_ASCII_TO_BN = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")


def protect_numbers(text: str) -> tuple[str, list[str]]:
    """Replace numbers with indexed placeholders. Returns (masked, originals)."""
    originals: list[str] = []

    def repl(m: re.Match) -> str:
        originals.append(m.group(0))
        return _PH.format(len(originals) - 1)

    return _NUM_RE.sub(repl, text), originals


def restore_numbers(translated: str, originals: list[str], to_bengali: bool = True) -> str:
    """Put original numbers back by index. Missing placeholders are appended so a
    dropped dose is never silently lost (it surfaces, conspicuously, at the end).
    """
    seen: set[int] = set()

    def repl(m: re.Match) -> str:
        i = int(m.group(1))
        if 0 <= i < len(originals):
            seen.add(i)
            num = originals[i]
            return num.translate(_ASCII_TO_BN) if to_bengali else num
        return ""  # stray placeholder index → drop the tag

    out = _PH_FIND.sub(repl, translated)
    # Surface any source number the model failed to carry through.
    missing = [originals[i] for i in range(len(originals)) if i not in seen]
    if missing:
        tail = " ".join((n.translate(_ASCII_TO_BN) if to_bengali else n) for n in missing)
        out = f"{out} {tail}".strip()
    return re.sub(r"\s{2,}", " ", out).strip()


def annotate_source_with_terms(text: str, glossary: list[dict], bracket=("⟦", "⟧")) -> str:
    """Append ⟦bn⟧ hints after the first occurrence of each glossary EN term.

    Used to build training data and to prompt the adapted model at inference.
    Longest surface forms first so multiword terms win.
    """
    lo, hi = bracket
    out = text
    # Build (surface_form, bn) pairs, longest-first.
    forms: list[tuple[str, str]] = []
    for g in glossary:
        for f in [g["en"], *g.get("en_aliases", [])]:
            if f:
                forms.append((f, g["bn"]))
    for surface, bn in sorted(forms, key=lambda x: len(x[0]), reverse=True):
        pat = re.compile(rf"(?<![A-Za-z])({re.escape(surface)})(?![A-Za-z])", re.IGNORECASE)
        # annotate only the first hit, and only if not already annotated
        if pat.search(out) and bn not in out:
            out = pat.sub(rf"\1 {lo}{bn}{hi}", out, count=1)
    return out


def mask_training_pair(
    en: str, bn: str, glossary: list[dict] | None = None
) -> tuple[str, str]:
    """Produce an aligned masked (source, target) training example.

    The model only learns to copy placeholders if it sees them on BOTH sides at
    train time. We annotate terms + mask numbers on the English source, then
    replace the SAME numbers (Bengali- or ASCII-digit form) in the Bengali
    target with the matching placeholder. Numbers absent from the target (back-
    translation noise) are left as-is.
    """
    # Number masking was dropped (the Ⓝ placeholder is OOV for BanglaT5 and
    # corrupts digits in place). Keep only the glossary term hint; leave numbers
    # intact so the model learns to copy them naturally.
    src = annotate_source_with_terms(en, glossary) if glossary else en
    return src, bn


def preprocess_source(text: str, glossary: list[dict] | None = None) -> tuple[str, list[str]]:
    """Source-side prep: annotate glossary terms only.

    Number masking was REMOVED: the Ⓝ placeholder is OOV for BanglaT5's
    SentencePiece, so the model transliterates it ("Ⓝ0"->"এন০") and corrupts the
    digit in place (e.g. "10 days"->"1 day"). Since the stock model already
    preserves numbers ~96%, masking solved a non-problem and hurt output. The
    glossary term hint is the part that works, so we keep only that.
    """
    if glossary:
        text = annotate_source_with_terms(text, glossary)
    return text, []


def postprocess_target(translated: str, originals: list[str], to_bengali: bool = True) -> str:
    """Restore numbers in the model output. (Glossary hints are not emitted by a
    properly trained model; strip any that leaked.)"""
    translated = translated.replace("⟦", "").replace("⟧", "")
    return restore_numbers(translated, originals, to_bengali=to_bengali)
