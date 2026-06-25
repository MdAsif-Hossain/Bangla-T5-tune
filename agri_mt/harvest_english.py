"""Harvest clean English agricultural sentences from the English source PDFs.

Two uses:
  * forward-distillation source (EN->BN via NLLB) for AgriEnBn;
  * the source side of the gold test set (a balanced sample, with a safety
    subset that contains dosage numbers and/or negations).

Reads AgriBot's English corpus (data/pdfs) with PyMuPDF, segments sentences,
keeps English + agricultural + reasonable-length ones, dedups, and tags safety.

    python -m agri_mt.harvest_english --gold 200
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import fitz

from agri_mt.agri_keywords import BROAD_AGRI_KEYWORDS_EN

EN_PDF_DIR = Path(r"F:/Projects/Agri_bot/data/pdfs")
HERE = Path(__file__).resolve().parent.parent
CORPUS_DIR = HERE / "data" / "corpus"
GOLD_DIR = HERE / "data" / "gold"

_KW = re.compile(
    r"(?<![a-z])(" + "|".join(sorted((re.escape(w) for w in BROAD_AGRI_KEYWORDS_EN), key=len, reverse=True)) + r")(?![a-z])",
    re.IGNORECASE,
)
_NEG = re.compile(r"\b(not|never|no|without|n't|do not|should not|must not)\b", re.I)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def _clean(text: str) -> str:
    text = re.sub(r"-\n", "", text)          # de-hyphenate line breaks
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


_ARTIFACT = re.compile(r"NARRATIVE|Figure\s*\d|Table\s*\d|Goto|et al|www\.|http|\.\.\.|\b[IVX]{2,}\b")
_CAPS_RUN = re.compile(r"(?:\b[A-Z]{3,}\b[ -]+){2,}")  # 2+ ALLCAPS words = header
# Unsalvageable: formulas, captions, copyright/boilerplate, and Myanmar-specific
# economics/geography that are noise for a Bangladesh advisory paper.
_REJECT = re.compile(
    r"©|=| X 100|Form-\d|Required citation|disclaimer|WebstaurantStore|"
    r"Courtesy|\(Fig\b|World Rice Conference|"
    r"USD|megatonne|MoALI|per capita|per annum|purchasing power|foreign exchange|"
    r"Ayeyarwady|Yesagyo|Magway|Pyidaungsu|Nay Pyi Taw"
)
_ABBR_END = re.compile(r"\b(No|pv|cv|var|sp|spp|Fig|al|Co|Ltd|Inc)\.$")
# Leading section labels to strip; trailing section numbers ("2.5.3.1.") to drop.
_LABEL = re.compile(
    r"^(Causal organism|Disease Cycle|Disease common name|Disease Management|"
    r"Control measures|Chemical control|Cultural control|Symptoms?( and signs)?|"
    r"Signs?|Survival|Transmission|Inoculum|Favou?rable environment)\b[:\s]*",
    re.I,
)
_SECNUM = re.compile(r"\s+\d+(?:\.\d+)*\.?$")
_CAPTION_CUT = re.compile(r"\s*\((?:Courtesy|Fig)\b.*$", re.I)

# Strict filtering (REJECT list, abbrev-truncation, sentence cleaning) is the
# improved behaviour. ``--legacy`` turns it off to reproduce the exact v1 source
# the gold set was first reviewed against (needed to realign that review).
_STRICT = True


def _clean_sentence(s: str) -> str:
    s = _CAPTION_CUT.sub("", s).strip()       # cut "... (Courtesy T." / "(Fig 3)"
    s = re.sub(r"\s*©.*$", "", s).strip()      # cut trailing copyright runs
    s = _SECNUM.sub("", s).strip()             # cut trailing "2.5.3.1."
    s = _LABEL.sub("", s).strip()              # cut leading "Disease Cycle ..."
    return s


def _good(s: str) -> bool:
    w = s.split()
    if not (6 <= len(w) <= 38):
        return False
    if len(re.findall(r"[A-Za-z]", s)) < 0.6 * len(s.replace(" ", "")):
        return False                          # mostly letters (drop tables/refs)
    if re.search(r"[ঀ-৿]", s):
        return False
    if not _KW.search(s):
        return False
    if sum(c.isdigit() for c in s) > 0.3 * len(s):
        return False                          # drop number-dump table rows
    if "•" in s or s.count("- ") >= 2:        # bullet dumps
        return False
    if _ARTIFACT.search(s) or _CAPS_RUN.search(s):
        return False
    if _STRICT and (_REJECT.search(s) or _ABBR_END.search(s)):
        return False
    if not s[0].isupper() or s[-1] not in ".!?":      # well-formed sentence only
        return False
    return True


def harvest(gold_n: int, legacy: bool = False) -> dict:
    global _STRICT
    _STRICT = not legacy
    seen: set[str] = set()
    sentences: list[str] = []
    for pdf in sorted(EN_PDF_DIR.glob("*.pdf")):
        doc = fitz.open(str(pdf))
        text = _clean(" ".join(p.get_text() for p in doc))
        doc.close()
        for s in _sentences(text):
            if _STRICT:
                s = _clean_sentence(s)
            key = s.lower()
            if key not in seen and _good(s):
                seen.add(key)
                sentences.append(s)

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    (CORPUS_DIR / "english_agri_sentences.txt").write_text("\n".join(sentences), encoding="utf-8")

    # Build a balanced gold source: prioritise safety sentences (numbers/negation).
    safety = [s for s in sentences if re.search(r"\d", s) or _NEG.search(s)]
    plain = [s for s in sentences if s not in set(safety)]
    n_safety = min(len(safety), gold_n // 2)
    gold = safety[:n_safety] + plain[: gold_n - n_safety]
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    (GOLD_DIR / "source_en.txt").write_text("\n".join(gold), encoding="utf-8")

    return {"total_harvested": len(sentences), "safety_available": len(safety),
            "gold_selected": len(gold), "gold_safety": n_safety}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=int, default=200)
    ap.add_argument("--legacy", action="store_true",
                    help="reproduce the pre-REJECT v1 source (to realign the first review)")
    a = ap.parse_args()
    r = harvest(a.gold, legacy=a.legacy)
    print(r)
    print("-> data/corpus/english_agri_sentences.txt (distillation source)")
    print("-> data/gold/source_en.txt (gold test-set source)")
