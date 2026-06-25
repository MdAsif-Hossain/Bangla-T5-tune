"""Safety-aware evaluation metrics for agricultural EN->BN translation.

Standard MT metrics (BLEU/chrF/COMET) treat a dropped dosage digit and a clumsy
word choice as comparable errors. In an advisory setting they are not: a flipped
negation or a corrupted dose can harm a crop. These three objective, fully
automatic metrics target exactly those failure modes and need no human raters.

  TermAcc            - of the agricultural glossary terms appearing in the
                       English source, the fraction rendered with the correct
                       canonical Bengali term (or an accepted Bengali alias).
  NumberMatch        - fraction of sentences whose source numbers ALL survive
                       into the Bengali output (Bengali digits normalized to
                       ASCII). The number-corruption / dropped-dose detector.
  NegationFaithfulness - fraction of sentences whose negation polarity is
                       preserved (source negated  <=>  hypothesis negated).
                       The "do NOT drain" -> "drain" flip detector.

All three return a corpus-level score in [0, 1] plus per-sentence detail so the
paper can report breakdowns (e.g. on the dosage/negation safety subset).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

BENGALI_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
_NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")

# English GRAMMATICAL negation cues only. We deliberately exclude lexical
# negatives like "avoid"/"prevent": they translate to positive Bengali
# constructions ("রোগ প্রতিরোধ করুন"), so including them would wrongly flag
# faithful translations. The dangerous failure we target is the grammatical
# flip ("do NOT drain" -> "drain"), which these cues capture.
_EN_NEG = re.compile(
    r"\b(not|never|no|none|neither|nor|without|cannot|can't|won't|don't|"
    r"doesn't|didn't|shouldn't|mustn't|isn't|aren't|wasn't|weren't)\b|n't\b",
    re.IGNORECASE,
)
# Bengali negation: detected by TOKEN, never by regex \b — Bengali combining
# vowel signs make \b fire inside words (e.g. "নি\b" matches inside নিষ্কাশন).
_BN_NEG_TOKENS = {"না", "নাই", "নেই", "নয়", "নন", "নহে", "নাহ"}
_BN_NEG_SUBSTR = ("বিনা", "ব্যতীত", "ছাড়া", "এড়ি", "এড়া", "হীন")  # privatives (incl. -হীন "-less")
_BN_TOKEN_SPLIT = re.compile(r"[\s।,;:!?\.\(\)‌‍]+")


def _bn_has_negation(s: str) -> bool:
    toks = _BN_TOKEN_SPLIT.split(s or "")
    if any(t in _BN_NEG_TOKENS for t in toks):
        return True
    return any(sub in (s or "") for sub in _BN_NEG_SUBSTR)


def _norm_digits(s: str) -> str:
    return (s or "").translate(BENGALI_DIGITS)


# Spelled-out Bengali numerals — natural Bengali often writes small numbers as
# words ("দশ মিমি" for "10 mm"). Credit them so NumberMatch doesn't penalise a
# correct word-form rendering. Whole-token match only (avoids এক ⊂ একাধিক).
_BN_NUM_WORDS = {
    "শূন্য": "0", "এক": "1", "দুই": "2", "তিন": "3", "চার": "4", "পাঁচ": "5",
    "ছয়": "6", "সাত": "7", "আট": "8", "নয়": "9", "দশ": "10", "এগারো": "11",
    "বারো": "12", "তেরো": "13", "চৌদ্দ": "14", "পনেরো": "15", "ষোলো": "16",
    "সতেরো": "17", "আঠারো": "18", "ঊনিশ": "19", "বিশ": "20", "ত্রিশ": "30",
    "চল্লিশ": "40", "পঞ্চাশ": "50", "ষাট": "60", "সত্তর": "70", "আশি": "80",
    "নব্বই": "90", "একশ": "100", "একশো": "100", "শত": "100",
}
_TOK_SPLIT = re.compile(r"[\s।,;:!?\.\(\)–—\-]+")


def numbers_in(s: str) -> list[str]:
    """Multiset of numeric tokens. Bengali digits normalized; space-separated
    thousand groups joined ("9 000"->"9000"); spelled-out Bengali numerals
    ("দশ"->"10") credited so correct word-form renderings are not penalised.
    """
    s0 = s or ""
    s = _norm_digits(s0)
    s = re.sub(r"(?<=\d)\s+(?=\d{3}(?:\D|$))", "", s)  # "9 000" -> "9000"
    nums = [n.replace(",", "") for n in _NUM_RE.findall(s)]
    nums += [_BN_NUM_WORDS[t] for t in _TOK_SPLIT.split(s0) if t in _BN_NUM_WORDS]
    return nums


# ---------------------------------------------------------------- TermAcc -----

# Trailing vowel-signs/marks stripped to tolerate Bengali inflection when
# matching a glossary term (ঢলে পড়া vs ঢলে পড়ে). Bengali is agglutinative;
# stem-tolerant term matching follows the Bangla-stemmed ROUGE used by prior work.
_BN_TRAIL = "ািীুূৃেৈোৌংঃ"


def _bn_term_in(form: str, hyp: str) -> bool:
    """True if the glossary form occurs in the hypothesis, tolerating a single
    trailing inflection on the form (so the stem still matches inflected text)."""
    if form and form in hyp:
        return True
    stripped = form
    while stripped and stripped[-1] in _BN_TRAIL:
        stripped = stripped[:-1]
    return len(stripped) >= 3 and stripped != form and stripped in hyp


@dataclass
class TermHit:
    en_term: str
    expected_bn: list[str]
    rendered: bool


def term_accuracy(
    source_en: str,
    hyp_bn: str,
    glossary: list[dict],
) -> tuple[float | None, list[TermHit]]:
    """Score correct rendering of glossary terms found in the English source.

    A term is *applicable* if any of its English surface forms occurs in the
    source. It is *rendered* if any accepted Bengali form occurs in the output.
    Returns (accuracy or None if no applicable terms, per-term hits).
    """
    src = (source_en or "").lower()
    hyp = hyp_bn or ""
    hits: list[TermHit] = []
    for g in glossary:
        en_forms = [g["en"], *g.get("en_aliases", [])]
        if not any(re.search(rf"\b{re.escape(f.lower())}\b", src) for f in en_forms if f):
            continue
        bn_forms = [g["bn"], *g.get("bn_aliases", [])]
        rendered = any(_bn_term_in(f, hyp) for f in bn_forms)
        hits.append(TermHit(g["en"], bn_forms, rendered))
    if not hits:
        return None, hits
    return sum(h.rendered for h in hits) / len(hits), hits


# ------------------------------------------------------------ NumberMatch -----

def number_match(source_en: str, hyp_bn: str) -> tuple[float | None, dict]:
    """1.0 if every source number survives into the hypothesis, else fraction.

    Returns (score or None when the source has no numbers, detail dict).
    Comparison is multiset-based so duplicated values must all appear.
    """
    src_nums = numbers_in(source_en)
    if not src_nums:
        return None, {"src": [], "missing": []}
    hyp_nums = numbers_in(hyp_bn)
    remaining = list(hyp_nums)
    missing = []
    for n in src_nums:
        if n in remaining:
            remaining.remove(n)
        else:
            missing.append(n)
    score = (len(src_nums) - len(missing)) / len(src_nums)
    return score, {"src": src_nums, "hyp": hyp_nums, "missing": missing}


# ----------------------------------------------------- NegationFaithfulness ---

def negation_faithful(source_en: str, hyp_bn: str) -> tuple[bool | None, dict]:
    """True if negation polarity matches between source and hypothesis.

    Returns (None) when the source carries no negation (not part of the
    negation subset), else (bool faithful, detail).
    """
    src_neg = bool(_EN_NEG.search(source_en or ""))
    if not src_neg:
        return None, {"src_neg": False}
    hyp_neg = _bn_has_negation(hyp_bn or "")
    return (src_neg == hyp_neg), {"src_neg": src_neg, "hyp_neg": hyp_neg}


# ------------------------------------------------------------- corpus roll-up -

@dataclass
class SafetyReport:
    n: int
    term_accuracy: float | None
    term_applicable: int
    number_match: float | None
    number_applicable: int
    negation_faithfulness: float | None
    negation_applicable: int

    def to_dict(self) -> dict:
        return asdict(self)


def score_corpus(
    sources: list[str],
    hypotheses: list[str],
    glossary: list[dict],
) -> SafetyReport:
    assert len(sources) == len(hypotheses), "sources/hypotheses length mismatch"
    term_num, term_den = 0.0, 0
    num_num, num_den = 0.0, 0
    neg_ok, neg_den = 0, 0
    for s, h in zip(sources, hypotheses):
        ta, hits = term_accuracy(s, h, glossary)
        if ta is not None:
            term_num += ta * len(hits)
            term_den += len(hits)
        nm, _ = number_match(s, h)
        if nm is not None:
            num_num += nm
            num_den += 1
        nf, _ = negation_faithful(s, h)
        if nf is not None:
            neg_ok += int(nf)
            neg_den += 1
    return SafetyReport(
        n=len(sources),
        term_accuracy=(term_num / term_den) if term_den else None,
        term_applicable=term_den,
        number_match=(num_num / num_den) if num_den else None,
        number_applicable=num_den,
        negation_faithfulness=(neg_ok / neg_den) if neg_den else None,
        negation_applicable=neg_den,
    )


def load_glossary_json(path: str | Path = "data/glossary.json") -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
