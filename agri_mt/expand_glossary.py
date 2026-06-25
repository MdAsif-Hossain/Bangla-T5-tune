"""Harvest candidate EN<->BN glossary terms from the recovered Bengali corpus.

The Bengali side is GROUND TRUTH (real government agronomy text, 98.4% clean), so
the Bengali terms are correct by construction. We:
  1. extract frequent Bengali n-grams whose head is a domain word
     (রোগ disease, পোকা pest, সার fertilizer, নাশক -cide, আগাছা weed, অভাব deficiency, ...),
  2. propose an English gloss for each with the cached BanglaT5 BN->EN model,
  3. write a review CSV where the author marks keep + fixes the English.

Approved rows are merged back into data/glossary.{tsv,json} by --merge.

    python -m agri_mt.expand_glossary --top 120        # build review CSV
    python -m agri_mt.expand_glossary --merge data/glossary_review.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
RECOVERED = HERE / "data" / "recovered" / "all_bn_sentences.txt"
REVIEW_CSV = HERE / "data" / "glossary_review.csv"
GLOSSARY_JSON = HERE / "data" / "glossary.json"

# Domain head/markers — an n-gram containing one of these is likely a term.
HEAD_WORDS = {
    "রোগ", "পোকা", "মাছি", "ফড়িং", "সার", "আগাছা", "ঘাস", "অভাব", "দাগ",
    "ধসা", "পচা", "পচন", "ঝলসা", "নাশক", "মোড়ানো", "শোষক", "কুশি", "শীষ",
    "ব্লাইট", "ব্লাস্ট", "টুংরো", "ভাইরাস", "ছত্রাক", "ব্যাকটেরিয়া",
}
# Generic Bengali words to drop from term edges.
STOP = {
    "ও", "এবং", "এই", "একটি", "করে", "করা", "হয়", "হয়ে", "যায়", "জন্য", "থেকে",
    "সাথে", "মধ্যে", "করতে", "হবে", "এর", "যে", "তবে", "করার", "দিতে", "নিতে",
    "অন্য", "আরও", "খুব", "বেশি", "কম", "সব", "এসব", "একে", "তার", "এতে", "পরে",
}
_PUNCT = re.compile(r"[।,;:!?\.\(\)\"'`\-–—০-৯\d%]")


# Inflectional suffixes to strip so রোগ/রোগের/রোগে collapse to one term.
_SUFFIXES = ("জনিতম", "াক্রান্তম", "গুলোর", "গুলো", "দের", "ের", "কে", "তে", "রা", "টি", "টা", "য়", "ে", "র")


def _normalize_term(tok: str) -> str:
    for suf in _SUFFIXES:
        if tok.endswith(suf) and len(tok) - len(suf) >= 2:
            return tok[: -len(suf)]
    return tok


def _tokens(line: str) -> list[str]:
    line = _PUNCT.sub(" ", line)
    return [
        _normalize_term(t)
        for t in line.split()
        if t and t not in STOP and len(t) > 1
    ]


def _is_term(ngram: tuple[str, ...]) -> bool:
    return any(any(h in tok for h in HEAD_WORDS) for tok in ngram)


def harvest(top: int) -> list[tuple[str, int]]:
    existing = {g["bn"] for g in json.loads(GLOSSARY_JSON.read_text(encoding="utf-8"))}
    counts: Counter[str] = Counter()
    for line in RECOVERED.read_text(encoding="utf-8").splitlines():
        toks = _tokens(line)
        for n in (1, 2, 3):
            for i in range(len(toks) - n + 1):
                ng = tuple(toks[i : i + n])
                if _is_term(ng):
                    counts[" ".join(ng)] += 1
    # Keep reasonably frequent, drop ones already in the glossary.
    cands = [(t, c) for t, c in counts.most_common() if c >= 3 and t not in existing]
    # Prefer multi-word terms (more specific) by light reweighting.
    cands.sort(key=lambda x: (x[1] + (3 if " " in x[0] else 0)), reverse=True)
    return cands[:top]


def english_candidates(bn_terms: list[str]) -> list[str]:
    from agri_mt.banglat5 import BanglaT5Translator  # reuse loader
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    import torch

    name = "csebuetnlp/banglat5_nmt_bn_en"
    tok = AutoTokenizer.from_pretrained(name, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(name).eval()
    out = []
    for i in range(0, len(bn_terms), 16):
        batch = bn_terms[i : i + 16]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=32)
        with torch.no_grad():
            gen = model.generate(**enc, max_length=32, num_beams=4)
        out.extend(tok.batch_decode(gen, skip_special_tokens=True))
    return out


def build_review(top: int) -> None:
    cands = harvest(top)
    bn_terms = [t for t, _ in cands]
    en_cands = english_candidates(bn_terms)
    with REVIEW_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["keep(Y/N)", "bn_term", "en_APPROVED(fix me)", "en_candidate", "freq", "type(crop/disease/pest/chemical/fertilizer/weed/symptom)"])
        for (bn, c), en in zip(cands, en_cands):
            w.writerow(["", bn, en, en, c, ""])
    print(f"Wrote {len(cands)} candidate terms -> {REVIEW_CSV.name}")
    print("Your job: open it, set keep=Y on good rows, fix the English, add a type. Then run --merge.")


def merge(csv_path: str) -> None:
    rows = list(csv.DictReader(Path(csv_path).read_text(encoding="utf-8-sig").splitlines()))
    gloss = json.loads(GLOSSARY_JSON.read_text(encoding="utf-8"))
    have = {g["bn"] for g in gloss}
    added = 0
    for r in rows:
        if r.get("keep(Y/N)", "").strip().upper() == "Y":
            bn = r["bn_term"].strip()
            en = (r["en_APPROVED(fix me)"] or "").strip()
            if bn and en and bn not in have:
                gloss.append({"en": en, "bn": bn, "type": (r.get("type", "") or "term").strip(),
                              "en_aliases": [], "bn_aliases": []})
                have.add(bn)
                added += 1
    GLOSSARY_JSON.write_text(json.dumps(gloss, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Merged {added} approved terms -> glossary now {len(gloss)} entries.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=120)
    ap.add_argument("--merge", default=None)
    a = ap.parse_args()
    if a.merge:
        merge(a.merge)
    else:
        build_review(a.top)
