"""Mining-yield probe: how much of a general EN<->BN corpus is agricultural?

Streams a Hugging Face parallel dataset (no full download) and counts sentence
pairs whose ENGLISH side contains agricultural cues, at two precision tiers
(broad keyword list vs. the tight KG glossary). Reports the yield rate so we
know whether the corpus can be *mined* for real in-domain pairs or whether we
must lean on back-translation of the recovered Bengali documents.

    python -m agri_mt.mine_parallel --dataset opus100 --limit 200000
    python -m agri_mt.mine_parallel --dataset banglanmt --limit 200000
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from agri_mt.agri_keywords import BROAD_AGRI_KEYWORDS_EN, load_glossary_en_forms

HERE = Path(__file__).resolve().parent.parent
OUT_DIR = HERE / "data" / "corpus"

# Known HF parallel sources. Each entry: (hf_id, config, how-to-pull text fields).
DATASETS = {
    "opus100": {"id": "Helsinki-NLP/opus-100", "config": "bn-en", "kind": "translation"},
    "banglanmt": {"id": "csebuetnlp/BanglaNMT", "config": None, "kind": "pair"},
    "samanantar": {"id": "ai4bharat/samanantar", "config": "bn", "kind": "pair"},
}


def _build_keyword_regex(words: set[str]) -> re.Pattern:
    # Sort longest-first so multiword cues ("application rate") match.
    parts = sorted((re.escape(w) for w in words), key=len, reverse=True)
    return re.compile(r"(?<![a-z])(" + "|".join(parts) + r")(?![a-z])", re.IGNORECASE)


def _iter_pairs(spec: dict, limit: int):
    """Yield (en, bn) from a streaming HF dataset, tolerant of schema variants."""
    from datasets import load_dataset

    ds = load_dataset(
        spec["id"], spec["config"], split="train", streaming=True,
    )
    n = 0
    for row in ds:
        en = bn = None
        if "translation" in row and isinstance(row["translation"], dict):
            t = row["translation"]
            en, bn = t.get("en"), t.get("bn")
        else:
            en = row.get("en") or row.get("english") or row.get("source") or row.get("src")
            bn = row.get("bn") or row.get("bengali") or row.get("target") or row.get("tgt")
        if en and bn:
            yield en, bn
            n += 1
            if n >= limit:
                return


def probe(dataset: str, limit: int, save_hits: bool = True) -> dict:
    spec = DATASETS[dataset]
    broad_re = _build_keyword_regex(BROAD_AGRI_KEYWORDS_EN)
    gloss_forms = load_glossary_en_forms()
    gloss_re = _build_keyword_regex(gloss_forms) if gloss_forms else None

    total = broad_hits = gloss_hits = 0
    saved: list[dict] = []
    for en, bn in _iter_pairs(spec, limit):
        total += 1
        is_broad = bool(broad_re.search(en))
        if is_broad:
            broad_hits += 1
            if save_hits and len(saved) < 5000:
                saved.append({"en": en, "bn": bn})
        if gloss_re and gloss_re.search(en):
            gloss_hits += 1

    result = {
        "dataset": dataset,
        "hf_id": spec["id"],
        "scanned": total,
        "broad_agri_hits": broad_hits,
        "broad_yield_rate": round(broad_hits / total, 5) if total else 0,
        "glossary_hits": gloss_hits,
        "glossary_yield_rate": round(gloss_hits / total, 5) if total else 0,
    }
    if save_hits and saved:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / f"mined_{dataset}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for r in saved:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        result["saved_to"] = str(out.relative_to(HERE))
        result["saved_count"] = len(saved)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="opus100", choices=list(DATASETS))
    ap.add_argument("--limit", type=int, default=200000)
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()
    res = probe(args.dataset, args.limit, save_hits=not args.no_save)
    print(json.dumps(res, ensure_ascii=False, indent=2))
