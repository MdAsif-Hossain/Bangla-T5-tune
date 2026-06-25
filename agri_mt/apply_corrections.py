"""Apply the human review to the gold candidates and QC the result.

Input : data/gold/testset_candidates.jsonl  (id, en, bn_candidate, bn, safety, stress)
        data/gold/_corrections.json          ({empty:[ids], fixes:{id:bn}})
Output: data/gold/testset_candidates.jsonl   (bn field filled)
        data/gold/testset.jsonl              (verified references only)

Rule per row: empty-list -> bn=""; in fixes -> corrected bn; else -> copy the
candidate (the reviewer marked it correct).

QC (the whole-set check a sentence-by-sentence review can miss):
  * NumberMatch  - every source number present in the reference;
  * Negation     - source negation reflected in the reference;
  * Glossary     - how many references carry a glossary term (TermAcc coverage).
"""
from __future__ import annotations

import json
from pathlib import Path

from agri_mt.safety_metrics import number_match, negation_faithful, term_accuracy

HERE = Path(__file__).resolve().parent.parent
CAND = HERE / "data" / "gold" / "testset_candidates.jsonl"
CORR = HERE / "data" / "gold" / "_corrections.json"
OUT = HERE / "data" / "gold" / "testset.jsonl"
GLOSS = json.loads((HERE / "data" / "glossary.json").read_text(encoding="utf-8"))


def _rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def apply_and_qc() -> None:
    corr = json.loads(CORR.read_text(encoding="utf-8"))
    empty, fixes = set(corr["empty"]), corr["fixes"]
    en_fixes = corr.get("en_fixes", {})
    rows = _rows(CAND)

    n_fixed = n_copied = n_empty = 0
    for r in rows:
        rid = r["id"]
        if rid in en_fixes:                 # correct OCR typos in the source itself
            r["en"] = en_fixes[rid]
        if rid in empty:
            r["bn"] = ""
            n_empty += 1
        elif rid in fixes:
            r["bn"] = fixes[rid]
            n_fixed += 1
        else:
            r["bn"] = r.get("bn_candidate", "")
            n_copied += 1

    CAND.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")

    gold = [
        {"id": r["id"], "en": r["en"], "bn": r["bn"],
         "safety": r.get("safety", False), "stress": r.get("stress", [])}
        for r in rows if r["bn"].strip()
    ]
    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in gold), encoding="utf-8")

    # ---- QC ----
    num_flags, neg_flags, term_cov = [], [], 0
    for r in gold:
        nm, d = number_match(r["en"], r["bn"])
        if nm is not None and nm < 1.0:
            num_flags.append((r["id"], d["missing"], r["en"][:60]))
        nf, _ = negation_faithful(r["en"], r["bn"])
        if nf is False:
            neg_flags.append((r["id"], r["en"][:60]))
        ta, hits = term_accuracy(r["en"], r["bn"], GLOSS)
        if ta is not None:
            term_cov += 1

    print(f"applied: {n_fixed} fixed, {n_copied} copied, {n_empty} emptied")
    print(f"gold testset: {len(gold)} refs  ({sum(g['safety'] for g in gold)} safety) -> {OUT.name}")
    print(f"glossary coverage: {term_cov}/{len(gold)} refs contain >=1 glossary term")
    print(f"\nNUMBER flags ({len(num_flags)}) - review (some are EN '9 000' vs BN '৯,০০০' artifacts):")
    for rid, miss, en in num_flags:
        print(f"  {rid}: missing {miss} | {en}")
    print(f"\nNEGATION flags ({len(neg_flags)}) - source negated, reference not:")
    for rid, en in neg_flags:
        print(f"  {rid}: {en}")


if __name__ == "__main__":
    apply_and_qc()
