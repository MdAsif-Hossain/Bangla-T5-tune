"""Tally a filled blind A/B rating sheet into an adapted win-rate.

Reads data/human_eval/rating_sheet.csv (with the 'better (A / B / =)' column
filled with A, B, or =) and the hidden _key.csv, and reports how often the
ADAPTED system was preferred. If a second rater file is given, also reports
Cohen's kappa agreement.

    python -m agri_mt.score_human_eval
    python -m agri_mt.score_human_eval --rater2 data/human_eval/rating_sheet_llm.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
HE = HERE / "data" / "human_eval"


def _picks(sheet: Path) -> dict[str, str]:
    out = {}
    for r in csv.DictReader(sheet.read_text(encoding="utf-8-sig").splitlines()):
        col = next(k for k in r if k.startswith("better"))
        out[r["id"]] = (r[col] or "").strip().upper()
    return out


def _key() -> dict[str, str]:
    return {r["id"]: r["adapted_side"]
            for r in csv.DictReader((HE / "_key.csv").read_text(encoding="utf-8-sig").splitlines())}


def _verdicts(picks, key) -> dict[str, str]:
    """id -> 'adapted' | 'baseline' | 'tie' (only for filled rows)."""
    v = {}
    for rid, p in picks.items():
        if p not in {"A", "B", "="}:
            continue
        if p == "=":
            v[rid] = "tie"
        else:
            v[rid] = "adapted" if p == key[rid] else "baseline"
    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", default=str(HE / "rating_sheet.csv"))
    ap.add_argument("--rater2", default=None)
    a = ap.parse_args()
    key = _key()
    v1 = _verdicts(_picks(Path(a.sheet)), key)

    n = len(v1)
    adapt = sum(x == "adapted" for x in v1.values())
    base = sum(x == "baseline" for x in v1.values())
    tie = sum(x == "tie" for x in v1.values())
    decisive = adapt + base
    print(f"rated: {n}/80")
    print(f"  adapted preferred : {adapt} ({adapt/n:.0%})")
    print(f"  baseline preferred: {base} ({base/n:.0%})")
    print(f"  tie               : {tie} ({tie/n:.0%})")
    if decisive:
        print(f"  adapted win-rate (excl. ties): {adapt/decisive:.0%}")

    if a.rater2:
        v2 = _verdicts(_picks(Path(a.rater2)), key)
        ids = [i for i in v1 if i in v2]
        agree = sum(v1[i] == v2[i] for i in ids)
        po = agree / len(ids)
        # Cohen's kappa over {adapted,baseline,tie}
        from collections import Counter
        c1, c2 = Counter(v1[i] for i in ids), Counter(v2[i] for i in ids)
        pe = sum((c1[k]/len(ids)) * (c2[k]/len(ids)) for k in {"adapted", "baseline", "tie"})
        kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
        print(f"\ninter-rater (author vs rater2): observed agreement {po:.0%}, "
              f"Cohen's kappa {kappa:.2f} over {len(ids)} items")


if __name__ == "__main__":
    main()
