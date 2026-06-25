"""Build a BLIND A/B human-rating sheet (baseline vs adapted) for the gold outputs.

Each row shows the English source and two Bengali translations, A and B, in
randomised order (one baseline, one adapted) so the rater cannot tell which system
produced which. A hidden key records the mapping. The rater marks which is better;
``score_human_eval.py`` then computes the adapted win-rate.

    python -m agri_mt.make_human_eval --n 80
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "data" / "human_eval"


def _load(path):
    return {json.loads(l)["en"]: json.loads(l)["hyp"]
            for l in (HERE / path).read_text(encoding="utf-8").splitlines() if l.strip()}


def build(n: int, seed: int = 7) -> None:
    base = _load("outputs/hyps_baseline.jsonl")
    adapt = _load("outputs/hyps_adapted.jsonl")
    # only rate pairs that actually differ (otherwise the judgement is vacuous)
    ens = [e for e in adapt if e in base and adapt[e].strip() != base[e].strip()]
    rng = random.Random(seed)
    rng.shuffle(ens)
    sel = ens[:n]

    OUT.mkdir(parents=True, exist_ok=True)
    rows, key = [], []
    for i, en in enumerate(sel, 1):
        swap = rng.random() < 0.5          # swap=True -> A is baseline
        a = base[en] if swap else adapt[en]
        b = adapt[en] if swap else base[en]
        rows.append({"id": f"h{i:03d}", "english": en, "A": a, "B": b,
                     "better (A / B / =)": "", "comment (optional)": ""})
        key.append({"id": f"h{i:03d}", "adapted_side": "A" if not swap else "B"})

    with (OUT / "rating_sheet.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    with (OUT / "_key.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "adapted_side"])
        w.writeheader(); w.writerows(key)
    print(f"Wrote {len(rows)} blind A/B pairs -> data/human_eval/rating_sheet.csv")
    print("Fill the 'better (A / B / =)' column with A, B, or = . Key hidden in _key.csv.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.parse_args()
    build(ap.parse_args().n)
