"""Generate gold-test-set CANDIDATES for human post-editing.

The gold test set must be human-verified to be credible, but writing Bengali
from scratch is slow. Instead we machine-translate each English source to a
candidate, and the author only *verifies/fixes* it (fast). Output rows have an
empty ``bn`` field that the author fills; ``eval_translation.py`` ignores any
row whose ``bn`` is still empty, so the set can be edited incrementally.

Sources: the safety probe superset + (optional) sampled English agronomy
sentences. The safety subset (dosages/negations) is flagged for per-category
reporting.

    python -m agri_mt.make_testset --engine nllb
    # then a native speaker fills the "bn" field in data/gold/testset_candidates.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
GOLD_DIR = HERE / "data" / "gold"
PROBE = HERE / "eval" / "safety_probe_set.json"


def _candidate_translations(srcs: list[str], engine: str) -> list[str]:
    if engine == "nllb":
        from agri_mt.backtranslate import load_nllb, translate, BEN, ENG

        tok, model, device = load_nllb()
        return translate(srcs, ENG, BEN, tok, model, device)
    else:  # banglat5
        from agri_mt.banglat5 import BanglaT5Translator

        return BanglaT5Translator().translate(srcs)


def _has_neg(en: str) -> bool:
    return bool(re.search(r"\b(not|never|no|without|n't|do not)\b", en, re.I))


def _has_num(en: str) -> bool:
    return bool(re.search(r"\d", en))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="nllb", choices=["nllb", "banglat5"])
    ap.add_argument("--extra_en", default=None,
                    help="Optional text file of extra English agri sentences (one per line)")
    args = ap.parse_args()

    probe = json.loads(PROBE.read_text(encoding="utf-8"))["sentences"]
    rows = [{"id": s["id"], "en": s["en"], "stress": s["stress"],
             "safety": True} for s in probe]

    if args.extra_en and Path(args.extra_en).exists():
        extra = [l.strip() for l in Path(args.extra_en).read_text(encoding="utf-8").splitlines() if l.strip()]
        for i, en in enumerate(extra):
            stress = [t for t, f in (("negation", _has_neg(en)), ("number", _has_num(en))) if f]
            rows.append({"id": f"x{i:04d}", "en": en, "stress": stress,
                         "safety": bool(stress)})

    cands = _candidate_translations([r["en"] for r in rows], args.engine)
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    out = GOLD_DIR / "testset_candidates.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r, c in zip(rows, cands):
            f.write(json.dumps(
                {"id": r["id"], "en": r["en"], "bn_candidate": c, "bn": "",
                 "safety": r["safety"], "stress": r["stress"]},
                ensure_ascii=False) + "\n")
    n_safety = sum(r["safety"] for r in rows)
    print(f"Wrote {len(rows)} candidates ({n_safety} safety) -> {out}")
    print("Next: a native Bengali speaker fills the empty 'bn' field (verify/fix bn_candidate).")


if __name__ == "__main__":
    main()
