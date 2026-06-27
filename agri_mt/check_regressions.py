"""Check whether a model's outputs fixed the diagnosed word-sense / fluency bugs.

Reads data/regression_cases.jsonl (the failures found in the 4-rater human study)
and a hypotheses file (default outputs/hyps_adapted.jsonl, lines of {"en","hyp"}),
matches each case by an English substring, and reports PASS/FAIL. Also flags
adjacent-word repetition (e.g. হলুদ হলুদ) across all hypotheses.

    python -m agri_mt.check_regressions
    python -m agri_mt.check_regressions --hyps outputs/hyps_adapted_v2.jsonl
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

try:                                   # print Bengali on a cp1252 Windows console
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = Path(__file__).resolve().parent.parent


def _load_hyps(path: Path) -> list[dict]:
    return [json.loads(l) for l in io.open(path, encoding="utf-8") if l.strip()]


def _find(hyps: list[dict], needle: str) -> str | None:
    n = needle.lower()
    for h in hyps:
        if n in (h.get("en", "") or "").lower():
            return h.get("hyp", "") or ""
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyps", default=str(HERE / "outputs" / "hyps_adapted.jsonl"))
    ap.add_argument("--cases", default=str(HERE / "data" / "regression_cases.jsonl"))
    a = ap.parse_args()

    hyps = _load_hyps(Path(a.hyps))
    cases = [json.loads(l) for l in io.open(a.cases, encoding="utf-8") if l.strip()]

    passed = 0
    print(f"regression check on {Path(a.hyps).name} ({len(hyps)} hyps)\n")
    for c in cases:
        out = _find(hyps, c["en_contains"])
        if out is None:
            print(f"  ?  {c['name']:26s} (source not found in hyps)"); continue
        has_good = any(s in out for s in c["must_any"])
        has_bad = [s for s in c["must_not"] if s in out]
        ok = has_good and not has_bad
        passed += ok
        why = "" if ok else ("  <-- missing " + "/".join(c["must_any"]) if not has_good
                             else "  <-- still has " + ", ".join(has_bad))
        print(f"  {'PASS' if ok else 'FAIL'}  {c['name']:26s}{why}")

    # adjacent-word repetition across all hyps (the হলুদ হলুদ artifact)
    rep = []
    for h in hyps:
        toks = (h.get("hyp", "") or "").split()
        if any(toks[i] == toks[i + 1] and len(toks[i]) > 1 for i in range(len(toks) - 1)):
            rep.append(h.get("en", "")[:40])
    print(f"\n  {len(cases)} cases: {passed} PASS, {len(cases) - passed} FAIL")
    print(f"  adjacent-word repetition in {len(rep)}/{len(hyps)} hyps"
          + (f" (e.g. '{rep[0]}...')" if rep else ""))


if __name__ == "__main__":
    main()
