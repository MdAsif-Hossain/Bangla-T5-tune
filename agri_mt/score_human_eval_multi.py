"""Score a multi-annotator blind A/B study into preference + Fleiss' kappa.

Rater 1 is the author (rating_sheet_author.csv); raters 2.. are the co-authors
(rating_sheet_rater2.csv, ...). All decode against the shared _key.csv into
{adapted, baseline, tie}. Reports per-rater adapted win-rate, the pooled and
majority-vote preference, and Fleiss' kappa (inter-annotator agreement) over the
items every rater judged.

    python -m agri_mt.score_human_eval_multi
"""
from __future__ import annotations

import argparse
import csv
import itertools
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
HE = HERE / "data" / "human_eval"
CATS = ("adapted", "baseline", "tie")


def _picks(sheet: Path) -> dict[str, str]:
    out = {}
    for r in csv.DictReader(sheet.read_text(encoding="utf-8-sig").splitlines()):
        col = next(k for k in r if k.startswith("better"))
        out[r["id"]] = (r[col] or "").strip().upper()
    return out


def _validate(name: str, picks: dict[str, str], n_expected: int = 80) -> None:
    """Surface mechanical entry errors (blanks / typos) so they can be fixed."""
    blank = [i for i, p in picks.items() if p == ""]
    bad = {i: p for i, p in picks.items() if p not in {"A", "B", "=", ""}}
    if len(picks) != n_expected:
        print(f"  ! {name}: {len(picks)} rows (expected {n_expected}) -- "
              "rows added/removed or ids changed")
    if blank:
        print(f"  ! {name}: {len(blank)} blank -> {', '.join(blank[:8])}"
              + (" ..." if len(blank) > 8 else ""))
    if bad:
        print(f"  ! {name}: {len(bad)} invalid (not A/B/=) -> "
              + ", ".join(f'{i}={p!r}' for i, p in list(bad.items())[:8]))


def _key() -> dict[str, str]:
    return {r["id"]: r["adapted_side"]
            for r in csv.DictReader((HE / "_key.csv").read_text(encoding="utf-8-sig").splitlines())}


def _verdicts(picks, key) -> dict[str, str]:
    v = {}
    for rid, p in picks.items():
        if p == "=":
            v[rid] = "tie"
        elif p in {"A", "B"}:
            v[rid] = "adapted" if p == key[rid] else "baseline"
    return v


def _cohen(a: dict[str, str], b: dict[str, str]) -> tuple[int, float]:
    ids = [i for i in a if i in b]
    if not ids:
        return 0, float("nan")
    po = sum(a[i] == b[i] for i in ids) / len(ids)
    ca, cb = Counter(a[i] for i in ids), Counter(b[i] for i in ids)
    pe = sum((ca[c] / len(ids)) * (cb[c] / len(ids)) for c in CATS)
    return len(ids), (po - pe) / (1 - pe) if pe < 1 else 1.0


def _fleiss(verdicts: list[dict[str, str]]) -> tuple[int, float]:
    """Fleiss' kappa over the items rated by ALL raters."""
    common = set(verdicts[0])
    for v in verdicts[1:]:
        common &= set(v)
    common = sorted(common)
    n = len(verdicts)
    if not common or n < 2:
        return 0, float("nan")
    # per-item category counts
    rows = [[sum(v[i] == c for v in verdicts) for c in CATS] for i in common]
    N = len(rows)
    Pi = [(sum(x * x for x in row) - n) / (n * (n - 1)) for row in rows]
    P_bar = sum(Pi) / N
    pj = [sum(row[j] for row in rows) / (N * n) for j in range(len(CATS))]
    Pe = sum(p * p for p in pj)
    kappa = (P_bar - Pe) / (1 - Pe) if Pe < 1 else 1.0
    return N, kappa


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--author", default=str(HE / "rating_sheet_author.csv"))
    ap.add_argument("--raters", nargs="*", default=None,
                    help="co-author sheets; default = all rating_sheet_rater*.csv present")
    args = ap.parse_args()
    key = _key()

    sheets = [("rater1 (author)", Path(args.author))]
    rater_files = ([Path(p) for p in args.raters] if args.raters
                   else sorted(HE.glob("rating_sheet_rater*.csv")))
    for i, p in enumerate(rater_files, start=2):
        if p.exists():
            sheets.append((f"rater{i} ({p.stem.replace('rating_sheet_', '')})", p))

    # data-quality pass first: flag blanks / typos so they can be fixed
    print("data check (mechanical errors only):")
    problems = False
    for name, p in sheets:
        if p.exists():
            picks = _picks(p)
            if any(v for v in picks.values()):   # skip not-yet-returned sheets
                before = problems
                _validate(name, picks)
                problems = problems or (len(picks) != 80) or \
                    any(v not in {"A", "B", "=", ""} for v in picks.values()) or \
                    any(v == "" for v in picks.values())
    print("  all returned sheets clean" if not problems else
          "  ^ fix the flagged rows, re-save, and re-run")
    print()

    verdicts = []
    print(f"{'rater':24s} {'rated':>6s} {'adapt':>6s} {'base':>6s} {'tie':>5s} {'win%':>6s}")
    for name, p in sheets:
        if not p.exists():
            print(f"{name:24s}  (missing: {p.name})"); continue
        v = _verdicts(_picks(p), key)
        n = len(v)
        if not n:
            print(f"{name:24s} {'--':>6s}   (blank: not returned yet)"); continue
        verdicts.append(v)
        a = sum(x == "adapted" for x in v.values())
        b = sum(x == "baseline" for x in v.values()); t = n - a - b
        win = a / (a + b) if (a + b) else float("nan")
        print(f"{name:24s} {n:>6d} {a:>6d} {b:>6d} {t:>5d} {win:>6.0%}")

    if len(verdicts) < 2:
        print("\nNeed >=2 filled sheets for agreement / majority."); return

    # majority vote per item (over items all raters judged)
    common = sorted(set.intersection(*[set(v) for v in verdicts]))
    maj = Counter()
    for i in common:
        c = Counter(v[i] for v in verdicts).most_common()
        maj[c[0][0] if (len(c) == 1 or c[0][1] > c[1][1]) else "tie"] += 1
    ma, mb, mt = maj["adapted"], maj["baseline"], maj["tie"]
    print(f"\nitems judged by all {len(verdicts)} raters: {len(common)}")
    print(f"  majority-vote: adapted {ma}, baseline {mb}, tie/split {mt}"
          + (f"  -> adapted win-rate {ma/(ma+mb):.0%}" if (ma + mb) else ""))

    # pooled preference across all ratings
    allv = [x for v in verdicts for x in v.values()]
    pa, pb = allv.count("adapted"), allv.count("baseline")
    print(f"  pooled ({len(allv)} ratings): adapted {pa}, baseline {pb}, "
          f"tie {len(allv)-pa-pb}" + (f"  -> win-rate {pa/(pa+pb):.0%}" if (pa + pb) else ""))

    N, fk = _fleiss(verdicts)
    print(f"\nFleiss' kappa: {fk:.2f} over {N} items x {len(verdicts)} raters (3 categories)")
    pair = [_cohen(verdicts[i], verdicts[j])[1]
            for i, j in itertools.combinations(range(len(verdicts)), 2)]
    if pair:
        print(f"mean pairwise Cohen's kappa: {sum(pair)/len(pair):.2f}")


if __name__ == "__main__":
    main()
