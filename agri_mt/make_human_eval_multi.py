"""Make blank per-rater copies of the blind A/B sheet for a multi-annotator study.

All raters judge the SAME 80 pairs in the SAME A/B layout (the hidden key in
``_key.csv`` is shared), so the author's existing ratings remain valid as rater 1
and only the co-authors need to fill a sheet. Each co-author independently marks
the 'better (A / B / =)' column.

    python -m agri_mt.make_human_eval_multi --raters 3

Writes data/human_eval/rating_sheet_rater2.csv ... rater{N+1}.csv (blank answers),
plus RATER_INSTRUCTIONS.txt. Score with agri_mt.score_human_eval_multi.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
HE = HERE / "data" / "human_eval"
BLANK_COLS = ("better (A / B / =)", "comment (optional)")

INSTRUCTIONS = """\
HUMAN EVALUATION -- AgriBanglaT5 (blind A/B)

You are comparing two Bengali translations of the same English agricultural
sentence. You do NOT know which system produced A or B (that is intentional).

For each row in your sheet (rating_sheet_raterX.csv):
  * Read the 'english' source, then the two translations 'A' and 'B'.
  * Decide which Bengali translation is BETTER for a farmer -- judged on:
        (1) correct agricultural TERMS (disease/pest/pesticide/crop names),
        (2) correct meaning (no inversions, e.g. drain vs irrigate),
        (3) fluency / naturalness.
  * Put exactly one of:  A  /  B  /  =   in the 'better (A / B / =)' column.
        A = A is better,  B = B is better,  = = genuinely equal / can't decide.
  * 'comment' is optional (e.g. note a wrong term).

Please judge INDEPENDENTLY -- do not discuss with the other raters until all
sheets are returned. Open the CSV in Excel/Google Sheets; keep the column order.
Return the saved CSV. Thank you!
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raters", type=int, default=3, help="number of CO-authors (rater2..)")
    args = ap.parse_args()

    src = HE / "rating_sheet.csv"
    rows = list(csv.DictReader(src.read_text(encoding="utf-8-sig").splitlines()))
    if not rows:
        raise SystemExit("rating_sheet.csv is empty -- run agri_mt.make_human_eval first.")

    made = []
    for r in range(2, 2 + args.raters):          # rater2, rater3, rater4 (rater1 = author)
        out = HE / f"rating_sheet_rater{r}.csv"
        blank = [{**row, **{c: "" for c in BLANK_COLS if c in row}} for row in rows]
        with out.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(blank)
        made.append(out.name)

    (HE / "RATER_INSTRUCTIONS.txt").write_text(INSTRUCTIONS, encoding="utf-8")
    print(f"Wrote {len(rows)} blind pairs to each of: {', '.join(made)}")
    print("Rater 1 = author (existing rating_sheet_author.csv).")
    print("Hand one rating_sheet_raterX.csv + RATER_INSTRUCTIONS.txt to each co-author.")


if __name__ == "__main__":
    main()
