"""Motivating diagnostic: does off-the-shelf BanglaT5 fail on safety-critical
agricultural advice, and do our safety metrics detect it?

Runs the stock csebuetnlp/banglat5_nmt_en_bn on the safety probe set and scores
TermAcc / NumberMatch / NegationFaithfulness. Also runs the same weights WITH
numeric protection to show numbers can be rescued even before fine-tuning.
Writes a JSON + a human-readable Markdown table for the paper's Section 1.

    python -m eval.diagnose_baseline
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from agri_mt.banglat5 import BanglaT5Translator, load_glossary
from agri_mt.safety_metrics import (
    score_corpus,
    term_accuracy,
    number_match,
    negation_faithful,
)

HERE = Path(__file__).resolve().parent.parent
PROBE = HERE / "eval" / "safety_probe_set.json"
OUT_DIR = HERE / "outputs"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    probe = json.loads(PROBE.read_text(encoding="utf-8"))
    sents = probe["sentences"]
    srcs = [s["en"] for s in sents]
    glossary = load_glossary(HERE / "data" / "glossary.json")

    tr = BanglaT5Translator()  # stock weights, CPU/GPU auto
    t0 = time.time()
    baseline = tr.translate(srcs, protect=False)
    protected = tr.translate(srcs, protect=True, glossary=glossary)
    elapsed = time.time() - t0

    rows = []
    for s, b, p in zip(sents, baseline, protected):
        ta_b, _ = term_accuracy(s["en"], b, glossary)
        nm_b, nmd_b = number_match(s["en"], b)
        nf_b, _ = negation_faithful(s["en"], b)
        nm_p, _ = number_match(s["en"], p)
        rows.append({
            "id": s["id"], "stress": s["stress"], "en": s["en"],
            "baseline_bn": b, "protected_bn": p,
            "termacc": ta_b, "numbermatch_baseline": nm_b,
            "numbermatch_protected": nm_p, "neg_faithful": nf_b,
            "numbers_missing_baseline": nmd_b.get("missing", []),
        })

    agg_base = score_corpus(srcs, baseline, glossary).to_dict()
    agg_prot = score_corpus(srcs, protected, glossary).to_dict()
    payload = {
        "model": "csebuetnlp/banglat5_nmt_en_bn (off-the-shelf)",
        "device": tr.device, "n": len(srcs), "seconds": round(elapsed, 1),
        "aggregate_baseline": agg_base, "aggregate_protected": agg_prot,
        "rows": rows,
    }
    (OUT_DIR / "diagnose_baseline.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # Markdown report for the paper.
    md = ["# Baseline diagnostic — off-the-shelf BanglaT5 on safety sentences\n",
          f"Model: `{payload['model']}` · device: {tr.device} · {len(srcs)} sentences\n",
          "## Aggregate (lower = worse)\n",
          "| Metric | Baseline | +Numeric protection |",
          "|---|---|---|",
          f"| TermAcc | {_pct(agg_base['term_accuracy'])} | {_pct(agg_prot['term_accuracy'])} |",
          f"| NumberMatch | {_pct(agg_base['number_match'])} | {_pct(agg_prot['number_match'])} |",
          f"| NegationFaithfulness | {_pct(agg_base['negation_faithfulness'])} | {_pct(agg_prot['negation_faithfulness'])} |",
          "\n## Per-sentence (baseline)\n",
          "| id | stress | NumMatch | Neg ok | missing numbers | EN |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        md.append(
            f"| {r['id']} | {','.join(r['stress'])} | {_pct(r['numbermatch_baseline'])} | "
            f"{r['neg_faithful']} | {r['numbers_missing_baseline']} | {r['en'][:60]} |"
        )
    (OUT_DIR / "diagnose_baseline.md").write_text("\n".join(md), encoding="utf-8")

    print(f"Baseline: TermAcc={_pct(agg_base['term_accuracy'])} "
          f"NumberMatch={_pct(agg_base['number_match'])} "
          f"NegFaithful={_pct(agg_base['negation_faithfulness'])}")
    print(f"+Numeric protection: NumberMatch={_pct(agg_prot['number_match'])}")
    print(f"-> outputs/diagnose_baseline.md , .json")


def _pct(x) -> str:
    return "n/a" if x is None else f"{x*100:.0f}%"


if __name__ == "__main__":
    main()
