"""Full evaluation harness for the paper.

Scores a system on the held-out gold test set with standard MT metrics
(SacreBLEU, chrF++) plus the safety suite (TermAcc/NumberMatch/Negation), with a
bootstrap CI on chrF++. COMET is optional (--comet) since it needs a download.

    python -m eval.eval_translation --system baseline
    python -m eval.eval_translation --system protected
    python -m eval.eval_translation --system adapted --adapter outputs/agribanglat5-lora --comet
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from agri_mt.banglat5 import BanglaT5Translator, load_glossary
from agri_mt.safety_metrics import score_corpus

HERE = Path(__file__).resolve().parent.parent
OUT_DIR = HERE / "outputs"
DEFAULT_TESTSET = HERE / "data" / "gold" / "testset.jsonl"


def _bootstrap_chrf(hyps, refs, n=1000, seed=42):
    import sacrebleu

    rng = random.Random(seed)
    idx = range(len(hyps))
    scores = []
    for _ in range(n):
        sample = [rng.choice(range(len(hyps))) for _ in idx]
        h = [hyps[i] for i in sample]
        r = [[refs[i] for i in sample]]
        scores.append(sacrebleu.corpus_chrf(h, r, word_order=2).score)
    scores.sort()
    return scores[int(0.025 * n)], scores[int(0.975 * n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=["baseline", "protected", "adapted"])
    ap.add_argument("--adapter", default=None, help="LoRA adapter path (for 'adapted')")
    ap.add_argument("--testset", default=str(DEFAULT_TESTSET))
    ap.add_argument("--comet", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.testset).read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r.get("bn")]  # only human-verified references
    srcs = [r["en"] for r in rows]
    refs = [r["bn"] for r in rows]
    glossary = load_glossary(HERE / "data" / "glossary.json")

    protect = args.system in ("protected", "adapted")
    tr = BanglaT5Translator(lora_adapter=args.adapter if args.system == "adapted" else None)
    t0 = time.time()
    hyps = tr.translate(srcs, protect=protect, glossary=glossary if protect else None)
    elapsed = time.time() - t0

    import sacrebleu

    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
    chrf = sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score
    lo, hi = _bootstrap_chrf(hyps, refs)
    safety = score_corpus(srcs, hyps, glossary).to_dict()

    payload = {
        "system": args.system, "adapter": args.adapter, "n": len(rows),
        "device": tr.device, "seconds": round(elapsed, 1),
        "BLEU": round(bleu, 2), "chrF++": round(chrf, 2),
        "chrF++_95CI": [round(lo, 2), round(hi, 2)],
        "safety": safety,
    }
    if args.comet:
        payload["COMET"] = _comet(srcs, hyps, refs)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"hyps_{args.system}.jsonl").write_text(
        "\n".join(json.dumps({"en": s, "ref": r, "hyp": h}, ensure_ascii=False)
                  for s, r, h in zip(srcs, refs, hyps)), encoding="utf-8")
    out = OUT_DIR / f"eval_{args.system}_{int(time.time())}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / f"eval_{args.system}_latest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _comet(srcs, hyps, refs):
    from comet import download_model, load_from_checkpoint

    model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
    data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hyps, refs)]
    return round(model.predict(data, batch_size=16, gpus=0).system_score, 4)


if __name__ == "__main__":
    main()
