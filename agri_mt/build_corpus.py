"""Assemble AgriEnBn from the source shards, filter, and split train/dev.

Inputs: any *.jsonl in data/corpus/ with {"en","bn"} (back-translated Bijoy,
EN->BN distillation, optionally strictly-filtered mined pairs). Filters out the
noise that hurts a small fine-tune: dups, empties, copy-throughs, wrong-language
sides, and wildly mismatched lengths. Optional LaBSE semantic filter (--labse).

    python -m agri_mt.build_corpus --labse 0.70
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
CORPUS_DIR = HERE / "data" / "corpus"
BENGALI_RE = re.compile(r"[ঀ-৿]")


def _ok_pair(en: str, bn: str) -> bool:
    if not en or not bn:
        return False
    if en.strip() == bn.strip():
        return False
    en_lat = len(re.findall(r"[A-Za-z]", en))
    bn_ben = len(BENGALI_RE.findall(bn))
    if en_lat < 3 or bn_ben < 3:           # each side must be in its language
        return False
    if BENGALI_RE.search(en):               # English side shouldn't be Bengali
        return False
    we, wb = len(en.split()), len(bn.split())
    if we < 2 or wb < 2:
        return False
    if not (0.4 <= we / max(wb, 1) <= 2.5): # length-ratio sanity
        return False
    return True


def _labse_scores(pairs, threshold):
    from sentence_transformers import SentenceTransformer, util

    m = SentenceTransformer("sentence-transformers/LaBSE")
    en = m.encode([p["en"] for p in pairs], convert_to_tensor=True, normalize_embeddings=True,
                  batch_size=64, show_progress_bar=False)
    bn = m.encode([p["bn"] for p in pairs], convert_to_tensor=True, normalize_embeddings=True,
                  batch_size=64, show_progress_bar=False)
    keep = []
    for i, p in enumerate(pairs):
        if float(util.cos_sim(en[i], bn[i])) >= threshold:
            keep.append(p)
    return keep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labse", type=float, default=0.0, help="LaBSE cosine threshold (0=off)")
    ap.add_argument("--dev_frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    shards = sorted(p for p in CORPUS_DIR.glob("*.jsonl")
                    if not p.name.startswith(("agrienbn", "mined_")))  # mined opt-in
    seen, pairs, by_src = set(), [], {}
    for shard in shards:
        for line in shard.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            en, bn = (r.get("en") or "").strip(), (r.get("bn") or "").strip()
            if not _ok_pair(en, bn):
                continue
            key = (en, bn)
            if key in seen:
                continue
            seen.add(key)
            r2 = {"en": en, "bn": bn, "src": r.get("src", shard.stem)}
            pairs.append(r2)
            by_src[r2["src"]] = by_src.get(r2["src"], 0) + 1

    if args.labse > 0 and pairs:
        before = len(pairs)
        pairs = _labse_scores(pairs, args.labse)
        print(f"LaBSE>={args.labse}: kept {len(pairs)}/{before}")

    random.Random(args.seed).shuffle(pairs)
    n_dev = int(len(pairs) * args.dev_frac)
    dev, train = pairs[:n_dev], pairs[n_dev:]

    for name, rows in [("agrienbn.train.jsonl", train), ("agrienbn.dev.jsonl", dev)]:
        with (CORPUS_DIR / name).open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = {"total": len(pairs), "train": len(train), "dev": len(dev),
             "by_source": dict(sorted(by_src.items())), "shards": [s.name for s in shards]}
    (CORPUS_DIR / "agrienbn.stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
