"""Back-translation / forward-distillation with NLLB-200 to build AgriEnBn.

After Gate 0b, this is the corpus BACKBONE:

  * BN->EN over the recovered Bijoy sentences (data/recovered/all_bn_sentences.txt)
    yields (EN, BN) pairs whose **Bengali side is genuine in-domain text** — the
    highest-quality source we have.
  * EN->BN over English agronomy sentences (FAO/IRRI PDFs, AgriBot answers)
    yields distillation pairs with a strong-teacher Bengali target.

Designed for free Kaggle GPU (T4/P100). Defaults to the distilled 600M model so
it also runs (slowly) on CPU for a smoke test. Larger = better:
nllb-200-1.3B / 3.3B on Kaggle for the real run.

    # on Kaggle:
    python -m agri_mt.backtranslate --direction bn2en \
        --infile data/recovered/all_bn_sentences.txt \
        --out data/corpus/bt_bijoy.jsonl --model facebook/nllb-200-1.3B
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

BEN, ENG = "ben_Beng", "eng_Latn"
HERE = Path(__file__).resolve().parent.parent


def load_nllb(model_name: str = "facebook/nllb-200-distilled-600M", device: str | None = None):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device).eval()
    return tok, model, device


def translate(
    texts: list[str], src_lang: str, tgt_lang: str, tok, model, device,
    batch_size: int = 16, max_length: int = 256, num_beams: int = 4,
) -> list[str]:
    import torch

    tok.src_lang = src_lang
    bos = tok.convert_tokens_to_ids(tgt_lang)
    out: list[str] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_length).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, forced_bos_token_id=bos,
                                 max_length=max_length, num_beams=num_beams)
        out.extend(tok.batch_decode(gen, skip_special_tokens=True))
    return out


def run(direction: str, infile: Path, out: Path, model_name: str, limit: int | None):
    src_lines = [l.strip() for l in Path(infile).read_text(encoding="utf-8").splitlines() if l.strip()]
    if limit:
        src_lines = src_lines[:limit]
    tok, model, device = load_nllb(model_name)
    if direction == "bn2en":
        en = translate(src_lines, BEN, ENG, tok, model, device)
        pairs = [{"en": e, "bn": b, "src": "bt_bijoy"} for e, b in zip(en, src_lines)]
    elif direction == "en2bn":
        bn = translate(src_lines, ENG, BEN, tok, model, device)
        pairs = [{"en": e, "bn": b, "src": "distill_en"} for e, b in zip(src_lines, bn)]
    else:
        raise ValueError(direction)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"[{direction}] {len(pairs)} pairs via {model_name} ({device}) -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", required=True, choices=["bn2en", "en2bn"])
    ap.add_argument("--infile", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="facebook/nllb-200-distilled-600M")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    run(a.direction, a.infile, a.out, a.model, a.limit)
