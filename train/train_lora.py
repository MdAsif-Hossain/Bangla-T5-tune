"""LoRA fine-tuning of BanglaT5 EN->BN on AgriEnBn (run on free Kaggle GPU).

Source/target are masked with the SAME number placeholders and glossary term
hints (agri_mt.constrained_decode.mask_training_pair), so the adapted model
learns to (a) copy dosage placeholders verbatim and (b) emit canonical Bengali
terms. LoRA keeps the 247M base frozen — tiny adapter, same inference cost.

Kaggle usage (T4/P100, one GPU):
    pip install -U transformers peft accelerate datasets sentencepiece
    python train/train_lora.py \
        --corpus data/corpus/agrienbn.train.jsonl \
        --dev    data/corpus/agrienbn.dev.jsonl \
        --out    outputs/agribanglat5-lora --epochs 3 --bs 16
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow `python train/train_lora.py` (script mode) to import the agri_mt package
# by putting the project root on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agri_mt.constrained_decode import mask_training_pair

BASE = "csebuetnlp/banglat5_nmt_en_bn"


def _read_pairs(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--dev", default=None)
    ap.add_argument("--out", default="outputs/agribanglat5-lora")
    ap.add_argument("--glossary", default="data/glossary.json")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max_len", type=int, default=192)
    ap.add_argument("--lora_r", type=int, default=16)
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
        Seq2SeqTrainer, Seq2SeqTrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, TaskType

    glossary = json.loads(Path(args.glossary).read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(BASE, use_fast=False)

    def build(path: str) -> Dataset:
        rows = _read_pairs(path)
        src, tgt = [], []
        for r in rows:
            s, t = mask_training_pair(r["en"], r["bn"], glossary)
            src.append(s)
            tgt.append(t)
        return Dataset.from_dict({"src": src, "tgt": tgt})

    def tokenize(batch):
        model_inputs = tok(batch["src"], max_length=args.max_len, truncation=True)
        labels = tok(text_target=batch["tgt"], max_length=args.max_len, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_ds = build(args.corpus).map(tokenize, batched=True, remove_columns=["src", "tgt"])
    eval_ds = build(args.dev).map(tokenize, batched=True, remove_columns=["src", "tgt"]) if args.dev else None

    model = AutoModelForSeq2SeqLM.from_pretrained(BASE)
    lora = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, r=args.lora_r, lora_alpha=args.lora_r * 2,
        lora_dropout=0.05, target_modules=["q", "v"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    targs = Seq2SeqTrainingArguments(
        output_dir=args.out, per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs, learning_rate=args.lr,
        num_train_epochs=args.epochs, warmup_ratio=0.1, weight_decay=1e-4,
        logging_steps=50, save_strategy="epoch",
        eval_strategy="epoch" if eval_ds else "no",
        predict_with_generate=True, report_to="none",
        # T5 is numerically unstable in fp16 (NaN loss). Use bf16 on Ampere+ GPUs,
        # else full fp32 (e.g. on T4). Never fp16 here.
        fp16=False, bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    )
    collator = DataCollatorForSeq2Seq(tok, model=model)
    _tr = dict(model=model, args=targs, train_dataset=train_ds,
               eval_dataset=eval_ds, data_collator=collator)
    try:  # transformers >=4.46 renamed tokenizer -> processing_class (removed in v5)
        trainer = Seq2SeqTrainer(**_tr, processing_class=tok)
    except TypeError:
        trainer = Seq2SeqTrainer(**_tr, tokenizer=tok)
    trainer.train()
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"Saved LoRA adapter -> {args.out}")


if __name__ == "__main__":
    main()
