"""Thin BanglaT5 EN->BN translator wrapper (baseline + safe variants + LoRA).

One class drives every configuration the paper compares:
  * baseline  : off-the-shelf csebuetnlp/banglat5_nmt_en_bn, as AgriBot ships it
  * protected : same weights + numeric-placeholder protection + glossary hints
  * adapted   : a LoRA-fine-tuned checkpoint (optionally + protection)

so evaluation runs identical decoding across systems and only the
preprocessing / weights differ.
"""
from __future__ import annotations

from pathlib import Path

from agri_mt.constrained_decode import preprocess_source, postprocess_target

BASE_MODEL = "csebuetnlp/banglat5_nmt_en_bn"


class BanglaT5Translator:
    def __init__(
        self,
        model_name_or_path: str = BASE_MODEL,
        lora_adapter: str | None = None,
        device: str | None = None,
    ):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        if lora_adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, lora_adapter)
            model = model.merge_and_unload()
        self.model = model.to(self.device).eval()

    def translate(
        self,
        texts: list[str],
        protect: bool = False,
        glossary: list[dict] | None = None,
        batch_size: int = 16,
        num_beams: int = 5,
        max_length: int = 192,
    ) -> list[str]:
        import torch

        prepared: list[str] = []
        number_maps: list[list[str]] = []
        for t in texts:
            if protect:
                src, nums = preprocess_source(t, glossary)
            else:
                src, nums = t, []
            prepared.append(src)
            number_maps.append(nums)

        outputs: list[str] = []
        for i in range(0, len(prepared), batch_size):
            batch = prepared[i : i + batch_size]
            enc = self.tok(batch, return_tensors="pt", padding=True, truncation=True,
                           max_length=max_length).to(self.device)
            with torch.no_grad():
                gen = self.model.generate(
                    **enc, max_length=max_length, num_beams=num_beams,
                    no_repeat_ngram_size=3, early_stopping=True,
                )
            outputs.extend(self.tok.batch_decode(gen, skip_special_tokens=True))

        if protect:
            outputs = [
                postprocess_target(o, nums) for o, nums in zip(outputs, number_maps)
            ]
        return outputs


def load_glossary(path: str | Path = "data/glossary.json") -> list[dict]:
    import json

    return json.loads(Path(path).read_text(encoding="utf-8"))
