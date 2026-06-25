<div align="center">

# 🌾 AgriBanglaT5

### Domain-adapting BanglaT5 for *safe, accurate* agricultural English→Bengali translation

[![🤗 Model](https://img.shields.io/badge/🤗%20Model-AgriBanglaT5-yellow)](https://huggingface.co/Sabuktagin/AgriBanglaT5)
[![🤗 Dataset](https://img.shields.io/badge/🤗%20Dataset-AgriEnBn-yellow)](https://huggingface.co/datasets/Sabuktagin/AgriEnBn)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![PEFT](https://img.shields.io/badge/LoRA-PEFT-1C3C3C)
![License](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey)
![Tests](https://img.shields.io/badge/tests-15%2F15-success)

</div>

---

Off-the-shelf [BanglaT5](https://huggingface.co/csebuetnlp/banglat5_nmt_en_bn) translates *general* Bengali well — but on **agricultural advice for farmers** it mistranslates the terms that matter, sometimes dangerously:

| English | Stock BanglaT5 | **AgriBanglaT5** |
|---|---|---|
| "control rice **blast**" | চালের **বিস্ফোরণ** → *"rice **explosion**"* 😬 | ধানের **ব্লাস্ট রোগ** ✅ |
| "**brown planthopper**" | বাদামি **চারাগাছ** → *"brown **seedling**"* 😬 | বাদামি **গাছফড়িং** ✅ |
| "**drain** the field" | পানি **সেচ দিন** → *"**irrigate**"* (the opposite!) 😬 | জমি **নিষ্কাশন** করুন ✅ |

AgriBanglaT5 is a **parameter-efficient (LoRA) adaptation** of BanglaT5 that fixes this — at the **same 247M size and CPU latency** — trained on a purpose-built corpus and evaluated with a **safety-aware** protocol. Built end-to-end: legacy-data recovery → corpus construction → GPU fine-tuning → rigorous evaluation → public release.

## 📊 Results (198-sentence human-verified gold set)

| Metric | Stock BanglaT5 | **AgriBanglaT5** | Δ |
|---|---|---|---|
| **Term accuracy** (agri vocab) | 76.4% | **81.8%** | **+5.4 pp** |
| **chrF++** | 48.1 | **54.2** | **+6.1** |
| BLEU | 23.4 | **28.1** | **+4.7** |
| Negation faithfulness | 89.7% | **93.1%** | **+3.4 pp** |
| Number preservation | 97.2% | 96.3% | ≈ tie |
| **Human eval** (native speaker, blind A/B) | — | **preferred 60%** of decisive pairs | — |

An **ablation** separates the mechanisms: glossary term hints drive *terminology* (76→**91%** TermAcc, no training); LoRA drives *fluency* (the chrF++/BLEU gains).

## 🛠️ How it works

```
Legacy Bijoy PDFs ──(font-aware recovery)──► 2,408 Unicode Bengali sentences
        │                                              │
        └─► back-translation (NLLB) ◄──────────────────┘   ┐
English agronomy text ─► distillation (NLLB) EN→BN          ├─► AgriEnBn (3,034 pairs)
KG-derived glossary (82 terms) ────────────────────────────┘            │
                                                                        ▼
                              LoRA fine-tune BanglaT5 + glossary term hints
                                                                        ▼
                       Safety-aware eval: TermAcc · NumberMatch · NegationFaithfulness
```

**Three contributions:**
1. **AgriEnBn** — the first agricultural EN→BN corpus, built partly by recovering Unicode Bengali from legacy **Bijoy/SutonnyMJ**-encoded government documents (where normal extraction yields gibberish).
2. **AgriBanglaT5** — LoRA-adapted BanglaT5 with inline glossary term constraints.
3. A **safety-aware evaluation** suite (term accuracy, dosage-number preservation, negation faithfulness) + an honest error taxonomy.

## 🚀 Quickstart

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

base = "csebuetnlp/banglat5_nmt_en_bn"
tok = AutoTokenizer.from_pretrained(base, use_fast=False)
model = PeftModel.from_pretrained(
    AutoModelForSeq2SeqLM.from_pretrained(base), "Sabuktagin/AgriBanglaT5"
).merge_and_unload().eval()

src = "Apply Tricyclazole ⟦ট্রাইসাইক্লাজল⟧ to control rice blast ⟦ব্লাস্ট রোগ⟧."
ids = tok(src, return_tensors="pt").input_ids
print(tok.decode(model.generate(ids, num_beams=5, max_length=192)[0], skip_special_tokens=True))
```
(The `⟦…⟧` glossary hints are appended automatically by the helper in `agri_mt/`.)

## 📁 Repository

```
agri_mt/        corpus build (Bijoy recovery, back-translation, glossary), constrained decode, safety metrics
eval/           evaluation harness + safety-metric scoring
train/          LoRA training (Kaggle runbook) + deployment guide
tests/          unit tests (15/15) for the safety metrics + number/term guarantees
data/           glossary (82 terms) + 198-sentence human-verified gold test set
```

Reproduce the evaluation:
```bash
pip install -r requirements.txt
python -m eval.eval_translation --system baseline
python -m eval.eval_translation --system adapted --adapter <path-or-hub-id>
```

## 🔬 What this project demonstrates
Low-resource NMT · domain adaptation with **LoRA/PEFT** · synthetic-data engineering (back-translation + distillation) · **recovering data from legacy encodings** · designing **task-specific evaluation metrics** · ablations, **human evaluation**, and honest negative results (a failed numeric-placeholder scheme; LLM-judges shown unreliable for low-resource Bengali) · end-to-end ML from raw PDFs to a released, reproducible model.

## ⚠️ Limitations
Agricultural specialist (esp. rice), not general-purpose · corpus is partly synthetic and Bangladesh-centric · single-annotator human eval · rare numeric errors remain. Released **CC BY-NC-SA 4.0** (inherited from BanglaT5 / BanglaNMT) — non-commercial.

## 📚 Citation & credit
Builds on **BanglaT5** (Bhattacharjee et al., EACL 2023) and **BanglaNMT** (Hasan et al., EMNLP 2020); uses **NLLB** (Costa-jussà et al., 2022) as the back-translation teacher.

**Author:** Md. Asif Hossain · East West University, Dhaka.
