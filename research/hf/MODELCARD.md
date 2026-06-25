---
license: cc-by-nc-sa-4.0
language:
  - bn
  - en
base_model: csebuetnlp/banglat5_nmt_en_bn
pipeline_tag: translation
library_name: peft
tags:
  - translation
  - bengali
  - agriculture
  - low-resource
  - lora
  - domain-adaptation
---

# AgriBanglaT5 — Agricultural English→Bengali Translation

A **LoRA adapter** for [`csebuetnlp/banglat5_nmt_en_bn`](https://huggingface.co/csebuetnlp/banglat5_nmt_en_bn)
that specialises it for **agricultural advisory** translation (English → Bengali).
Stock BanglaT5 mistranslates agricultural terminology (pesticide / disease / crop
names); this adapter, trained on the **AgriEnBn** corpus with glossary term hints,
renders the domain vocabulary correctly and translates more fluently — at the same
247M size and CPU latency.

## Results (AgriEnBn gold test set, 198 human-verified sentences)

| Metric | Stock BanglaT5 | **AgriBanglaT5** |
|---|---|---|
| chrF++ | 48.1 | **54.2** (+6.1) |
| BLEU | 23.4 | **28.1** (+4.7) |
| TermAcc (agri terms) | 76.4% | **81.8%** (+5.4 pp) |
| Negation faithfulness | 89.7% | **93.1%** (+3.4 pp) |
| NumberMatch | 97.2% | 96.3% (tie) |

## Usage
```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

base = "csebuetnlp/banglat5_nmt_en_bn"
tok = AutoTokenizer.from_pretrained(base, use_fast=False)
model = PeftModel.from_pretrained(AutoModelForSeq2SeqLM.from_pretrained(base),
                                  "<user>/AgriBanglaT5").merge_and_unload().eval()

# For the full terminology gain, append the canonical Bengali term as a hint
# after each glossary term in the source, e.g.:
#   "Apply Tricyclazole ⟦ট্রাইসাইক্লাজল⟧ at 0.6 gram per liter."
# (see the AgriEnBn repo for the glossary + helper).
src = "Spray Imidacloprid ⟦ইমিডাক্লোপ্রিড⟧ against brown planthopper ⟦বাদামি গাছফড়িং⟧."
ids = tok(src, return_tensors="pt").input_ids
print(tok.decode(model.generate(ids, num_beams=5, max_length=192)[0], skip_special_tokens=True))
```

## Training
- **Data:** AgriEnBn — 3,034 EN→BN pairs (back-translation of recovered Bijoy-encoded
  Bangladeshi agronomy documents + EN-side distillation), LaBSE-filtered ≥0.70.
- **Method:** LoRA (r=16, α=32) on the `q,v` projections; glossary inline term hints
  on the source. 3 epochs; held-out eval loss 2.56→2.36.

## Limitations
- Specialist for **agriculture** (esp. rice); not intended for general-domain text.
- Rare numeric errors remain (one pH range in the test set). Verify dosages.
- Trained partly on back-translated (synthetic) data.
- **Non-commercial** (CC BY-NC-SA 4.0), inherited from BanglaT5 / BanglaNMT.

## Citation
Cite this work, plus **BanglaT5** (Bhattacharjee et al., EACL Findings 2023) and
**BanglaNMT** (Hasan et al., EMNLP 2020).
