---
license: cc-by-nc-sa-4.0
language:
  - bn
  - en
task_categories:
  - translation
tags:
  - bengali
  - agriculture
  - low-resource
  - parallel-corpus
pretty_name: AgriEnBn
---

# AgriEnBn — Agricultural English↔Bengali Corpus + Gold Test Set

The first parallel corpus for **agricultural** English→Bengali machine translation,
built to specialise [BanglaT5](https://huggingface.co/csebuetnlp/banglat5_nmt_en_bn)
for the domain. Created for the **AgriBanglaT5** model.

## Contents
| File | Rows | Description |
|---|---|---|
| `agrienbn.train.jsonl` | 2,883 | training pairs (`{en, bn, src}`) |
| `agrienbn.dev.jsonl` | 151 | dev pairs |
| `testset.jsonl` | 198 | **human-verified** gold test set (`{en, bn, safety}`); 99 safety subset |
| `glossary.json` | 77 | EN↔BN agronomy terms (canonical + aliases + type) |

## How it was built
1. **Bijoy recovery** — Bangladeshi government agronomy PDFs store Bengali in the
   legacy ASCII-mapped *SutonnyMJ* font (normal extraction yields gibberish). We
   recover Unicode Bengali font-aware (2,408 sentences, 98.4% Bengali purity).
2. **Back-translation** of that in-domain Bengali (NLLB-200-1.3B) → real-Bengali-target
   pairs; plus **distillation** of English agronomy sentences (FAO/IRRI) EN→BN.
3. **Filtering** — dedup, language/length checks, LaBSE cosine ≥ 0.70.
4. **Glossary** — derived from a dialect knowledge graph, expert-verified.
5. **Gold test set** — NLLB candidates **post-edited and verified by a native speaker**,
   with a safety subset (dosages + negations).

## Intended use
Training/evaluating agricultural EN→BN MT. The gold test set ships with a
**safety-aware evaluation** (term accuracy, number preservation, negation
faithfulness) — see the code repo.

## Limitations & licensing
- Training pairs are partly synthetic (back-translation); the **gold test set** is
  human-verified.
- Source documents are Bangladesh/rice-centric.
- **CC BY-NC-SA 4.0** (inherited from BanglaT5/BanglaNMT). Non-commercial.

## Citation
Cite this work, **BanglaT5** (Bhattacharjee et al., EACL Findings 2023), **BanglaNMT**
(Hasan et al., EMNLP 2020), and **NLLB** (Costa-jussà et al., 2022).
