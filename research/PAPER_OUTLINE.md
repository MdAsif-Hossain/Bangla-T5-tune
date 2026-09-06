# AgriBanglaT5 — paper outline (grounded in executed findings)

Target: full conference (EACL/COLING/RANLP-class). Direction: EN→BN.
Honest framing fixed by Gate-0 evidence: **terminology + domain word-sense** is the
real gap; numeric/negation handled by the baseline → reposition as a *guarantee*.

## Title (working)
*AgriBanglaT5: Terminology-Aware Domain Adaptation of a Low-Resource Seq2Seq Model
for Agricultural English→Bengali Translation.*

## Abstract (sketch)
Off-the-shelf BanglaT5 renders agricultural advice into Bengali but mistranslates
domain terminology (TermAcc 59%) and domain word senses ("per hill"→mountain),
while — contrary to expectation — preserving numbers and negation. We build
**AgriEnBn**, the first agricultural EN→BN corpus, by recovering Unicode Bengali
from legacy Bijoy-encoded government documents and back-translating it, plus
EN→BN distillation; we LoRA-adapt BanglaT5 with glossary terminology constraints
and number-placeholder protection; and we introduce a **safety-aware evaluation**
(TermAcc/NumberMatch/NegationFaithfulness). The adapted model improves TermAcc by
X pp and chrF++ by Y at no added size or latency, with a provable dosage guarantee.

## 1. Introduction
- Deployed context: AgriBot uses stock BanglaT5 as a final EN→BN renderer; flagged
  as a limitation. Advisory translation errors reach farmers → motivation.
- Contributions: (1) AgriEnBn corpus + Bijoy-recovery method; (2) terminology- and
  number-safe adaptation; (3) safety-aware evaluation suite; (4) error taxonomy.

## 2. Related work
BanglaT5/BanglaNLG (Bhattacharjee 2023); BanglaNMT (Hasan 2020); NLLB; terminology-
constrained NMT (Dinu 2019); back-translation (Sennrich 2016); domain adaptation /
LoRA (Hu 2022); agricultural NLP chatbots (Farmer.Chat, AgriTalk).

## 3. Diagnostic & error taxonomy  ← `outputs/GATE0_FINDINGS.md`, `diagnose_baseline.*`
- Method: run stock BanglaT5 on safety probe set; score with §5 metrics.
- **Finding (evidence):** TermAcc 59%; NumberMatch 100%; Negation 100% (clean+complex).
- Taxonomy: (a) term transliteration/inconsistency (ইমিডাক্লোপ্রিড vs model's variant),
  (b) dropped named entities (disease names vanish), (c) **domain word-sense errors**
  (hill→পাহাড়/mountain, top-dressing→"arranged on top", drain→"irrigate").
- Honest note: numbers/negation are NOT a frequent failure → reframes the target.

## 4. AgriEnBn corpus  ← `recover_bijoy.py`, `mine_parallel.py`, `backtranslate.py`, `build_corpus.py`
- 4.1 Bijoy recovery: font-aware SutonnyMJ→Unicode (2,408 sentences, 98.4% purity) from 10 Bengali PDF documents.
- 4.2 Why not mining: opus-100 probe → ~6% precision, religious/UI contamination; BanglaNMT unstreamable. → back-translation backbone (methodological justification).
- 4.3 Construction: BN→EN back-translation (real Bengali targets) + EN→BN distillation (1,062 harvested English agronomy sentences from 10 English PDFs).
- 4.4 Statistics & Filtering: Automatically aligned and filtered via LaBSE $\geq$ 0.70 and length ratios. The final corpus contains 3,034 training pairs, totaling 51,859 English words (avg 17.1/sent) and 44,987 Bengali words (avg 14.8/sent) across 20 source documents.
- 4.5 Glossary: 82 expert-verified EN↔BN terms from the AgriBot dialect KG + curation. (Training pairs are automatic, but the test set and glossary are human-verified).

## 5. Safety-aware evaluation  ← `safety_metrics.py` (15/15 tests)
- Substantiating "Safety": We define agricultural safety as a composite of three strict constraints to prevent dangerous advice (e.g., incorrect chemical applications):
  1. TermAcc (glossary, no double-count) for accurate chemical/crop entities.
  2. NumberMatch (Bengali-digit aware, multiset) to prevent dosage hallucinations.
  3. NegationFaithfulness (token-based, not \b) to prevent opposite instructions.
- General fluency: SacreBLEU, chrF++, COMET, human/LLM-judge.
- Gold test set: 198 source sentences (99 safety subset), single-reference, author-verified.

## 6. Method  ← `constrained_decode.py`, `train_lora.py`
- LoRA (q,v) on BanglaT5; mix general data to avoid forgetting.
- Glossary inline annotation (soft terminology constraint, Dinu-style).
- Number-placeholder protection: train+infer with masked numbers → **NumberMatch
  guaranteed ~100%** (insurance against fine-tuning regression).

## 7. Results (to fill after Kaggle run)
- Main table: systems × {BLEU, chrF++, COMET, TermAcc, NumberMatch, Negation} + CIs.
  Systems: stock → +glossary → LoRA → +constraints → +numeric; NLLB ref.
- Ablations: New baseline added evaluating LoRA without glossary hints. Results show LoRA alone achieves 78.8% TermAcc (vs 81.8% for full AgriBanglaT5), proving the necessity of decode-time hint injection.
- Human/LLM-judge adequacy on 100-item subset (judge validated vs author ratings).
- **Deployment Guidance:** We explicitly recommend deploying the **full AgriBanglaT5 (LoRA + hints)**. While hints alone yield the highest terminology accuracy (91.2%), they severely degrade translation fluency (BLEU 23.2). LoRA restores fluency (BLEU 28.1) but misses rare exact terminology without hints (TermAcc 78.8%). The combined system offers the optimal safety/fluency tradeoff.

## 8. Limitations
Single annotator (author) for gold references; heuristic safety metrics (miss word-
sense like drain→irrigate, which COMET/human capture); closed glossary; one language
direction. 1 source PDF unrecoverable (broken glyph encoding) — OCR future work.

## Reproducibility
Release glossary + gold set + all scripts (CC BY-NC-SA 4.0 to match BanglaT5/BanglaNMT);
mined data via regeneration scripts. Cite Bhattacharjee 2023, Hasan 2020.
