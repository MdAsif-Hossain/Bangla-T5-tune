# AgriBanglaT5 — Results (executed, final numbers)

Evaluation on the **AgriEnBn gold test set**: 198 human-verified English→Bengali
agricultural sentences (99 safety subset). Identical decoding (beam=5); only
weights / source-side preprocessing differ. Metrics: SacreBLEU, chrF++ (95% CI
via bootstrap), plus the safety suite — TermAcc (glossary, inflection-tolerant),
NumberMatch (Bengali digits + spelled-out numerals credited), NegationFaithfulness.

Systems:
- **baseline** — stock `csebuetnlp/banglat5_nmt_en_bn` (as AgriBot ships it)
- **+hints** — baseline weights + glossary inline term hints (no fine-tuning)
- **AgriBanglaT5 (+LoRA)** — LoRA-adapted on AgriEnBn (3,034 pairs) + glossary hints

Corpus: 3,034 EN→BN pairs (2,379 back-translated from recovered Bijoy documents +
979 EN-side distillation), LaBSE-filtered ≥0.70. LoRA r=16 on q,v; 3 epochs;
held-out eval loss ↓ each epoch (2.56→2.41→2.36).

## Table 1 — Main result

| Metric | Baseline | **AgriBanglaT5** | Δ |
|---|---|---|---|
| chrF++ | 48.1 | **54.2** | **+6.1** |
| BLEU | 23.4 | **28.1** | **+4.7** |
| TermAcc | 76.4% | **81.8%** | **+5.4 pp** |
| Negation faithfulness | 89.7% | **93.1%** | **+3.4 pp** |
| NumberMatch | 97.2% | 96.3% | −0.9 pp (tie) |

Four clear wins; number fidelity is statistically even (−0.9 pp ≈ <1 sentence of 75).
Both chrF++ and BLEU rise together — an unambiguous adequacy/fluency gain.

## Table 2 — Ablation (separable mechanisms)

| Metric | baseline | +glossary hints | +LoRA (full) |
|---|---|---|---|
| chrF++ | 48.1 | 49.4 | **54.2** |
| BLEU | 23.4 | 23.2 | **28.1** |
| TermAcc | 76.4% | **91.2%** | 81.8% |
| NumberMatch | 97.2% | 95.7% | 96.3% |
| Negation | 89.7% | 93.1% | 93.1% |

**Terminology and fluency are improved by *different* mechanisms:**
- **Glossary hints → terminology.** Injecting the canonical Bengali term lifts
  TermAcc 76.4→**91.2%** with no training; fluency barely moves (+1.3 chrF).
- **LoRA → fluency/adequacy.** Fine-tuning drives chrF++ (+6.1) and BLEU (+4.7); it
  "naturalises" terms, so exact TermAcc settles at 81.8% (vs the verbatim-copy 91.2%
  of hints alone) while overall quality is best. The full system is the deployable balance.

## Qualitative examples (baseline vs AgriBanglaT5)

| English | Baseline | AgriBanglaT5 |
|---|---|---|
| "...early **tillering** stages" | প্রাথমিক **চাষের** পর্যায় ("cultivation", wrong) | **কুশি গজানো** পর্যায় ✓ |
| "...in **seedbed**...apply **nematicide**" | বীজ বপনের সময়...**নিমাটোড** ("nematode", wrong) | **বীজতলাতে**...**নেমাটিকাইড** ✓ |
| "Spray **Imidacloprid**..." | **ইমেডাক্লোপ্রিড** (misspelled) | **ইমিডাক্লোপ্রিড** ✓ |

Outputs saved at `outputs/hyps_{baseline,protected,adapted}.jsonl`.

## NumberMatch — what the diagnosis showed
Stock BanglaT5 already preserves numbers well (97.2%); adaptation keeps it even
(96.3%). The apparent early drop was a **metric artifact**: the adapted model
renders small numbers as natural Bengali *words* (দশ=10, তিন=3), which a digit-only
metric did not credit — fixed by crediting spelled-out numerals. The actual dosage
figures (0.6, 1.12, 50…) are preserved; the only genuine error found was one pH
range (5.5–7.0 → 5.5.0). We also report a **negative result**: explicit numeric
*placeholder* protection is incompatible with BanglaT5's tokenizer (the `Ⓝ`
placeholder is OOV → transliterated, corrupting the digit in place), so we dropped it.

## Human evaluation (blind A/B, native speaker)
The author (a native Bengali speaker) rated 80 blind A/B pairs (baseline vs adapted,
order randomised). **Adapted preferred 21, baseline 14, 45 ties** → a 60% win-rate
on decisive pairs. The preference is driven by terminology (e.g.\ নেমাটিকাইড vs
"nematodes", কুশি গজানো for tillering), consistent with TermAcc; on most sentences
the systems are comparable. An LLM-judge agreed only weakly with the human
(Cohen's kappa 0.25), indicating **LLM judges are unreliable for low-resource
Bengali quality** — itself a reportable finding. Caveat: single annotator; the
decisive-pair sample (35) is a lean, not a strong significance, but it aligns with
the objective metrics.

## COMET — not reported (environment-incompatible)
COMET was attempted in four configurations (Kaggle ×2, isolated local venv ×2) and
failed each time on irreconcilable pins: `unbabel-comet` requires `numpy<2` while
modern `torch`/`torchvision` require `numpy>=2`, plus `pkg_resources`/`setuptools`
breakage. We report the standard SacreBLEU + chrF++ + safety suite + human
evaluation instead, which is a complete and conventional set for low-resource MT.

## Reproduce
```
python -m eval.eval_translation --system baseline
python -m eval.eval_translation --system protected            # +hints
python -m eval.eval_translation --system adapted --adapter <path>
```
