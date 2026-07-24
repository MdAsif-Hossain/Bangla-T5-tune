# Graph Report - agri_mt  (2026-07-25)

## Corpus Check
- Corpus is ~9,022 words - fits in a single context window. You may not need a graph.

## Summary
- 150 nodes · 203 edges · 15 communities (13 shown, 2 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 8 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Safety Metrics & Corrections
- Back-translation & Translation
- Constrained Decoding & Hints
- Glossary Management
- Bijoy Recovery
- Multi-Rater Scoring (Fleiss)
- Keyword Mining
- Glossary Expansion
- Human-Eval Scoring
- English Harvesting
- Regression Checks
- Corpus Building
- Human-Eval Sheets
- Multi-Rater Sheets
- Package Init

## God Nodes (most connected - your core abstractions)
1. `main()` - 8 edges
2. `GlossaryEntry` - 6 edges
3. `load_glossary()` - 6 edges
4. `extract_recovered_text()` - 6 edges
5. `recover_all()` - 6 edges
6. `term_accuracy()` - 6 edges
7. `apply_and_qc()` - 5 edges
8. `merge_additions()` - 5 edges
9. `export()` - 5 edges
10. `harvest()` - 5 edges

## Surprising Connections (you probably didn't know these)
- `probe()` --calls--> `load_glossary_en_forms()`  [INFERRED]
  mine_parallel.py → agri_keywords.py
- `apply_and_qc()` --calls--> `negation_faithful()`  [INFERRED]
  apply_corrections.py → safety_metrics.py
- `apply_and_qc()` --calls--> `number_match()`  [INFERRED]
  apply_corrections.py → safety_metrics.py
- `apply_and_qc()` --calls--> `term_accuracy()`  [INFERRED]
  apply_corrections.py → safety_metrics.py
- `_candidate_translations()` --calls--> `load_nllb()`  [INFERRED]
  make_testset.py → backtranslate.py

## Import Cycles
- None detected.

## Communities (15 total, 2 thin omitted)

### Community 0 - "Safety Metrics & Corrections"
Cohesion: 0.12
Nodes (22): apply_and_qc(), Path, Apply the human review to the gold candidates and QC the result.  Input : data/g, _rows(), _bn_has_negation(), _bn_term_in(), load_glossary_json(), negation_faithful() (+14 more)

### Community 1 - "Back-translation & Translation"
Cohesion: 0.13
Nodes (14): load_nllb(), Path, Back-translation / forward-distillation with NLLB-200 to build AgriEnBn.  After, run(), translate(), BanglaT5Translator, load_glossary(), Path (+6 more)

### Community 2 - "Constrained Decoding & Hints"
Cohesion: 0.18
Nodes (13): annotate_source_with_terms(), mask_training_pair(), postprocess_target(), preprocess_source(), protect_numbers(), Terminology- and number-safe translation wrappers for BanglaT5.  Two mechanisms,, Source-side prep: annotate glossary terms only.      Number masking was REMOVED:, Restore numbers in the model output. (Glossary hints are not emitted by a     pr (+5 more)

### Community 3 - "Glossary Management"
Cohesion: 0.26
Nodes (11): _dedup(), export(), GlossaryEntry, load_glossary(), merge_additions(), Path, Agricultural EN<->BN glossary.  Source of truth: AgriBot's dialect knowledge gra, Fold human-curated extras in, keeping TermAcc honest:      * if the Bengali form (+3 more)

### Community 4 - "Bijoy Recovery"
Cohesion: 0.27
Nodes (11): _converter(), extract_recovered_text(), _is_bijoy(), _normalize_bengali_words(), _normalizer(), Path, Recover Unicode Bengali from legacy Bijoy/SutonnyMJ-encoded agronomy PDFs.  Agri, Return the page text of a PDF with Bijoy spans converted to Unicode. (+3 more)

### Community 5 - "Multi-Rater Scoring (Fleiss)"
Cohesion: 0.27
Nodes (11): _cohen(), _fleiss(), _key(), main(), _picks(), Path, Score a multi-annotator blind A/B study into preference + Fleiss' kappa.  Rater, Surface mechanical entry errors (blanks / typos) so they can be fixed. (+3 more)

### Community 6 - "Keyword Mining"
Cohesion: 0.24
Nodes (8): load_glossary_en_forms(), Agricultural domain lexicon for mining/filtering parallel sentences.  Two tiers:, _build_keyword_regex(), _iter_pairs(), probe(), Mining-yield probe: how much of a general EN<->BN corpus is agricultural?  Strea, Yield (en, bn) from a streaming HF dataset, tolerant of schema variants., Pattern

### Community 7 - "Glossary Expansion"
Cohesion: 0.36
Nodes (7): build_review(), english_candidates(), harvest(), _is_term(), _normalize_term(), Harvest candidate EN<->BN glossary terms from the recovered Bengali corpus.  The, _tokens()

### Community 8 - "Human-Eval Scoring"
Cohesion: 0.39
Nodes (7): _key(), main(), _picks(), Path, Tally a filled blind A/B rating sheet into an adapted win-rate.  Reads data/huma, id -> 'adapted' | 'baseline' | 'tie' (only for filled rows)., _verdicts()

### Community 9 - "English Harvesting"
Cohesion: 0.48
Nodes (6): _clean(), _clean_sentence(), _good(), harvest(), Harvest clean English agricultural sentences from the English source PDFs.  Two, _sentences()

### Community 10 - "Regression Checks"
Cohesion: 0.53
Nodes (5): _find(), _load_hyps(), main(), Path, Check whether a model's outputs fixed the diagnosed word-sense / fluency bugs.

### Community 11 - "Corpus Building"
Cohesion: 0.60
Nodes (4): _labse_scores(), main(), _ok_pair(), Assemble AgriEnBn from the source shards, filter, and split train/dev.  Inputs:

### Community 12 - "Human-Eval Sheets"
Cohesion: 0.67
Nodes (3): build(), _load(), Build a BLIND A/B human-rating sheet (baseline vs adapted) for the gold outputs.

## Knowledge Gaps
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Should `Safety Metrics & Corrections` be split into smaller, more focused modules?**
  _Cohesion score 0.12 - nodes in this community are weakly interconnected._
- **Should `Back-translation & Translation` be split into smaller, more focused modules?**
  _Cohesion score 0.13450292397660818 - nodes in this community are weakly interconnected._