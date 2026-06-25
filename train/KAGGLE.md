# Running AgriBanglaT5 on free Kaggle GPU

Everything below runs in **one Kaggle notebook** with **GPU T4 x1** enabled
(Settings → Accelerator → GPU T4 x1). Total runtime ≈ 1–2 h, well within the
free 30 h/week quota.

## 0. Get the project onto Kaggle (no GitHub needed)
You do **not** upload this file. You upload the project zip and run the cells below
inside a Kaggle notebook.

1. **kaggle.com → Datasets → New Dataset** → upload **`agribanglat5_kaggle.zip`**
   (in your project folder) → name it `agribanglat5` → **Create**. Kaggle unzips it.
2. **Create → New Notebook**. Then **Settings → Accelerator → GPU T4 x1**, and
   **Settings → Internet → On** (needed to download the models).
3. Right panel **+ Add Input** → search your `agribanglat5` dataset → add it.
   It mounts read-only at `/kaggle/input/agribanglat5/`.
4. First notebook cell — copy to a writable folder and install deps:

```python
import glob, os, shutil, zipfile
dest = "/kaggle/working/agribanglat5"
if not os.path.isdir(os.path.join(dest, "agri_mt")):       # only build if missing
    hits = glob.glob("/kaggle/input/**/agri_mt", recursive=True)
    if hits:
        shutil.copytree(os.path.dirname(hits[0]), dest, dirs_exist_ok=True)
    else:
        z = glob.glob("/kaggle/input/**/*.zip", recursive=True)[0]
        print("unzipping", z); zipfile.ZipFile(z).extractall(dest)
os.chdir(dest)
print("cwd:", os.getcwd(), "| agri_mt:", os.path.isdir("agri_mt"),
      "| corpus built:", os.path.isfile("data/corpus/agrienbn.train.jsonl"))

!pip -q install peft accelerate datasets sentencepiece sacrebleu sentence-transformers
!pip -q install -U "transformers>=4.46"
```
Safe to re-run (it won't wipe a corpus you already built). The Kaggle kernel can
reset its working directory on reconnect, so **every cell below starts with
`%cd`** to re-point itself. If `corpus built: True`, skip cells 1–3 and go to cell 4.

The zip already contains the inputs: `data/recovered/all_bn_sentences.txt`
(2,408 BN sentences), `data/corpus/english_agri_sentences.txt`, `data/glossary.json`,
`data/gold/`. Now run cells 1–5 below.

> Alternative (for the public release later): push to GitHub and `!git clone` — but
> first un-ignore `data/recovered/` and `data/corpus/` in `.gitignore`, or the
> inputs won't be in the repo.

## 1. Back-translation = corpus backbone (BN→EN over the recovered Bijoy text)
The Bengali side stays genuine in-domain text; NLLB supplies the English side.

```python
%cd /kaggle/working/agribanglat5
!python -m agri_mt.backtranslate --direction bn2en \
    --infile data/recovered/all_bn_sentences.txt \
    --out data/corpus/bt_bijoy.jsonl \
    --model facebook/nllb-200-1.3B
```

## 2. Forward distillation (EN→BN over harvested English agronomy sentences)
```python
%cd /kaggle/working/agribanglat5
!python -m agri_mt.backtranslate --direction en2bn \
    --infile data/corpus/english_agri_sentences.txt \
    --out data/corpus/distill_en.jsonl \
    --model facebook/nllb-200-1.3B
```

## 3. Assemble + filter AgriEnBn
```python
%cd /kaggle/working/agribanglat5
!python -m agri_mt.build_corpus --labse 0.70   # LaBSE semantic filter
```

## 4. LoRA fine-tune BanglaT5 (the adaptation)
```python
%cd /kaggle/working/agribanglat5
!python train/train_lora.py \
    --corpus data/corpus/agrienbn.train.jsonl \
    --dev    data/corpus/agrienbn.dev.jsonl \
    --out    outputs/agribanglat5-lora --epochs 3 --bs 16
```

## 5. Save the adapter
```python
%cd /kaggle/working/agribanglat5
import shutil; shutil.make_archive("/kaggle/working/agribanglat5-lora", "zip", "outputs/agribanglat5-lora")
print("Done → download agribanglat5-lora.zip from the right-hand Output panel")
```

## 6. Back on your laptop — evaluate
```bash
python -m eval.eval_translation --system baseline
python -m eval.eval_translation --system adapted --adapter outputs/agribanglat5-lora --comet
```

### Notes
- `facebook/nllb-200-1.3B` fits a T4; drop to `nllb-200-distilled-600M` if VRAM is tight,
  or step up to `nllb-200-3.3B` for best back-translation quality if you use P100/2×T4.
- Re-run 1–2 with more monolingual Bengali (BARI/BRRI/AIS leaflets) to grow the corpus
  later — same scripts, just longer input files.
