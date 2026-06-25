# COMET scoring (run in a SEPARATE Kaggle notebook)

COMET's pinned dependencies clash with our translation stack, so run it isolated —
**a fresh Kaggle notebook, not the training one.** GPU optional (CPU is fine for 198
sentences). Internet **On**.

## Setup
1. Kaggle → **New Dataset** → upload **`outputs/hyps_for_comet.zip`** (contains
   `hyps_baseline.jsonl` and `hyps_adapted.jsonl`) → name it `agri-hyps`.
2. New Notebook → **+ Add Input** → add `agri-hyps`. Internet **On**.

> **Important:** COMET's installer downgrades numpy to 1.26, which breaks Kaggle's
> torch (compiled for numpy 2) → `numpy.dtype size changed` error. So we install,
> then force numpy back, then **restart the kernel**. Two cells.

### Cell 1 — install, then repair numpy
```python
!pip -q install unbabel-comet
!pip -q install -U "numpy>=2.0"
print("\n>>> Installed. Now RESTART THE KERNEL (Run menu -> 'Restart & clear cell outputs'), then run Cell 2.")
```
**Then click Run → Restart & clear cell outputs** (the installed packages survive the
restart; the numpy fix takes effect).

### Cell 2 — score (after the restart)
```python
import glob, json
from comet import download_model, load_from_checkpoint

model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))

def find(name):
    return glob.glob(f"/kaggle/input/**/{name}", recursive=True)[0]

def comet_score(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    data = [{"src": r["en"], "mt": r["hyp"], "ref": r["ref"]} for r in rows]
    return model.predict(data, batch_size=16, gpus=0).system_score

for sys in ["baseline", "adapted"]:
    print(f"{sys:10} COMET = {comet_score(find(f'hyps_{sys}.jsonl')):.4f}")
```

### If it still fights you
COMET is a *nice-to-have*, not essential. chrF++ + BLEU + the safety metrics +
human eval is a perfectly standard metric set for a low-resource MT paper. If the
environment keeps breaking, **skip COMET** and we lean on the human evaluation
instead (which is stronger anyway). Don't lose time fighting dependency hell.

## What to expect
- COMET (wmt22-comet-da) ranges roughly 0–1; higher is better.
- We expect **adapted > baseline**, consistent with chrF++/BLEU. Send me both numbers
  and I'll add a COMET row to the results table.
