"""Standalone COMET scorer — run with the ISOLATED .venv_comet interpreter so
COMET's pinned deps (numpy<2 etc.) don't touch the main translation env.

    .venv_comet/Scripts/python.exe eval/score_comet.py
"""
import json
from comet import download_model, load_from_checkpoint

model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))


def score(path: str) -> float:
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    data = [{"src": r["en"], "mt": r["hyp"], "ref": r["ref"]} for r in rows]
    return model.predict(data, batch_size=8, gpus=0).system_score


for s in ["baseline", "adapted"]:
    print(f"{s:9} COMET = {score(f'outputs/hyps_{s}.jsonl'):.4f}", flush=True)
print("COMET_DONE")
