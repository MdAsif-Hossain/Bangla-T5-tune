"""Generate the paper figures from the real result/loss data.

    python scripts/make_figures.py
Writes PDF (vector, for the paper) + PNG (preview) into research/paper_mt/figures/.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = Path(__file__).resolve().parent.parent
FIG = HERE / "research" / "paper_mt" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

def L(s): return json.loads((HERE / f"outputs/eval_{s}_latest.json").read_text(encoding="utf-8"))
b, p, a = L("baseline"), L("protected"), L("adapted")


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("wrote", name)


# ---- Fig 1: ablation (separable mechanisms) ----
sys_names = ["Baseline", "+Hints", "+LoRA (full)"]
termacc = [d["safety"]["term_accuracy"]*100 for d in (b, p, a)]
chrf = [d["chrF++"] for d in (b, p, a)]
fig, ax = plt.subplots(figsize=(5.2, 3.4))
x = range(3); w = 0.38
bars1 = ax.bar([i-w/2 for i in x], termacc, w, label="TermAcc (%)", color="#2c7fb8")
bars2 = ax.bar([i+w/2 for i in x], chrf, w, label="chrF++", color="#d95f0e")
for bb_ in list(bars1)+list(bars2):
    ax.text(bb_.get_x()+bb_.get_width()/2, bb_.get_height()+0.8, f"{bb_.get_height():.1f}",
            ha="center", va="bottom", fontsize=9)
ax.set_xticks(list(x)); ax.set_xticklabels(sys_names)
ax.set_ylim(0, 112); ax.set_ylabel("score")
# legend on top, horizontal, clear of the bars (caption carries the takeaway)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=2, frameon=False)
save(fig, "fig_ablation")


# ---- Fig 2: training loss ----
# Use the LAST checkpoint (highest step) so log_history is the full run, not a
# truncated prefix -- next(glob(...)) was grabbing an early checkpoint.
ckpts = list((HERE/"results"/"agribanglat5"/"outputs"/"agribanglat5-lora").glob("checkpoint-*/trainer_state.json"))
ts = max(ckpts, key=lambda p: int(p.parent.name.split("-")[1]))
hist = json.loads(ts.read_text(encoding="utf-8"))["log_history"]
tr = sorted((h["epoch"], h["loss"]) for h in hist if "loss" in h)
ev = sorted((h["epoch"], h["eval_loss"]) for h in hist if "eval_loss" in h)
fig, ax = plt.subplots(figsize=(4.6, 3.2))
ax.plot(*zip(*tr), "o-", color="#2c7fb8", label="train loss (running)")
ax.plot(*zip(*ev), "s--", color="#d95f0e", label="held-out loss")
ax.set_xlabel("epoch"); ax.set_ylabel("cross-entropy loss")
ax.set_xticks(range(0, 4))
ax.legend(frameon=False)
save(fig, "fig_loss")


# ---- Fig 3: pipeline diagram ----
# Two document sources -> corpus; the glossary feeds the fine-tune as TERM HINTS
# (it does not produce parallel pairs), so its arrow targets the LoRA step.
fig, ax = plt.subplots(figsize=(7.4, 3.0)); ax.axis("off")
ax.set_xlim(0, 10); ax.set_ylim(0, 4)
def box(x, y, w, h, text, fc="#eef5fb", ec="#2c7fb8"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.3))
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=9)
def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, color="#555", lw=1.2))
# document sources -> corpus
box(0.1, 2.7, 2.3, 0.8, "Bijoy/SutonnyMJ\nPDFs")
box(0.1, 1.3, 2.3, 0.8, "English agronomy\n(FAO/IRRI)")
# corpus -> fine-tune -> model
box(3.1, 1.55, 2.2, 0.9, "AgriEnBn\n(3,034 pairs)", fc="#fdf0e6", ec="#d95f0e")
box(6.0, 1.55, 2.0, 0.9, "LoRA\nfine-tune")
box(8.3, 1.55, 1.65, 0.9, "AgriBanglaT5", fc="#e8f5e9", ec="#2e7d32")
# glossary feeds the fine-tune as term hints (NOT the corpus)
box(5.6, 0.1, 2.8, 0.72, "Glossary hints (82)")
arrow(2.4, 3.0, 3.1, 2.25); arrow(2.4, 1.75, 3.1, 1.9)
arrow(5.3, 2.0, 6.0, 2.0); arrow(8.0, 2.0, 8.3, 2.0)
arrow(7.0, 0.82, 7.0, 1.55)
ax.text(4.2, 3.45, "back-translate / distil (NLLB)", fontsize=7.5, color="#555", ha="center")
ax.text(7.15, 1.18, "hints", fontsize=7.5, color="#555", ha="left", va="center")
save(fig, "fig_pipeline")

print("All figures ->", FIG)
