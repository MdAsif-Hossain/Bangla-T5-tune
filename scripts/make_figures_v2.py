"""Generate the upgraded paper figures into ./figures_v2/.

SELF-CONTAINED: all data is hard-coded below (the verified v1 numbers), so this
runs anywhere matplotlib works -- e.g. paste into a Google Colab / Kaggle cell.
(Local runs are blocked by this machine's Application Control policy on
matplotlib's compiled DLLs, hence the standalone form.)

    python scripts/make_figures_v2.py     # if matplotlib works locally
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG = Path("figures_v2")          # created in the current working directory
FIG.mkdir(parents=True, exist_ok=True)

# ---- verified v1 data (baked in) ----
sys_names = ["Baseline", "+Hints", "+LoRA (full)"]
termacc = [76.4, 91.2, 81.8]
chrf    = [48.1, 49.4, 54.2]
# training loss history (checkpoint-273)
tr = [(0.549, 3.5342), (1.099, 3.3001), (1.648, 3.0837), (2.198, 3.037), (2.747, 3.018)]
ev = [(1.0, 2.5595), (2.0, 2.4068), (3.0, 2.3578)]

plt.rcParams.update({
    "font.size": 12, "axes.labelsize": 12, "xtick.labelsize": 11,
    "ytick.labelsize": 11, "axes.spines.top": False, "axes.spines.right": False,
})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("wrote", name)


# ---- Fig 1: ablation ----
fig, ax = plt.subplots(figsize=(5.4, 3.6))
x = range(3); w = 0.38
ax.yaxis.grid(True, linestyle='--', alpha=0.6); ax.set_axisbelow(True)
bars1 = ax.bar([i-w/2 for i in x], termacc, w, label="TermAcc (%)",
               color="#2c7fb8", edgecolor="black", linewidth=1.2)
bars2 = ax.bar([i+w/2 for i in x], chrf, w, label="chrF++",
               color="#d95f0e", edgecolor="black", linewidth=1.2, hatch="///")
for bb_ in list(bars1)+list(bars2):
    ax.text(bb_.get_x()+bb_.get_width()/2, bb_.get_height()+1.5, f"{bb_.get_height():.1f}",
            ha="center", va="bottom", fontsize=10, fontweight='bold', color="#333333")
ax.set_xticks(list(x)); ax.set_xticklabels(sys_names)
ax.set_ylim(0, 115); ax.set_ylabel("score")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False)
save(fig, "fig_ablation")


# ---- Fig 2: training loss ----
fig, ax = plt.subplots(figsize=(4.8, 3.4))
ax.yaxis.grid(True, linestyle='--', alpha=0.5); ax.set_axisbelow(True)
ax.plot(*zip(*tr), "o-", color="#2c7fb8", label="train loss", linewidth=2, markersize=7)
ax.plot(*zip(*ev), "s--", color="#d95f0e", label="held-out loss", linewidth=2, markersize=7)
ax.set_xlabel("epoch"); ax.set_ylabel("cross-entropy loss")
ax.set_xticks(range(0, 4))
ax.legend(frameon=True, edgecolor="black", fancybox=False)
save(fig, "fig_loss")


# ---- Fig 3: pipeline diagram ----
fig, ax = plt.subplots(figsize=(7.6, 3.2)); ax.axis("off")
ax.set_xlim(0, 10); ax.set_ylim(0, 4)

def box(x, y, w, h, text, fc="#eef5fb", ec="#2c7fb8"):
    ax.add_patch(FancyBboxPatch((x+0.05, y-0.05), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc="#d9d9d9", ec="none", alpha=0.7))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.5))
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=10)

def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=15, color="#333", lw=1.5))

box(0.1, 2.7, 2.3, 0.8, "$\\bf{Bijoy/SutonnyMJ}$\nPDFs")
box(0.1, 1.3, 2.3, 0.8, "$\\bf{English\\ Agronomy}$\n(FAO/IRRI)")
box(3.1, 1.55, 2.2, 0.9, "$\\bf{AgriEnBn}$\n(3,034 pairs)", fc="#fdf0e6", ec="#d95f0e")
box(6.0, 1.55, 2.0, 0.9, "$\\bf{LoRA}$\nFine-tune")
box(8.3, 1.55, 1.65, 0.9, "$\\bf{AgriBanglaT5}$", fc="#e8f5e9", ec="#2e7d32")
box(5.6, 0.1, 2.8, 0.72, "$\\bf{Glossary\\ Hints}$ (82)")
arrow(2.4, 3.0, 3.1, 2.25); arrow(2.4, 1.75, 3.1, 1.9)
arrow(5.3, 2.0, 6.0, 2.0); arrow(8.0, 2.0, 8.3, 2.0)
arrow(7.0, 0.82, 7.0, 1.55)
ax.text(4.2, 3.45, "back-translate / distil (NLLB)", fontsize=8.5, color="#444", ha="center", style="italic")
ax.text(7.15, 1.18, "hints", fontsize=8.5, color="#444", ha="left", va="center", style="italic")
save(fig, "fig_pipeline")

print("All figures ->", FIG.resolve())
