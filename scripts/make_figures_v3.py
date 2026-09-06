"""Generate updated paper figures (v3) — addresses Reviewer 2 camera-ready feedback.

Changes vs v2 pipeline diagram:
  * Separate back-translation (BN→EN) and forward-distillation (EN→BN) paths
  * Explicit Bijoy→Unicode recovery step
  * LaBSE filtering step (≥0.70) shown
  * Dataset splits shown (Train / Dev / Test)
  * Sentence counts on each path

Also regenerates ablation and loss figures (unchanged data, polished style).

    python scripts/make_figures_v3.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Output into the paper figures directory (overwrites existing fig_pipeline.*)
FIG = Path(__file__).resolve().parent.parent / "research" / "paper_mt" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# ---- verified data (baked in) ----
sys_names = ["Baseline", "+Hints", "+LoRA (full)"]
termacc = [76.4, 91.2, 81.8]
chrf    = [48.1, 49.4, 54.2]
# training loss history (checkpoint-273)
tr = [(0.549, 3.5342), (1.099, 3.3001), (1.648, 3.0837), (2.198, 3.037), (2.747, 3.018)]
ev = [(1.0, 2.5595), (2.0, 2.4068), (3.0, 2.3578)]

plt.rcParams.update({
    "font.size": 11, "axes.labelsize": 11, "xtick.labelsize": 10,
    "ytick.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("wrote", FIG / f"{name}.pdf")


# ---- Fig 1: ablation (unchanged data, polished style) ----
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


# ---- Fig 2: training loss (unchanged data, polished style) ----
fig, ax = plt.subplots(figsize=(4.8, 3.4))
ax.yaxis.grid(True, linestyle='--', alpha=0.5); ax.set_axisbelow(True)
ax.plot(*zip(*tr), "o-", color="#2c7fb8", label="train loss", linewidth=2, markersize=7)
ax.plot(*zip(*ev), "s--", color="#d95f0e", label="held-out loss", linewidth=2, markersize=7)
ax.set_xlabel("epoch"); ax.set_ylabel("cross-entropy loss")
ax.set_xticks(range(0, 4))
ax.legend(frameon=True, edgecolor="black", fancybox=False)
save(fig, "fig_loss")


# ---- Fig 3: UPDATED pipeline diagram (Reviewer 2 feedback) ----
# Shows: separate BT/distill paths, Bijoy recovery, LaBSE filter, splits
fig, ax = plt.subplots(figsize=(9.0, 5.2))
ax.axis("off")
ax.set_xlim(0, 12)
ax.set_ylim(0, 6.5)

# Colour palette
C_SRC   = ("#eef5fb", "#2c7fb8")   # source documents (blue)
C_PROC  = ("#f5f0ff", "#6a3d9a")   # processing steps (purple)
C_CORP  = ("#fdf0e6", "#d95f0e")   # corpus (orange)
C_MODEL = ("#e8f5e9", "#2e7d32")   # final model (green)
C_GLOSS = ("#e3f2fd", "#1565c0")   # glossary (dark blue)
C_EVAL  = ("#fff3e0", "#e65100")   # eval (deep orange)


def box(x, y, w, h, text, fc="#eef5fb", ec="#2c7fb8", fontsize=9.5, bold_first=False):
    # drop shadow
    ax.add_patch(FancyBboxPatch((x + 0.06, y - 0.06), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc="#d9d9d9", ec="none", alpha=0.5))
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, linespacing=1.3)


def arrow(x1, y1, x2, y2, color="#444", lw=1.4):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                 arrowstyle="-|>", mutation_scale=14,
                                 color=color, lw=lw))


def label(x, y, text, fontsize=8, color="#555", ha="center", style="italic"):
    ax.text(x, y, text, fontsize=fontsize, color=color, ha=ha, style=style)


# ============ ROW 1 (top): Source documents ============
# Bijoy PDFs (top-left)
box(0.2, 5.0, 2.2, 0.9,
    "$\\bf{Bijoy/SutonnyMJ}$\nPDFs (gov. docs)", *C_SRC)

# Bijoy recovery step
box(3.1, 5.0, 2.2, 0.9,
    "$\\bf{Font{\\text -}aware}$\n$\\bf{recovery}$\n→ Unicode", *C_PROC, fontsize=8.5)
arrow(2.4, 5.45, 3.1, 5.45)
label(2.75, 5.75, "PyMuPDF +\nbijoy2unicode", fontsize=7.5)

# Recovered Bengali sentences
box(6.0, 5.0, 2.4, 0.9,
    "2,408 Bengali\nsentences", *C_SRC)
arrow(5.3, 5.45, 6.0, 5.45)
label(5.65, 5.75, "98.4% purity", fontsize=7.5)

# ============ ROW 2 (middle): Translation paths ============
# Back-translation path (BN→EN via NLLB)
box(6.0, 3.6, 2.4, 0.9,
    "$\\bf{Back{\\text -}translate}$\nBN → EN (NLLB)", *C_PROC)
arrow(7.2, 5.0, 7.2, 4.5)
label(7.55, 4.7, "BN→EN", fontsize=7.5, ha="left")

# English agronomy (left side, row 2)
box(0.2, 3.6, 2.2, 0.9,
    "$\\bf{English\\ agronomy}$\nFAO / IRRI\n(1,062 sent.)", *C_SRC)

# Forward distillation path (EN→BN via NLLB)
box(3.1, 3.6, 2.2, 0.9,
    "$\\bf{Distillation}$\nEN → BN (NLLB)", *C_PROC)
arrow(2.4, 4.05, 3.1, 4.05)
label(2.75, 3.4, "EN→BN", fontsize=7.5)

# ============ ROW 3: Filtering & corpus ============
# LaBSE filter
box(3.5, 2.0, 2.6, 0.9,
    "$\\bf{LaBSE\\ filter}$\n(cosine ≥ 0.70)", *C_PROC)
# Arrows from BT and distill into filter
arrow(4.2, 3.6, 4.6, 2.9)   # distill → filter
arrow(7.2, 3.6, 5.4, 2.9)   # BT → filter

# AgriEnBn corpus box
box(7.0, 2.0, 2.5, 0.9,
    "$\\bf{AgriEnBn}$\n3,034 pairs", *C_CORP, fontsize=10)
arrow(6.1, 2.45, 7.0, 2.45)

# Split labels beneath the corpus box
label(8.25, 1.75, "Train: 2,882  |  Dev: 152  |  Test: 198 (gold)", fontsize=7.5, color="#666")

# ============ ROW 4 (bottom): Training & model ============
# LoRA fine-tune
box(7.0, 0.4, 2.0, 0.9,
    "$\\bf{LoRA}$\nfine-tune\n(r=16, q/v)", *C_PROC, fontsize=9)
arrow(8.0, 2.0, 8.0, 1.3)

# AgriBanglaT5 output
box(9.6, 0.4, 2.2, 0.9,
    "$\\bf{AgriBanglaT5}$\n(247M)", *C_MODEL, fontsize=10)
arrow(9.0, 0.85, 9.6, 0.85)

# Glossary hints
box(3.5, 0.4, 2.6, 0.9,
    "$\\bf{Glossary}$ (82 terms)\nEN ↔ BN", *C_GLOSS)
arrow(6.1, 0.85, 7.0, 0.85)
label(6.55, 1.1, "⟦hints⟧", fontsize=8, color="#1565c0")

save(fig, "fig_pipeline")

print(f"\nAll figures → {FIG.resolve()}")
