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
ax.set_xlim(0, 19.5)
ax.set_ylim(0, 6.5)

# Colour palette
C_SRC   = ("#eef5fb", "#2c7fb8")   # source documents (blue)
C_PROC  = ("#f5f0ff", "#6a3d9a")   # processing steps (purple)
C_CORP  = ("#fdf0e6", "#d95f0e")   # corpus (orange)
C_MODEL = ("#e8f5e9", "#2e7d32")   # final model (green)
C_GLOSS = ("#e3f2fd", "#1565c0")   # glossary (dark blue)
C_EVAL  = ("#fff3e0", "#e65100")   # eval (deep orange)


def box(x, y, w, h, text, fc="#eef5fb", ec="#2c7fb8", fontsize=9.5, fw="bold"):
    # drop shadow
    ax.add_patch(FancyBboxPatch((x + 0.06, y - 0.06), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc="#d9d9d9", ec="none", alpha=0.5))
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, linespacing=1.3, fontweight=fw)


def arrow(x1, y1, x2, y2, color="#444", lw=1.4):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                 arrowstyle="-|>", mutation_scale=14,
                                 color=color, lw=lw))


def label(x, y, text, fontsize=8, color="#555", ha="center", va="baseline", style="italic"):
    ax.text(x, y, text, fontsize=fontsize, color=color, ha=ha, va=va, style=style)


# ============ ROW 1 (top): Source documents ============
# Bijoy PDFs (top-left)
box(0.5, 5.0, 3.2, 0.9,
    "Bijoy/SutonnyMJ\nPDFs (gov. docs)", *C_SRC, fontsize=8.0)

# Bijoy recovery step
box(6.8, 5.0, 3.2, 0.9,
    "Font-aware\nrecovery\n→ Unicode", *C_PROC, fontsize=8.5)
arrow(4.0, 5.45, 6.6, 5.45)
label(4.3, 5.55, "PyMuPDF +\nbijoy2unicode", fontsize=7.0, ha="left", va="bottom")

# Recovered Bengali sentences
box(13.1, 5.0, 3.2, 0.9,
    "2,408 Bengali\nsentences", *C_SRC, fw="normal", fontsize=8.5)
arrow(10.3, 5.45, 12.9, 5.45)
label(10.6, 5.55, "98.4% purity", fontsize=7.0, ha="left", va="bottom")

# ============ ROW 2 (middle): Translation paths ============
# Back-translation path (BN→EN via NLLB)
box(13.1, 3.6, 3.2, 0.9,
    "Back-translate\nBN → EN (NLLB)", *C_PROC, fontsize=8.5)
arrow(14.7, 5.0, 14.7, 4.5)
label(14.9, 4.75, "BN→EN", fontsize=7.0, ha="left", va="center")

# English agronomy (left side, row 2)
box(0.5, 3.6, 3.2, 0.9,
    "English agronomy\nFAO / IRRI\n(1,062 sent.)", *C_SRC, fontsize=8.0)

# Forward distillation path (EN→BN via NLLB)
box(6.8, 3.6, 3.2, 0.9,
    "Distillation\nEN → BN (NLLB)", *C_PROC, fontsize=8.5)
arrow(4.0, 4.05, 6.6, 4.05)
label(4.3, 4.15, "EN→BN", fontsize=7.0, ha="left", va="bottom")

# ============ ROW 3: Filtering & corpus ============
# Quality filter (LaBSE, dedup, len)
box(6.8, 2.0, 3.2, 0.9,
    "Quality filter\n(dedup, len,\nLaBSE ≥0.70)", *C_PROC, fontsize=7.5)
# Arrows from BT and distill into filter
arrow(8.4, 3.6, 8.4, 2.9)   # distill → filter
arrow(14.7, 3.6, 9.8, 2.9)   # BT → filter

# AgriEnBn corpus box
box(13.1, 2.0, 3.2, 0.9,
    "AgriEnBn\n3,034 pairs", *C_CORP, fontsize=9.5)
arrow(10.3, 2.45, 12.9, 2.45)

# Split labels beneath the corpus box (with white bbox to cut the arrow)
ax.text(14.7, 1.65, "Train: 2,883  |  Dev: 151  |  Test: 198 (gold)",
        fontsize=7.5, color="#666", ha="center", va="center", style="italic",
        bbox=dict(facecolor="white", edgecolor="none", pad=1.5), zorder=5)

# ============ ROW 4 (bottom): Training & model ============
# LoRA fine-tune
box(13.7, 0.4, 2.0, 0.9,
    "LoRA\nfine-tune\n(r=16, q/v)", *C_PROC, fontsize=8.5)
arrow(14.7, 2.0, 14.7, 1.3)

# AgriBanglaT5 output
box(16.5, 0.4, 2.6, 0.9,
    "AgriBanglaT5\n(247M)", *C_MODEL, fontsize=8.5)
arrow(15.9, 0.85, 16.3, 0.85)

# Glossary hints
box(6.8, 0.4, 3.2, 0.9,
    "Glossary\n(82 terms)\nEN ↔ BN", *C_GLOSS, fontsize=8.0)
arrow(10.3, 0.85, 13.5, 0.85)
label(11.5, 1.0, "⟦hints⟧", fontsize=7.5, color="#1565c0", va="bottom")

save(fig, "fig_pipeline")

print(f"\nAll figures -> {FIG.resolve()}")
