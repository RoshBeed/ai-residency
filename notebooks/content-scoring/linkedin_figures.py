"""Two diagrams for a LinkedIn post: the three architectures, and what separates them.

    uv run python ../../notebooks/content-scoring/linkedin_figures.py

Reads the recovered interaction medians rather than recomputing, so the post's numbers and
the write-up's cannot disagree. Sized for a phone: few elements, large type, one idea each.
"""

# ruff: noqa: RUF001  -- the multiplication sign and rho are figure labels, not typos.

import json
import sys
from pathlib import Path

sys.path.insert(0, "services/content-scoring")
sys.path.insert(0, "libs/common/src")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

from ai_residency_common.figures import (
    AQUA,
    BLUE,
    INK,
    INK_SOFT,
    MUTED,
    ORANGE,
    RULE,
    apply_style,
    save_figure,
)

OUT = Path("services/content-scoring/assets/linkedin")

# ---------------------------------------------------------------- diagram 1
apply_style()
fig, axes = plt.subplots(1, 3, figsize=(12, 4.6))
titles = [
    ("EARLY FUSION", "concatenate first,\nthen one network", BLUE),
    ("LATE FUSION\n(shared trunk)", "a tower each, then a\nnetwork they share", BLUE),
    ("LATE FUSION\n(score level)", "a tower each down to\none score, then add", ORANGE),
]
for ax, (name, sub, _) in zip(axes, titles, strict=True):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title(name, fontsize=13, fontweight="bold", color=INK, pad=6)
    ax.text(5, 9.0, sub, ha="center", va="top", fontsize=9.5, color=INK_SOFT)


def box(ax, x, y, w, h, label, fc, fg="white", fs=9):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor="none", zorder=2))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", color=fg, fontsize=fs, zorder=3)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=11,
            color=MUTED,
            linewidth=1.2,
            zorder=1,
        )
    )


# early
a = axes[0]
box(a, 0.6, 1.0, 3.6, 1.0, "title", "#e8f0fb", INK)
box(a, 5.2, 1.0, 3.6, 1.0, "metadata", "#e8f0fb", INK)
box(a, 0.6, 3.4, 8.2, 1.2, "one network over both", BLUE)
box(a, 3.1, 6.2, 3.2, 1.0, "score", ORANGE)
arrow(a, 2.4, 2.0, 3.5, 3.4)
arrow(a, 7.0, 2.0, 5.9, 3.4)
arrow(a, 4.7, 4.6, 4.7, 6.2)

# late-deep
b = axes[1]
box(b, 0.6, 1.0, 3.6, 1.0, "title", "#e8f0fb", INK)
box(b, 5.2, 1.0, 3.6, 1.0, "metadata", "#e8f0fb", INK)
box(b, 0.6, 3.0, 3.6, 1.0, "tower", BLUE)
box(b, 5.2, 3.0, 3.6, 1.0, "tower", BLUE)
box(b, 0.6, 5.0, 8.2, 1.0, "shared network", BLUE)
box(b, 3.1, 7.0, 3.2, 1.0, "score", ORANGE)
arrow(b, 2.4, 2.0, 2.4, 3.0)
arrow(b, 7.0, 2.0, 7.0, 3.0)
arrow(b, 2.4, 4.0, 3.5, 5.0)
arrow(b, 7.0, 4.0, 5.9, 5.0)
arrow(b, 4.7, 6.0, 4.7, 7.0)

# late-snoek
c = axes[2]
box(c, 0.6, 1.0, 3.6, 1.0, "title", "#e8f0fb", INK)
box(c, 5.2, 1.0, 3.6, 1.0, "metadata", "#e8f0fb", INK)
box(c, 0.6, 3.0, 3.6, 1.0, "tower", ORANGE)
box(c, 5.2, 3.0, 3.6, 1.0, "tower", ORANGE)
box(c, 1.4, 5.0, 2.0, 0.9, "score", ORANGE, fs=8)
box(c, 6.0, 5.0, 2.0, 0.9, "score", ORANGE, fs=8)
box(c, 3.1, 7.0, 3.2, 1.0, "add them", ORANGE)
arrow(c, 2.4, 2.0, 2.4, 3.0)
arrow(c, 7.0, 2.0, 7.0, 3.0)
arrow(c, 2.4, 4.0, 2.4, 5.0)
arrow(c, 7.0, 4.0, 7.0, 5.0)
arrow(c, 2.4, 5.9, 3.9, 7.0)
arrow(c, 7.0, 5.9, 5.5, 7.0)
c.text(
    5,
    0.1,
    "cannot represent title × metadata",
    ha="center",
    fontsize=9,
    color=ORANGE,
    fontweight="bold",
)
axes[0].text(
    5, 0.1, "sees every interaction", ha="center", fontsize=9, color=BLUE, fontweight="bold"
)
axes[1].text(
    5, 0.1, "sees every interaction", ha="center", fontsize=9, color=BLUE, fontweight="bold"
)

fig.suptitle(
    '"Late fusion" names two different things — and only one has the problem',
    fontsize=14,
    fontweight="bold",
    y=0.99,
)
fig.tight_layout(rect=(0, 0, 1, 0.94))
save_figure(fig, OUT / "01-architectures.png", tight=False)
plt.close(fig)

# ---------------------------------------------------------------- diagram 2
records = json.loads(
    Path("services/content-scoring/assets/fusion-interaction-medians.json").read_text()
)
keys = sorted(records, key=float)  # the JSON keys are "0.0", "0.125", "1.0" ...
alphas = [float(k) for k in keys]


def med(k, key):
    return records[k][key]["median"]


apply_style()
fig, ax = plt.subplots(figsize=(9, 5.6))
for key, label, colour, style in (
    ("early@64", "Early fusion", BLUE, "-"),
    ("late-deep@64", "Late fusion — shared trunk", AQUA, "--"),
    ("late-snoek@64", "Late fusion — score level", ORANGE, "-"),
):
    ax.plot(
        alphas,
        [med(k, key) for k in keys],
        color=colour,
        linewidth=2.6,
        linestyle=style,
        marker="o",
        markersize=5,
        label=label,
    )
ax.axvline(0.75, color=RULE, linewidth=1.2, linestyle=(0, (4, 3)))
ax.text(0.85, 0.688, "they diverge here", fontsize=10, color=INK_SOFT)
ax.annotate(
    "",
    xy=(8, 0.796),
    xytext=(8, 0.678),
    arrowprops=dict(arrowstyle="<->", color=MUTED, linewidth=1.4),
)
ax.text(7.75, 0.737, "0.118 apart", ha="right", fontsize=11, color=INK, fontweight="bold")
ax.set_xlabel("how much of the signal needs BOTH inputs together  →", fontsize=11)
ax.set_ylabel("accuracy (Spearman ρ)", fontsize=11)
ax.set_title(
    "Score-level late fusion cannot learn what needs both inputs at once",
    fontsize=13,
    fontweight="bold",
    loc="left",
)
ax.text(
    0.1,
    0.7165,
    "no interaction:\neach architecture ties",
    fontsize=9.5,
    color=INK_SOFT,
    va="bottom",
)
ax.legend(fontsize=10.5, loc="upper left")
fig.tight_layout()
save_figure(fig, OUT / "02-interaction.png", tight=False)
plt.close(fig)
print("wrote both diagrams")
