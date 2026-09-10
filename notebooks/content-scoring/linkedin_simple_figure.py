"""One diagram for a non-ML audience: two ways to build the predictor, and why it matters."""

import sys
from pathlib import Path

sys.path.insert(0, "libs/common/src")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

from ai_residency_common.figures import BLUE, INK, INK_SOFT, MUTED, ORANGE, apply_style, save_figure

OUT = Path("services/content-scoring/assets/linkedin")
apply_style()
fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))


def box(ax, x, y, w, h, label, fc, fg="white", fs=10):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor="none", zorder=2))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", color=fg, fontsize=fs, zorder=3)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=13,
            color=MUTED,
            linewidth=1.4,
            zorder=1,
        )
    )


for ax in axes:
    ax.set_xlim(0, 10)
    ax.set_ylim(-1.8, 10)
    ax.axis("off")

a = axes[0]
a.set_title("Modular\nscore the parts, add them up", fontsize=13, fontweight="bold", color=INK)
box(a, 0.4, 0.8, 4.0, 1.1, "the title", "#e8f0fb", INK)
box(a, 5.6, 0.8, 4.0, 1.1, "when / where / who", "#e8f0fb", INK)
box(a, 0.4, 3.2, 4.0, 1.1, "model A", BLUE)
box(a, 5.6, 3.2, 4.0, 1.1, "model B", BLUE)
box(a, 3.0, 5.6, 4.0, 1.1, "add the scores", BLUE)
box(a, 3.4, 8.0, 3.2, 1.1, "upvotes", ORANGE)
arrow(a, 2.4, 1.9, 2.4, 3.2)
arrow(a, 7.6, 1.9, 7.6, 3.2)
arrow(a, 2.4, 4.3, 4.0, 5.6)
arrow(a, 7.6, 4.3, 6.0, 5.6)
arrow(a, 5.0, 6.7, 5.0, 8.0)
a.text(
    5,
    0.0,
    "nicer to operate\nbut it can only ever ADD the two scores",
    ha="center",
    va="top",
    fontsize=9.5,
    color=BLUE,
    fontweight="bold",
)

b = axes[1]
b.set_title("Monolithic\none model sees everything", fontsize=13, fontweight="bold", color=INK)
box(b, 0.4, 0.8, 4.0, 1.1, "the title", "#e8f0fb", INK)
box(b, 5.6, 0.8, 4.0, 1.1, "when / where / who", "#e8f0fb", INK)
box(b, 0.4, 4.2, 9.2, 1.4, "one model, both inputs together", ORANGE)
box(b, 3.4, 8.0, 3.2, 1.1, "upvotes", ORANGE)
arrow(b, 2.4, 1.9, 3.6, 4.2)
arrow(b, 7.6, 1.9, 6.4, 4.2)
arrow(b, 5.0, 5.6, 5.0, 8.0)
b.text(
    5,
    0.0,
    "harder to operate\nbut it wins here — titles and timing are coupled",
    ha="center",
    va="top",
    fontsize=9.5,
    color=ORANGE,
    fontweight="bold",
)

fig.suptitle(
    "Predicting Hacker News upvotes: is there a seam to cut along?",
    fontsize=14,
    fontweight="bold",
    y=0.98,
)
fig.text(
    0.5,
    0.015,
    "A good title doesn't ADD upvotes, it MULTIPLIES them — so adding two scores "
    'cannot express "this title, at this time".',
    ha="center",
    fontsize=10.5,
    color=INK_SOFT,
)
fig.tight_layout(rect=(0, 0.05, 1, 0.94))
save_figure(fig, OUT / "00-modular-vs-monolith.png", tight=False)
plt.close(fig)
print("wrote the diagram")
