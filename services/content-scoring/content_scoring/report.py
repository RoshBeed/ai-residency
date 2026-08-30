"""Figures for the write-up.

Every number here comes from `evaluate.py` or from the W&B run history. Nothing is
recomputed: a figure that disagreed with the README table would be worse than no figure.

Palette and axis treatment match `word-embeddings/report.py`, so the two write-ups read
as one portfolio rather than two. The four-slot categorical set (blue, orange, aqua,
violet) was validated on the all-pairs list, which is what a scatter needs -- the
adjacent-pairs default is only enough for bars and lines. Aqua sits below 3:1 against
the surface, so every chart that uses it also carries direct value labels.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ai_residency_common import get_logger
from config import Config

__all__ = [
    "main",
    "plot_ablation",
    "plot_capacity",
    "plot_fusion_comparison",
    "plot_score_distribution",
    "plot_slices",
]

logger = get_logger(__name__)

BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
INK, INK_SOFT, MUTED, RULE = "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
SURFACE = "#fcfcfb"
SERIES = (BLUE, ORANGE, AQUA, VIOLET)

# Resolved from this file, not from the working directory. Every other path in the
# service is relative to the repo root because that is where `make` runs, but the
# figures are also written by hand from inside the service directory -- and a cwd
# mistake here creates `services/content-scoring/services/content-scoring/assets`
# rather than failing.
ASSETS = Path(__file__).resolve().parent.parent / "assets"

LABELS = {
    "text_only": "Text only",
    "features_only": "Features only",
    "early": "Early fusion",
    "late": "Late fusion",
    "global_mean": "Predict the mean",
    "hour_mean": "Hour mean",
    "domain_mean": "Domain mean",
    "tfidf_ridge": "TF-IDF + Ridge",
}


def _style() -> None:
    """Recessive axes and grid, so the data is the only assertive thing on the page."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": RULE,
            "axes.labelcolor": INK_SOFT,
            "axes.titlecolor": INK,
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": RULE,
            "grid.alpha": 0.4,
            "grid.linewidth": 0.6,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "font.size": 10,
            "legend.frameon": False,
        }
    )


def plot_fusion_comparison(report: dict[str, dict[str, float]], output: Path) -> None:
    """The four architectures against the four baselines, on two metrics.

    Args:
        report: `evaluate.py --json` output, model name to its metric block.
        output: Where to write the PNG.

    Two panels rather than two y-axes on one. Spearman and NDCG@10 have different
    ranges and mean different things; overlaying them on a shared axis would invite
    exactly the comparison that is not valid.
    """
    _style()
    baselines = [
        name
        for name in ("global_mean", "hour_mean", "domain_mean", "tfidf_ridge")
        if name in report
    ]
    models = [name for name in ("text_only", "features_only", "early", "late") if name in report]
    order = baselines + models

    figure, panels = plt.subplots(1, 2, figsize=(11, 4.4))
    for panel, metric, title in (
        (panels[0], "spearman", "Spearman ρ (whole test split)"),
        (panels[1], "ndcg@10", "NDCG@10 (ranked within each day)"),
    ):
        raw = [report[name][metric] for name in order]
        # Predict-the-mean has no rank correlation at all -- a constant prediction
        # cannot order anything -- so its Spearman is NaN rather than 0. Drawn as an
        # empty bar labelled "n/a": dropping the row would hide the floor, and plotting
        # it as 0.0 would assert something the metric does not say.
        values = [0.0 if value != value else value for value in raw]
        largest = max(values) or 1.0
        colours = [MUTED] * len(baselines) + list(SERIES[: len(models)])
        bars = panel.barh(
            [LABELS.get(name, name) for name in order], values, color=colours, height=0.62
        )
        for bar, original in zip(bars, raw, strict=True):
            panel.text(
                bar.get_width() + largest * 0.015,
                bar.get_y() + bar.get_height() / 2,
                "n/a" if original != original else f"{original:.3f}",
                va="center",
                fontsize=9,
                color=INK_SOFT,
            )
        panel.set_title(title, fontsize=11, loc="left")
        panel.set_xlim(0, largest * 1.18)
        panel.invert_yaxis()
        panel.grid(axis="y", visible=False)

    figure.suptitle(
        "Four architectures, four baselines, one held-out test period",
        fontsize=12,
        x=0.02,
        ha="left",
    )
    figure.tight_layout()
    figure.savefig(output, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)
    logger.info("Wrote %s", output)


def plot_score_distribution(config: Config, output: Path) -> None:
    """The long tail, and what `log1p` does to it.

    Args:
        config: Supplies the split paths.
        output: Where to write the PNG.

    The single most useful picture in the write-up: it is the reason the target is
    logged and the reason a squared error on the raw scale would train a model to
    predict the tail and ignore everything else.
    """
    _style()
    scores = pd.read_parquet(config.split_path("train"), columns=["score"])["score"].to_numpy()

    figure, panels = plt.subplots(1, 2, figsize=(11, 3.8))
    # Log-spaced bins on a log axis, rather than clipping the tail into the last bin.
    # Clipping at 200 drew a spike there that reads as "thousands of posts scored exactly
    # 200" -- an artefact of the chart inventing a mode the data does not have.
    edges = np.logspace(0, np.log10(scores.max()), 60)
    panels[0].hist(scores, bins=edges, color=BLUE, edgecolor="none")
    panels[0].set_xscale("log")
    panels[0].set_yscale("log")
    panels[0].set_title("Raw score", fontsize=11, loc="left")
    panels[0].set_xlabel("upvotes (log scale)")
    panels[0].set_ylabel("posts (log scale)")

    panels[1].hist(np.log1p(scores), bins=60, color=ORANGE, edgecolor="none")
    panels[1].set_title("log1p(score) — what the models predict", fontsize=11, loc="left")
    panels[1].set_xlabel("log1p(upvotes)")
    panels[1].set_ylabel("posts")

    median = float(np.median(scores))
    panels[0].axvline(median, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    panels[0].text(
        median * 1.25,
        panels[0].get_ylim()[1] * 0.3,
        f"median {median:.0f}",
        fontsize=9,
        color=MUTED,
    )

    figure.suptitle(
        f"{len(scores):,} training posts: median {median:.0f}, mean {scores.mean():.1f}, "
        f"max {scores.max():,}",
        fontsize=12,
        x=0.02,
        ha="left",
    )
    figure.tight_layout()
    figure.savefig(output, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)
    logger.info("Wrote %s", output)


def plot_ablation(rows: list[list[Any]], output: Path) -> None:
    """What each feature group contributes to the late-fusion model.

    Args:
        rows: `(group, spearman, delta)` from `leave_one_group_out`.
        output: Where to write the PNG.
    """
    if not rows:
        logger.warning("No ablation rows; skipping %s", output)
        return
    _style()
    ablated = [row for row in rows if row[0] != "(none)"]
    names = [LABELS.get(str(row[0]), str(row[0]).replace("_", " ")) for row in ablated]
    deltas = [float(row[2]) for row in ablated]

    figure, axes = plt.subplots(figsize=(7.5, 3.8))
    # Drops are the finding, so they carry the hue; a group whose removal *helps* is
    # drawn in the second slot rather than in a status colour -- this is not an error
    # state, it is a result.
    colours = [BLUE if delta <= 0 else ORANGE for delta in deltas]
    bars = axes.barh(names, deltas, color=colours, height=0.6)
    for bar, delta in zip(bars, deltas, strict=True):
        axes.text(
            delta + (0.001 if delta >= 0 else -0.001),
            bar.get_y() + bar.get_height() / 2,
            f"{delta:+.3f}",
            va="center",
            ha="left" if delta >= 0 else "right",
            fontsize=9,
            color=INK_SOFT,
        )
    axes.axvline(0, color=MUTED, linewidth=1)
    axes.set_xlabel("change in Spearman ρ when this branch is zeroed")
    axes.set_title("Leave one group out, late fusion", fontsize=12, loc="left")
    axes.invert_yaxis()
    axes.grid(axis="y", visible=False)
    figure.tight_layout()
    figure.savefig(output, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)
    logger.info("Wrote %s", output)


def plot_slices(rows: list[list[Any]], output: Path) -> None:
    """Where the model works and where it does not.

    Args:
        rows: `(family, bucket, n, spearman, mean_score)` from `slice_table`.
        output: Where to write the PNG.

    One panel per slice family. A single ρ over 69,275 posts hides all of this, and the
    OOV panel is the one that says whether extending the vocabulary paid for itself.
    """
    if not rows:
        logger.warning("No slice rows; skipping %s", output)
        return
    _style()
    families: dict[str, list[list[Any]]] = {}
    for row in rows:
        families.setdefault(str(row[0]), []).append(row)

    figure, panels = plt.subplots(1, len(families), figsize=(3.2 * len(families), 3.6), sharey=True)
    panel_list = panels if isinstance(panels, np.ndarray) else [panels]
    for panel, (family, family_rows), colour in zip(
        panel_list, families.items(), list(SERIES) * 3, strict=False
    ):
        names = [str(row[1]).replace("_", " ") for row in family_rows]
        values = [float(row[3]) for row in family_rows]
        panel.bar(names, values, color=colour, width=0.62)
        for index, value in enumerate(values):
            panel.text(
                index, value + 0.005, f"{value:.2f}", ha="center", fontsize=8, color=INK_SOFT
            )
        panel.set_title(family.replace("_", " "), fontsize=10, loc="left")
        panel.tick_params(axis="x", rotation=35)
        panel.grid(axis="x", visible=False)
    panel_list[0].set_ylabel("Spearman ρ")

    figure.suptitle("Late fusion, sliced", fontsize=12, x=0.02, ha="left")
    figure.tight_layout()
    figure.savefig(output, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)
    logger.info("Wrote %s", output)


def plot_capacity(capacity: dict[str, dict[str, list[Any]]], output: Path) -> None:
    """Early versus late fusion as depth grows.

    Args:
        capacity: Layer count to `{"early": [mean, spread, params], "late": [...]}`.
        output: Where to write the PNG.

    Drawn against measured parameter count, not layer count. At equal width and depth
    late fusion has five branches and a head where early has one MLP -- roughly four
    times the parameters -- so a curve against layers would silently credit early fusion
    for being the smaller model and call it an architecture effect.
    """
    if not capacity:
        logger.warning("No capacity rows; skipping %s", output)
        return
    _style()
    layers = sorted(int(k) for k in capacity)
    figure, panels = plt.subplots(1, 2, figsize=(11, 4.0))

    for name, colour in (("early", ORANGE), ("late", BLUE)):
        means = [capacity[str(layer)][name][0] for layer in layers]
        spreads = [capacity[str(layer)][name][1] for layer in layers]
        params = [capacity[str(layer)][name][2] / 1000 for layer in layers]
        label = LABELS[name]
        panels[0].errorbar(
            layers,
            means,
            yerr=spreads,
            color=colour,
            linewidth=2,
            marker="o",
            markersize=7,
            capsize=3,
            label=label,
        )
        panels[1].errorbar(
            params,
            means,
            yerr=spreads,
            color=colour,
            linewidth=2,
            marker="o",
            markersize=7,
            capsize=3,
            label=label,
        )

    panels[0].set_xlabel("hidden layers per branch")
    panels[0].set_xticks(layers)
    panels[0].set_ylabel("test Spearman ρ")
    panels[0].set_title("The crossover", fontsize=11, loc="left")
    panels[1].set_xlabel("trainable parameters (thousands)")
    panels[1].set_title("Against what the models actually cost", fontsize=11, loc="left")
    for panel in panels:
        panel.legend(loc="lower left")

    figure.suptitle(
        "Late fusion leads where capacity is scarce; early fusion degrades more slowly",
        fontsize=12,
        x=0.02,
        ha="left",
    )
    figure.tight_layout()
    figure.savefig(output, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)
    logger.info("Wrote %s", output)


def main() -> None:
    """Write every figure from an evaluation report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        default="services/content-scoring/assets/evaluation.json",
        help="Output of `python -m content_scoring.evaluate --json <path>`.",
    )
    parser.add_argument(
        "--capacity",
        default="services/content-scoring/assets/capacity.json",
        help="Depth-sweep results, for the crossover figure.",
    )
    arguments = parser.parse_args()
    config = Config()
    ASSETS.mkdir(parents=True, exist_ok=True)

    plot_score_distribution(config, ASSETS / "score-distribution.png")

    path = Path(arguments.report)
    if not path.exists():
        logger.warning(
            "%s does not exist; run `make evaluate` with --json first for the rest", path
        )
        return
    with path.open(encoding="utf-8") as handle:
        report = json.load(handle)

    plot_fusion_comparison(report["metrics"], ASSETS / "fusion-comparison.png")
    capacity_path = Path(arguments.capacity)
    if capacity_path.exists():
        with capacity_path.open(encoding="utf-8") as handle:
            plot_capacity(json.load(handle), ASSETS / "fusion-capacity.png")
    plot_slices(report.get("slices", []), ASSETS / "slice-performance.png")
    plot_ablation(report.get("ablation", []), ASSETS / "feature-ablation.png")


if __name__ == "__main__":
    main()
