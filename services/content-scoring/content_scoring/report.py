"""Figures for the write-up.

Every number here comes from `evaluate.py` or from the W&B run history. Nothing is
recomputed: a figure that disagreed with the README table would be worse than no figure.

Palette, axis treatment and resolution come from `ai_residency_common.figures`, so
every write-up in the portfolio reads as one rather than nine. See that module for why
the DPI is a constant.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Rectangle

from ai_residency_common import get_logger
from ai_residency_common.figures import (
    BLUE,
    INK_SOFT,
    MUTED,
    ORANGE,
    RULE,
    SERIES,
    SURFACE,
    VIOLET,
    apply_style,
    save_figure,
)
from config import Config, Fusion
from content_scoring.model import build_model, describe, first_layer_blocks

__all__ = [
    "main",
    "plot_ablation",
    "plot_architectures",
    "plot_capacity",
    "plot_fusion_comparison",
    "plot_fusion_curve",
    "plot_interaction",
    "plot_score_distribution",
    "plot_seed_distribution",
    "plot_slices",
]

logger = get_logger(__name__)

# Resolved from this file, not from the working directory. Every other path in the
# service is relative to the repo root because that is where `make` runs, but the
# figures are also written by hand from inside the service directory -- and a cwd
# mistake here creates `services/content-scoring/services/content-scoring/assets`
# rather than failing.
ASSETS = Path(__file__).resolve().parent.parent / "assets"

# Width of the extended word vectors, for the architecture figure only.
EMBED_SIZE = 300

# What a reader calls each input group in a figure. `content_scoring.model.INPUT_LABELS`
# names the same five for the diagrams; these are the sentence-case forms.
GROUP_LABELS = {
    "text": "the title",
    "temporal": "when it was posted",
    "domain": "where it links",
    "title_surface": "the title's shape",
    "author": "the author's history",
}

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


def plot_fusion_comparison(report: dict[str, dict[str, float]], output: Path) -> None:
    """The four architectures against the four baselines, on two metrics.

    Args:
        report: `evaluate.py --json` output, model name to its metric block.
        output: Where to write the PNG.

    Two panels rather than two y-axes on one. Spearman and NDCG@10 have different
    ranges and mean different things; overlaying them on a shared axis would invite
    exactly the comparison that is not valid.
    """
    apply_style()
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
    save_figure(figure, output)
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
    apply_style()
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
    save_figure(figure, output)
    plt.close(figure)
    logger.info("Wrote %s", output)


def plot_ablation(rows: list[list[Any]], output: Path) -> None:
    """What each input group contributes to the dense model.

    Args:
        rows: `(group, spearman, delta)` from `leave_one_group_out`.
        output: Where to write the PNG.

    Ordered by size of the drop rather than by the order the groups happen to be
    declared in, because the ranking is the finding.
    """
    if not rows:
        logger.warning("No ablation rows; skipping %s", output)
        return
    apply_style()
    ablated = sorted((row for row in rows if row[0] != "(none)"), key=lambda row: float(row[2]))
    names = [GROUP_LABELS.get(str(row[0]), str(row[0]).replace("_", " ")) for row in ablated]
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
    axes.set_xlabel("change in Spearman ρ when this input is zeroed")
    axes.set_title("Leave one input out, early fusion", fontsize=12, loc="left")
    # Room for the longest label to sit inside the axes rather than on top of the tick.
    axes.set_xlim(min(deltas) * 1.28, max(0.0, max(deltas)) + 0.006)
    axes.grid(axis="y", visible=False)
    figure.tight_layout()
    save_figure(figure, output)
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
    apply_style()
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
    save_figure(figure, output)
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
    apply_style()
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
    save_figure(figure, output)
    plt.close(figure)
    logger.info("Wrote %s", output)


def plot_architectures(config: Config, output: Path) -> None:
    """Early against late: one layer differs, everything after it is shared.

    Args:
        config: Supplies the hidden width and the domain embedding size.
        output: Where to write the PNG.

    Both models are the same stack -- a first layer of 329 inputs onto 256 hidden units,
    a shared 256 → 256 layer, then the head -- and differ only in which of the first
    layer's weights exist. The top row is that layer as a group-by-block grid: early
    fills all twenty-five cells, late keeps the five on the diagonal. The bottom row is
    everything after it, drawn for both because it is identical in both, which is what
    makes the pair a controlled comparison rather than two different networks.

    Drawn as a grid rather than as the weight matrix itself because the title is 300 of
    the 329 inputs, so a to-scale matrix renders the four metadata groups as invisible
    slivers. The true widths are printed on the axes, so the even spacing hides nothing.

    Shapes are read off built models, so the figure cannot drift from the network. A stub
    embedding matrix is enough because only layer widths are drawn, not weights.
    """
    apply_style()
    stub = torch.zeros(64, EMBED_SIZE)
    spec: tuple[tuple[Fusion, str], ...] = (
        ("early", "one dense layer"),
        ("late", "the same layer, block-diagonal"),
    )

    figure, panels = plt.subplots(
        2, 2, figsize=(11, 4.8), height_ratios=[6, 0.55], gridspec_kw={"hspace": 0.5}
    )
    groups = first_layer_blocks(build_model(stub, 8, replace(config, fusion="late")))
    size = len(groups)

    for column, (fusion, subtitle) in enumerate(spec):
        model = build_model(stub, 8, replace(config, fusion=fusion))
        blocks = first_layer_blocks(model)
        facts = describe(model)
        top, bottom = panels[0][column], panels[1][column]

        # Rows are input groups, columns are the hidden units each group owns. A dense
        # layer connects every group to every column; the block-diagonal one connects
        # each group only to its own.
        grid = np.ones((size, size)) if len(blocks) == 1 else np.eye(size)
        _connectivity(top, grid, groups)
        weights = (
            sum(width for _, width, _ in groups) * sum(out for _, _, out in groups)
            if len(blocks) == 1
            else sum(width * out for _, width, out in groups)
        )
        top.set_title(
            f"{'Early' if fusion == 'early' else 'Late'} fusion — {subtitle}\n"
            f"{weights:,} first-layer weights",
            fontsize=10,
            pad=14,
        )

        bottom.imshow(
            np.ones((1, size)),
            cmap=matplotlib.colors.ListedColormap([SURFACE, VIOLET]),
            vmin=0,
            vmax=1,
            interpolation="nearest",
            aspect="auto",
        )
        bottom.grid(False)
        bottom.set_xticks([])
        bottom.set_yticks([])
        for spine in bottom.spines.values():
            spine.set_visible(False)
        bottom.text(
            (size - 1) / 2,
            0,
            f"shared {facts['shared_layers']}, then the head — identical in both",
            ha="center",
            va="center",
            fontsize=9,
            color="white",
        )

    panels[0][0].set_ylabel(
        "input group, and how many columns it brings", color=INK_SOFT, fontsize=9
    )

    save_figure(figure, output)
    plt.close(figure)
    logger.info("Wrote %s", output)


def _connectivity(
    panel: Any, grid: "np.ndarray[Any, Any]", groups: list[tuple[str, int, int]]
) -> None:
    """Draw one group-by-block connectivity grid onto a panel."""
    panel.imshow(
        grid,
        cmap=matplotlib.colors.ListedColormap([SURFACE, BLUE]),
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    for row in range(len(groups)):
        for column in range(len(groups)):
            panel.add_patch(
                Rectangle(
                    (column - 0.5, row - 0.5), 1, 1, fill=False, edgecolor=RULE, linewidth=0.8
                )
            )
    panel.grid(False)
    panel.set_xticks(range(len(groups)))
    panel.set_xticklabels([str(out) for _, _, out in groups], fontsize=8)
    panel.set_yticks(range(len(groups)))
    panel.set_yticklabels([f"{label}  ({width})" for label, width, _ in groups], fontsize=8.5)
    panel.set_xlabel("hidden units, by the group that owns them", color=INK_SOFT, fontsize=9)
    panel.tick_params(length=0)
    for spine in panel.spines.values():
        spine.set_visible(False)


def _curve_runs(project: str) -> list[dict[str, Any]]:
    """Every finished fusion-curve run, as flat records.

    Args:
        project: `entity/project` on W&B.

    Returns:
        One dict per run with `widths`, `fusion_layer`, `seed`, `val_mae` and `spearman`.

    Reads the *effective* fusion layer from the summary rather than the requested one
    from the config: a value past the end of a shape clamps, so grouping on the request
    would merge two different models into one point on the curve.

    **Only the grid's own runs count.** Three kinds of run share this project and two of
    them would corrupt the curve:

    - the Bayes search varies learning rate, dropout and width, so a point meant to hold
      ten identical runs would hold a hundred differently-tuned ones and the spread would
      describe the hyperparameter search rather than the seed;
    - the published training runs use the default shape, which *is* a point on the curve
      -- 94 of them landed on `256,256` at fusion layer 2 before this filter existed.

    `run.sweep` excludes the first. `save_checkpoints` and `promote` exclude the second:
    the grid writes nothing by design, and nothing else that writes nothing shares its
    shape. Both work on runs that have already finished, which matters because the filter
    was missing when the first 150 ran.
    """
    import wandb

    api = wandb.Api()
    records = []
    for run in api.runs(project):
        if run.state != "finished" or run.sweep is not None:
            continue
        summary, config = run.summary, run.config
        if config.get("save_checkpoints", True) or config.get("promote", True):
            continue
        widths = summary.get("model/hidden_widths")
        layer = summary.get("model/fusion_layer")
        if widths is None or layer is None:
            continue  # a run from before the fusion point was swept
        records.append(
            {
                "widths": str(widths),
                "fusion_layer": int(layer),
                "seed": int(config.get("seed", 0)),
                "val_mae": float(summary.get("best/val_mae", float("nan"))),
                "spearman": float(summary.get("test/spearman", float("nan"))),
            }
        )
    return records


# The grid, grouped by the question each comparison answers. Depths 2-6 pair a taper
# against a constant-width control, so a difference between them is the shape and not the
# depth; the width panel holds three depth-3 shapes that differ only in how wide the
# isolated blocks are.
CURVE_PANELS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("2 layers", ("320,256", "256,256")),
    ("3 layers", ("320,256,192", "256,256,256")),
    ("4 layers", ("320,256,192,128", "256,256,256,256")),
    ("5 layers", ("320,256,192,128,96", "256,256,256,256,256")),
    ("6 layers", ("320,256,192,128,96,32", "128,128,128,128,128,128")),
    ("3 layers, width varied", ("640,512,384", "256,256,256", "256,64,256", "64,64,64")),
)


def plot_fusion_curve(records: list[dict[str, Any]], output: Path) -> None:
    """Accuracy against the layer at which the input groups first meet.

    Args:
        records: `_curve_runs` output.
        output: Where to write the PNG.

    **Small multiples, not thirteen lines on one axis.** The categorical palette holds
    four hues and is never cycled, so thirteen shapes on shared axes would give four
    shapes the same colour and make the legend a guess. Each panel holds two to four
    series that are meant to be compared with each other, which is also the structure of
    the experiment: a taper against a constant-width control at each depth, and then the
    width probes that decide whether the effect is fusion or capacity.

    **Median and interquartile band, not mean and standard deviation.** Late fusion's
    seed spread reached 0.106 Spearman where early fusion's was 0.026, with runs landing
    in two clusters rather than around one centre. A mean would describe neither, and an
    error bar drawn from a standard deviation would imply a precision these runs do not
    have. `plot_seed_distribution` shows the raw points behind these bands.

    Spearman only. Validation MAE moves within about 1.5 upvotes across the whole grid
    and not monotonically, so plotting it beside a metric that moves by half its range
    invites reading noise as signal -- the README states that it does not separate these
    models rather than giving it half a figure.
    """
    if not records:
        logger.warning("No fusion-curve runs found; skipping %s", output)
        return
    apply_style()
    frame = pd.DataFrame(records)

    figure, panels = plt.subplots(2, 3, figsize=(12.5, 6.6), sharey=True)
    for index, (title, shapes) in enumerate(CURVE_PANELS):
        panel = panels[index // 3][index % 3]
        for slot, shape in enumerate(shapes):
            rows = frame[frame["widths"] == shape]
            if rows.empty:
                continue
            grouped = rows.groupby("fusion_layer")["spearman"]
            points = sorted(int(str(key)) for key in grouped.groups)
            colour = SERIES[slot % len(SERIES)]
            panel.plot(
                points,
                [grouped.get_group(p).median() for p in points],
                color=colour,
                linewidth=2,
                marker="o",
                markersize=5,
                label=shape,
            )
            panel.fill_between(
                points,
                [grouped.get_group(p).quantile(0.25) for p in points],
                [grouped.get_group(p).quantile(0.75) for p in points],
                color=colour,
                alpha=0.15,
                linewidth=0,
            )
        panel.set_title(title, fontsize=10, loc="left")
        panel.set_xticks(range(1, 7))
        panel.legend(fontsize=7.5, loc="upper right")
        if index // 3 == 1:
            panel.set_xlabel("layer the groups first meet", fontsize=9)
        if index % 3 == 0:
            panel.set_ylabel("test Spearman ρ", fontsize=9)

    figure.suptitle(
        "Delaying fusion costs accuracy — but most of the cost is narrow blocks, not late fusion",
        fontsize=11,
        x=0.01,
        ha="left",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    save_figure(figure, output, tight=False)
    plt.close(figure)
    logger.info("Wrote %s", output)


def plot_seed_distribution(records: list[dict[str, Any]], output: Path) -> None:
    """Every seed at every fusion point, drawn individually.

    Args:
        records: `_curve_runs` output.
        output: Where to write the PNG.

    The companion to the curve, and the reason the run uses ten seeds rather than three.
    If late fusion is bimodal -- sometimes training to early fusion's accuracy and
    sometimes collapsing 0.10 below it -- then a median and a band hide the only thing
    worth knowing, which is *how often it works* rather than how good it is on average.
    """
    if not records:
        logger.warning("No fusion-curve runs found; skipping %s", output)
        return
    apply_style()
    frame = pd.DataFrame(records)
    # Only the shapes the grid was designed around. The project has accumulated others --
    # one-off runs, a hyperparameter search that set the same field -- and drawing every
    # shape it can find turns six readable panels into twenty-three, most of them a single
    # fusion point with a handful of seeds.
    designed = [shape for _, shapes in CURVE_PANELS for shape in shapes]
    shapes = [
        shape
        for shape in sorted(frame["widths"].unique(), key=lambda w: (len(w.split(",")), w))
        if shape in designed
    ]

    columns = min(4, len(shapes))
    rows_needed = (len(shapes) + columns - 1) // columns
    figure, panels = plt.subplots(
        rows_needed, columns, figsize=(3.1 * columns, 2.5 * rows_needed), squeeze=False
    )
    generator = np.random.default_rng(0)
    for index, shape in enumerate(shapes):
        panel = panels[index // columns][index % columns]
        rows = frame[frame["widths"] == shape]
        # Jittered, so ten seeds landing on the same value are visible as ten points
        # rather than as one.
        jitter = generator.uniform(-0.12, 0.12, len(rows))
        panel.scatter(rows["fusion_layer"] + jitter, rows["spearman"], s=14, color=BLUE, alpha=0.75)
        panel.set_title(shape, fontsize=8.5, loc="left")
        panel.set_xticks(sorted(rows["fusion_layer"].unique()))
        panel.tick_params(labelsize=7)
    for index in range(len(shapes), rows_needed * columns):
        panels[index // columns][index % columns].axis("off")
    figure.supxlabel("layer at which the groups first meet", fontsize=9, color=INK_SOFT)
    figure.supylabel("test Spearman ρ, one point per seed", fontsize=9, color=INK_SOFT)
    figure.tight_layout()
    save_figure(figure, output, tight=False)
    plt.close(figure)
    logger.info("Wrote %s", output)


def _cell(records: dict[str, Any], alpha: float, key: str) -> float:
    """One cell's median, from either a raw-seed file or a recovered medians file."""
    row = records.get(f"{alpha:g}") or records.get(f"{alpha}") or records[str(alpha)]
    value = row[key]
    return float(value["median"]) if isinstance(value, dict) else float(np.median(value))


def plot_interaction(records: dict[str, Any], output: Path) -> None:
    """Accuracy against interaction strength, for three fusion architectures.

    Args:
        records: `fusion_probes.py interaction` output, alpha to `arch@width` to scores.
        output: Where to write the PNG.

    The control that makes the null on real data worth reporting. On a synthetic target
    `g + h + alpha * g * h`, the architecture that reduces each modality to one score
    cannot represent the product at any alpha, and the gap grows without bound; the two
    that can are indistinguishable from each other everywhere. At `alpha = 0` -- which is
    where the real task sits -- all three are identical, so a setup that finds no
    difference on Hacker News is measuring the data rather than failing to measure.

    Two panels: the curves, and the gap against width. The second is the structural
    claim -- if capacity closed the gap it would be a fitting problem, and it does not.
    """
    if not records:
        logger.warning("No interaction records; skipping %s", output)
        return
    apply_style()
    alphas = sorted(float(a) for a in records)
    widths = sorted({int(key.split("@")[1]) for key in next(iter(records.values()))})
    architectures = ("early", "late-deep", "late-snoek")

    figure, panels = plt.subplots(1, 2, figsize=(11, 4.4))
    reference = widths[0]
    for slot, architecture in enumerate(architectures):
        medians = [_cell(records, a, f"{architecture}@{reference}") for a in alphas]
        panels[0].plot(
            alphas,
            medians,
            color=SERIES[slot],
            linewidth=2,
            marker="o",
            markersize=4,
            label=architecture,
        )
    panels[0].set_title(
        f"Accuracy against interaction strength ({reference} hidden units)", fontsize=10, loc="left"
    )
    panels[0].set_xlabel("α — how much of the signal lives in title × metadata")
    panels[0].set_ylabel("test Spearman ρ")
    panels[0].legend(fontsize=8)

    for slot, width in enumerate(widths):
        gaps = [
            _cell(records, a, f"early@{width}") - _cell(records, a, f"late-snoek@{width}")
            for a in alphas
        ]
        panels[1].plot(
            alphas,
            gaps,
            color=SERIES[slot % len(SERIES)],
            linewidth=2,
            marker="o",
            markersize=4,
            label=f"{width} units",
        )
    panels[1].axhline(0, color=MUTED, linewidth=1)
    panels[1].set_title(
        "Early minus Snoek late fusion — capacity does not close it", fontsize=10, loc="left"
    )
    panels[1].set_xlabel("α")
    panels[1].set_ylabel("gap in Spearman ρ")
    panels[1].legend(fontsize=8)

    figure.tight_layout()
    save_figure(figure, output, tight=False)
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
        "--curve",
        action="store_true",
        help="Also draw the fusion-point curve, which reads run history back from W&B.",
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
    plot_architectures(config, ASSETS / "fusion-architectures.png")

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

    # Behind a flag because it is the only figure that needs the network: everything else
    # is drawn from the evaluation JSON on disk, so `make report` still works offline.
    if arguments.curve and config.wandb_project:
        records = _curve_runs(config.wandb_project)
        plot_fusion_curve(records, ASSETS / "fusion-curve.png")
        plot_seed_distribution(records, ASSETS / "fusion-seeds.png")
    elif arguments.curve:
        logger.warning("No wandb_project configured, so the fusion curve cannot be drawn")


if __name__ == "__main__":
    main()
