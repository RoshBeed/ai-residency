"""Figures for the write-up, generated from the W&B runs that produced them.

    python -m word_embeddings.report              # every figure
    python -m word_embeddings.report --figure sweep

Reads run history back out of W&B rather than re-training, so a figure can be
redrawn without spending the compute again, and so what is plotted is provably the
same run the numbers in the README came from.

Two choices worth stating, because both are the difference between an honest figure
and a misleading one:

**Loss is plotted relative to each run's own random-init baseline.** CBOW at 5
negatives starts at 4.16 and skip-gram at the same setting starts at 41.59, because
skip-gram scores 2*window positives per row where CBOW scores one. Plotting both
against one axis in absolute terms would say nothing except which architecture has
more terms in its sum; dividing each by its own starting point asks the question
that matters, which is how far each got from knowing nothing. The alternative --
two y-axes -- makes any two curves look however you like.

**The benchmarks are separate panels, not three lines on one axis.** They are
different measures: two rank correlations and an accuracy. They share the interval
[0, 1] and nothing else, and putting them on one axis invites reading "SimLex is
worse than WordSim" as a fact about the model rather than about the benchmarks.
"""

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA

from ai_residency_common import get_logger, load_checkpoint, resolve_required, set_seed
from config import Config
from word_embeddings.evaluate import PUBLISHED_TEXT8, checkpoint_vocabulary

__all__ = ["main", "plot_benchmarks", "plot_country_capital", "plot_loss", "plot_sweep"]

logger = get_logger(__name__)

# The reference categorical palette, first three slots -- the set that validates on
# all pairs rather than only adjacent ones, which is what a scatter needs.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_SOFT, MUTED, RULE = "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
SURFACE = "#fcfcfb"

ASSETS = Path("services/word-embeddings/assets")


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


def _runs(project: str, sweep_id: str | None = None) -> list[Any]:
    """Fetch finished runs.

    Args:
        project: `entity/project` on W&B.
        sweep_id: Restrict to one sweep, when given.

    Returns:
        The finished runs.
    """
    import wandb

    api = wandb.Api()
    if sweep_id:
        # wandb's Api is untyped; strict mypy rejects the call rather than the value.
        sweep: Any = api.sweep(f"{project}/{sweep_id}")  # type: ignore[no-untyped-call]
        return [run for run in sweep.runs if run.state == "finished"]
    return [run for run in api.runs(project) if run.state == "finished"]


def _history(run: Any, keys: list[str]) -> dict[str, list[float]]:
    """Pull specific metric series out of a run's history, dropping incomplete rows."""
    columns: dict[str, list[float]] = {key: [] for key in keys}
    for row in run.scan_history(keys=keys):
        if all(row.get(key) is not None for key in keys):
            for key in keys:
                columns[key].append(float(row[key]))
    return columns


def _best_per_architecture(runs: list[Any]) -> dict[str, Any]:
    """The highest-scoring finished run for each architecture."""
    best: dict[str, Any] = {}
    for run in runs:
        architecture = run.config.get("architecture")
        score = run.summary.get("best/score")
        if architecture is None or score is None:
            continue
        if architecture not in best or score > best[architecture].summary["best/score"]:
            best[architecture] = run
    return best


def plot_loss(runs: list[Any], output: Path) -> None:
    """Training loss as a fraction of each run's own random-init baseline."""
    _style()
    figure, axes = plt.subplots(figsize=(7.5, 4.2))

    for (architecture, run), colour in zip(
        sorted(_best_per_architecture(runs).items()), (BLUE, ORANGE), strict=False
    ):
        baseline = run.config.get("random_baseline_loss")
        history = _history(run, ["train/step", "train/loss"])
        if not history["train/step"] or not baseline:
            continue
        steps = np.array(history["train/step"])
        relative = np.array(history["train/loss"]) / baseline
        axes.plot(steps, relative, color=colour, linewidth=2, label=architecture)
        axes.annotate(
            architecture,
            (steps[-1], relative[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            color=INK_SOFT,
            fontsize=9,
            va="center",
        )

    axes.axhline(1.0, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    axes.annotate(
        "untrained (loss = its own random-init value)",
        (0.01, 1.0),
        xycoords=("axes fraction", "data"),
        xytext=(0, 5),
        textcoords="offset points",
        color=MUTED,
        fontsize=8.5,
    )
    axes.set_xlabel("optimizer step")
    axes.set_ylabel("loss / random-init baseline")
    axes.set_title("Training loss, each run against its own starting point", pad=12)
    axes.set_ylim(0, 1.1)
    axes.legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(output, dpi=200, facecolor=SURFACE)
    plt.close(figure)
    logger.info("Wrote %s", output)


def plot_benchmarks(runs: list[Any], output: Path) -> None:
    """Benchmark scores against epoch, one panel per benchmark."""
    _style()
    panels = [
        ("eval/wordsim353_spearman", "WordSim-353", "Spearman rho", "wordsim353_spearman"),
        ("eval/simlex999_spearman", "SimLex-999", "Spearman rho", "simlex999_spearman"),
        ("eval/analogy_total", "Google analogies", "accuracy", "analogy_total"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    best = sorted(_best_per_architecture(runs).items())

    for panel, (key, title, ylabel, published_key) in zip(axes, panels, strict=True):
        for (architecture, run), colour in zip(best, (BLUE, ORANGE), strict=False):
            history = _history(run, ["epoch", key])
            panel.plot(
                history["epoch"],
                history[key],
                color=colour,
                linewidth=2,
                marker="o",
                markersize=4,
                label=architecture,
            )
        published = PUBLISHED_TEXT8.get(published_key)
        if published:
            panel.axhline(published, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
            panel.annotate(
                f"published {published:.2f}",
                (0.02, published),
                xycoords=("axes fraction", "data"),
                xytext=(0, 4),
                textcoords="offset points",
                color=MUTED,
                fontsize=8,
            )
        panel.set_title(title, fontsize=11, pad=8)
        panel.set_xlabel("epoch")
        panel.set_ylabel(ylabel)
        panel.set_ylim(bottom=0)

    axes[0].legend(loc="lower right")
    figure.suptitle(
        "Intrinsic benchmarks over training, against published word2vec-on-text8 results",
        fontsize=12,
        y=1.02,
    )
    figure.tight_layout()
    figure.savefig(output, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)
    logger.info("Wrote %s", output)


def plot_sweep(runs: list[Any], output: Path) -> None:
    """Final score against each swept hyperparameter, one panel each.

    A dot per run rather than a line: these are independent runs, and joining them
    would imply a trajectory that does not exist.
    """
    _style()
    parameters = [
        ("architecture", "architecture"),
        ("embed_size", "embedding size"),
        ("window_size", "window"),
        ("num_negatives", "negatives"),
        ("subsample_threshold", "subsample t"),
        ("learning_rate", "learning rate"),
    ]
    scored = [run for run in runs if run.summary.get("best/score") is not None]
    if not scored:
        logger.warning("No finished runs with a score; skipping the sweep figure")
        return

    figure, axes = plt.subplots(1, len(parameters), figsize=(15, 3.6), sharey=True)
    rng = np.random.default_rng(0)

    for panel, (key, label) in zip(axes, parameters, strict=True):
        values = [run.config.get(key) for run in scored]
        scores = [run.summary["best/score"] for run in scored]
        colours = [BLUE if run.config.get("architecture") == "cbow" else ORANGE for run in scored]
        if key == "learning_rate":
            panel.scatter(values, scores, c=colours, s=26, alpha=0.85, edgecolors="none")
            panel.set_xscale("log")
        else:
            # Sorted numerically where the values are numbers: a plain string sort
            # puts window 10 before 5, which reads as a trend that is not there.
            unique = {str(value): value for value in values}
            categories = sorted(
                unique,
                key=lambda name: (
                    float(unique[name]) if isinstance(unique[name], int | float) else name
                ),
            )
            positions = {name: index for index, name in enumerate(categories)}
            # Jittered, so runs sharing a setting do not hide behind one another.
            jitter = rng.uniform(-0.16, 0.16, len(values))
            panel.scatter(
                [
                    positions[str(value)] + offset
                    for value, offset in zip(values, jitter, strict=True)
                ],
                scores,
                c=colours,
                s=26,
                alpha=0.85,
                edgecolors="none",
            )
            panel.set_xticks(range(len(categories)))
            panel.set_xticklabels(categories, fontsize=8)
        panel.set_xlabel(label, fontsize=9)

    axes[0].set_ylabel("composite score")
    handles = [
        Line2D([], [], marker="o", linestyle="", color=BLUE, label="cbow"),
        Line2D([], [], marker="o", linestyle="", color=ORANGE, label="skipgram"),
    ]
    axes[-1].legend(handles=handles, loc="lower right", fontsize=8)
    figure.suptitle(
        f"Sweep: composite score against each hyperparameter ({len(scored)} runs)",
        fontsize=12,
        y=1.04,
    )
    figure.tight_layout()
    figure.savefig(output, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)
    logger.info("Wrote %s", output)


_COUNTRY_CAPITAL = (
    ("france", "paris"),
    ("italy", "rome"),
    ("spain", "madrid"),
    ("germany", "berlin"),
    ("greece", "athens"),
    ("japan", "tokyo"),
    ("russia", "moscow"),
    ("china", "beijing"),
    ("portugal", "lisbon"),
    ("poland", "warsaw"),
)


def _place_labels(axes: Any, points: "np.ndarray[Any, Any]", labels: list[str]) -> None:
    """Annotate every point, nudging labels that would sit on top of each other.

    Words cluster in a PCA projection precisely when the embedding has learned
    something, so the good version of this figure is the one most prone to
    overlapping labels -- `spain` and `italy` landing on the same pixel is a
    property of the result, not of the plotting.
    """
    span = float(points[:, 1].max() - points[:, 1].min()) or 1.0
    minimum_gap = span * 0.035
    placed: list[tuple[float, float]] = []

    for (x, y), label in zip(points, labels, strict=True):
        offset = 0.0
        while any(
            abs(x - other_x) < span * 0.16 and abs(y + offset - other_y) < minimum_gap
            for other_x, other_y in placed
        ):
            offset -= minimum_gap
        placed.append((x, y + offset))
        axes.annotate(
            label,
            (x, y),
            xytext=(6, 4 + offset / span * 190),
            textcoords="offset points",
            fontsize=8.5,
            color=INK_SOFT,
        )


def plot_country_capital(config: Config, output: Path) -> None:
    """Mikolov et al. Figure 2, drawn on this run's own vectors.

    The README shows the paper's version of this plot. Reproducing it on vectors
    trained here is the claim actually being made -- that these embeddings encode
    country-capital as a consistent direction -- rather than a citation of someone
    else's.
    """
    _style()
    set_seed(config.seed)
    path, _ = resolve_required(config.embedding_checkpoint, config.hf_repo_id)
    checkpoint = load_checkpoint(path)
    embeddings = checkpoint["model_state_dict"]["in_embed.weight"].cpu()
    word_to_index = checkpoint_vocabulary(checkpoint["extra"])

    pairs = [
        (country, capital)
        for country, capital in _COUNTRY_CAPITAL
        if country in word_to_index and capital in word_to_index
    ]
    if not pairs:
        logger.warning("Vocabulary holds none of the country/capital pairs; skipping")
        return

    words = [word for pair in pairs for word in pair]
    matrix = torch.stack([embeddings[word_to_index[word]] for word in words]).numpy()
    projected = PCA(n_components=2, random_state=config.seed).fit_transform(matrix)

    figure, axes = plt.subplots(figsize=(7.5, 6))
    for index in range(len(pairs)):
        start, end = projected[2 * index], projected[2 * index + 1]
        axes.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={"arrowstyle": "->", "color": RULE, "linewidth": 1.2},
        )
        # Indexed rather than unpacked: `scatter(*point, s=28)` would feed the
        # second coordinate into the size argument if the array ever grew a column.
        axes.scatter(start[0], start[1], color=BLUE, s=28, zorder=3)
        axes.scatter(end[0], end[1], color=ORANGE, s=28, zorder=3)

    _place_labels(axes, projected, [word for pair in pairs for word in pair])

    handles = [
        Line2D([], [], marker="o", linestyle="", color=BLUE, label="country"),
        Line2D([], [], marker="o", linestyle="", color=ORANGE, label="capital"),
    ]
    axes.legend(handles=handles, loc="best")
    axes.set_title("Country and capital vectors, projected by PCA", pad=12)
    figure.tight_layout()
    figure.savefig(output, dpi=200, facecolor=SURFACE)
    plt.close(figure)
    logger.info("Wrote %s", output)


def main() -> None:
    """Entry point: `python -m word_embeddings.report`."""
    parser = argparse.ArgumentParser(description="Generate the write-up's figures")
    parser.add_argument("--project", default="roshbeed/ai-residency-word-embeddings")
    parser.add_argument("--sweep", default=None, help="Restrict run-based figures to one sweep")
    parser.add_argument(
        "--figure",
        choices=("loss", "benchmarks", "sweep", "country-capital", "all"),
        default="all",
    )
    args = parser.parse_args()

    ASSETS.mkdir(parents=True, exist_ok=True)
    config = Config()

    if args.figure in ("loss", "benchmarks", "sweep", "all"):
        runs = _runs(args.project, args.sweep)
        logger.info("Read %d finished runs", len(runs))
        if args.figure in ("loss", "all"):
            plot_loss(runs, ASSETS / "training-loss.png")
        if args.figure in ("benchmarks", "all"):
            plot_benchmarks(runs, ASSETS / "benchmarks-vs-epoch.png")
        if args.figure in ("sweep", "all"):
            plot_sweep(runs, ASSETS / "sweep-results.png")

    if args.figure in ("country-capital", "all"):
        plot_country_capital(config, ASSETS / "country-capital-pca.png")


if __name__ == "__main__":
    main()
