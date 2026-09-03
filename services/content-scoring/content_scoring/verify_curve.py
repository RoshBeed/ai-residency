"""Reconcile the fusion-curve grid against what actually reached W&B.

    make -C services/content-scoring fusion-verify           # report
    make -C services/content-scoring fusion-verify FIX=1     # requeue the gaps

The progress file records a combination when the training process exits zero, which is
not the same as its results being *stored*: `set_summary` runs at the end of a run, and
when the W&B service was backed up uploading per-epoch table artifacts, 57 runs exited
cleanly with no summary ever landing, and another 37 never registered at all. The grid
counted 490 done while 95 of them had no data — a discrepancy nothing would have caught
before the figures came out empty.

Logging tables is off for grid runs now, which removes the cause. This is the check that
the cause stayed removed.
"""

import argparse
import pathlib

from ai_residency_common import get_logger
from config import Config
from content_scoring.report import _curve_runs

logger = get_logger(__name__)

PROGRESS = pathlib.Path(__file__).resolve().parent.parent / "assets" / "fusion-curve.progress"


def main() -> None:
    """Report, and optionally requeue, combinations whose results never landed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", nargs="+", required=True, help="The grid's layer shapes.")
    parser.add_argument("--seeds", nargs="+", type=int, required=True, help="The grid's seeds.")
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Drop the gaps from the progress file so a rerun redoes them.",
    )
    arguments = parser.parse_args()
    config = Config()
    if not config.wandb_project:
        logger.error("No wandb_project configured; nothing to reconcile against")
        return

    usable = {
        (record["widths"], record["fusion_layer"], record["seed"])
        for record in _curve_runs(config.wandb_project)
    }
    wanted = {
        (shape, point, seed)
        for shape in arguments.shapes
        for point in range(1, len(shape.split(",")) + 1)
        for seed in arguments.seeds
    }
    missing = wanted - usable
    logger.info("%d combinations wanted, %d have usable results", len(wanted), len(wanted & usable))
    if not missing:
        logger.info("No gaps.")
        return

    by_shape: dict[str, int] = {}
    for shape, _, _ in missing:
        by_shape[shape] = by_shape.get(shape, 0) + 1
    for shape, count in sorted(by_shape.items(), key=lambda item: -item[1]):
        logger.warning("  %-26s %d missing", shape, count)

    if not arguments.fix:
        logger.warning("%d gaps; rerun with FIX=1 to requeue them", len(missing))
        return
    if PROGRESS.exists():
        kept = [
            line
            for line in PROGRESS.read_text().splitlines()
            if line.count("|") == 2
            and (line.split("|")[0], int(line.split("|")[1]), int(line.split("|")[2]))
            not in missing
        ]
        PROGRESS.write_text("\n".join(kept) + "\n")
        logger.info("Progress trimmed to %d; rerun `make fusion-curve` to fill the gaps", len(kept))


if __name__ == "__main__":
    main()
