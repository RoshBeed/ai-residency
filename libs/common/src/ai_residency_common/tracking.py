"""Experiment tracking with Weights & Biases.

Only two of the six repos logged metrics anywhere, and the other four printed loss
numbers to stdout where they could not be compared across runs. This wraps W&B so
every repo reports the same way.

Without any W&B credential the run falls back to offline mode rather than failing or
prompting, which keeps training usable on a fresh clone and in containers that have
no secrets. Offline runs still record locally and can be synced later with
`wandb sync`.

Beyond scalars, this exposes the three things a training run needs to be *watchable*
rather than merely recorded: `define_metrics` to give each family of metrics its own
x-axis, `log_table` for qualitative probes, and `set_summary` for the one row per run
that a sweep sorts on.
"""

import os
from collections.abc import Iterable, Sequence
from typing import Any

from ai_residency_common.logging import get_logger

__all__ = [
    "define_metrics",
    "finish_tracking",
    "init_tracking",
    "log_figure",
    "log_metrics",
    "log_table",
    "set_summary",
]

logger = get_logger(__name__)

_run: Any = None


def _has_credential() -> bool:
    """Whether W&B can reach the server without prompting.

    `WANDB_API_KEY` is only one of the ways a machine is logged in; `wandb login`
    writes to `~/.netrc`, and checking the environment alone declares a logged-in
    developer offline. The cost of getting this wrong is asymmetric and quiet: the
    run trains fine, the dashboard stays empty, and you find out at the end.
    """
    if os.environ.get("WANDB_API_KEY"):
        return True
    try:
        import wandb

        return bool(wandb.api.api_key)
    except Exception:
        return False


def init_tracking(
    project: str | None,
    *,
    config: dict[str, Any] | None = None,
    run_name: str | None = None,
) -> bool:
    """Start a tracking run.

    Args:
        project: W&B project name. Passing None disables tracking entirely, which is
            how a repo opts out.
        config: Hyperparameters to record against the run.
        run_name: Optional human-readable name for this run.

    Returns:
        True if a run started (online or offline), False if tracking is disabled or
        W&B is unavailable.
    """
    global _run

    if project is None:
        logger.info("No W&B project configured; not tracking this run")
        return False

    try:
        import wandb
    except ImportError:
        logger.warning("wandb is not installed; not tracking this run")
        return False

    if not _has_credential():
        os.environ.setdefault("WANDB_MODE", "offline")
        logger.warning("No W&B credential; logging offline (sync later with `wandb sync`)")

    _run = wandb.init(project=project, config=config, name=run_name)
    logger.info("Tracking run in W&B project %s", project)
    return True


def define_metrics(spec: dict[str, str]) -> None:
    """Declare which x-axis each family of metrics is plotted against.

    Args:
        spec: Metric name or glob to the metric that is its step, e.g.
            `{"train/*": "train/step", "eval/*": "epoch"}`. A step metric maps to
            itself and is declared automatically.

    A run that logs per-batch metrics unstepped and per-epoch metrics with
    `step=epoch` silently loses the per-epoch series: the unstepped calls advance
    W&B's own counter into the thousands, and a later `step=3` is behind it, so it is
    dropped. Giving each family its own step metric removes the shared counter that
    makes that possible, rather than relying on every call site to agree.
    """
    if _run is None:
        return
    for step_metric in dict.fromkeys(spec.values()):
        _run.define_metric(step_metric)
    for name, step_metric in spec.items():
        _run.define_metric(name, step_metric=step_metric)


def log_metrics(metrics: dict[str, float], *, step: int | None = None) -> None:
    """Record metrics against the active run.

    A no-op when tracking was never started, so training code can call this
    unconditionally without branching on whether W&B is configured.

    Args:
        metrics: Metric names to values, e.g. `{"train_loss": 1.5}`.
        step: Step or epoch these belong to.
    """
    if _run is None:
        return
    _run.log(metrics, step=step)


def log_table(
    name: str,
    columns: Sequence[str],
    rows: list[list[Any]],
    *,
    step: int | None = None,
) -> None:
    """Log a table of non-scalar results, such as a nearest-neighbour probe.

    Args:
        name: Key the table appears under in the run.
        columns: Column headers.
        rows: Row values, one list per row.
        step: Step or epoch this belongs to.
    """
    if _run is None:
        return
    import wandb

    # Copied into a fresh list rather than passed through. `wandb.Table` takes
    # `list[Iterable[Any]]` and `list` is invariant, so a `list[list[Any]]` is not one --
    # correct of mypy, and harmless here, since Table only iterates the rows.
    data: list[Iterable[Any]] = list(rows)
    _run.log({name: wandb.Table(columns=list(columns), data=data)}, step=step)


def log_figure(name: str, figure: Any, *, step: int | None = None) -> None:
    """Log a matplotlib figure as an image.

    Args:
        name: Key the image appears under in the run.
        figure: A matplotlib `Figure`.
        step: Step or epoch this belongs to.
    """
    if _run is None:
        return
    import wandb

    _run.log({name: wandb.Image(figure)}, step=step)


def set_summary(values: dict[str, Any]) -> None:
    """Set the single-row summary a sweep table sorts and ranks on.

    Args:
        values: Summary keys to values, e.g. the best epoch's benchmark scores.

    W&B defaults a summary to the *last* logged value of each metric, which for a run
    that ends past its best epoch ranks it by where it finished rather than by how
    good it got.
    """
    if _run is None:
        return
    _run.summary.update(values)


def finish_tracking() -> None:
    """Close the active run, if there is one."""
    global _run
    if _run is None:
        return
    _run.finish()
    _run = None
