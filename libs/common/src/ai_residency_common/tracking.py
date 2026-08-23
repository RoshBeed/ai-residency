"""Experiment tracking with Weights & Biases.

Only two of the six repos logged metrics anywhere, and the other four printed loss
numbers to stdout where they could not be compared across runs. This wraps W&B so
every repo reports the same way.

Without `WANDB_API_KEY` the run falls back to offline mode rather than failing or
prompting, which keeps training usable on a fresh clone and in containers that have
no secrets. Offline runs still record locally and can be synced later with
`wandb sync`.
"""

import os
from typing import Any

from ai_residency_common.logging import get_logger

__all__ = ["finish_tracking", "init_tracking", "log_metrics"]

logger = get_logger(__name__)

_run: Any = None


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

    if not os.environ.get("WANDB_API_KEY"):
        os.environ.setdefault("WANDB_MODE", "offline")
        logger.warning("No WANDB_API_KEY; logging offline (sync later with `wandb sync`)")

    _run = wandb.init(project=project, config=config, name=run_name)
    logger.info("Tracking run in W&B project %s", project)
    return True


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


def finish_tracking() -> None:
    """Close the active run, if there is one."""
    global _run
    if _run is None:
        return
    _run.finish()
    _run = None
