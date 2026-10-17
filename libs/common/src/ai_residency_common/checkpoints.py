"""Checkpoint saving and loading.

Two things this fixes portfolio-wide. First, the repos disagreed on what a checkpoint
file contains: some wrote a bare `state_dict`, others a dict with assorted metadata
keys, so nothing could be loaded generically. Second, `load_checkpoint` defaults to
`map_location="cpu"`, because checkpoints trained on Apple Silicon carry `mps:0`
tensor metadata and fail to deserialise inside the CPU-only Linux containers this
portfolio deploys to.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from ai_residency_common.logging import get_logger

__all__ = [
    "Resumed",
    "load_checkpoint",
    "resume_from",
    "resume_path_for",
    "save_checkpoint",
    "save_resume",
]

logger = get_logger(__name__)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    optimizer: Optimizer | None = None,
    epoch: int | None = None,
    metrics: dict[str, float] | None = None,
    extra: dict[str, Any] | None = None,
    state_dict: Mapping[str, Any] | None = None,
) -> None:
    """Write a model checkpoint, creating parent directories as needed.

    Args:
        path: Destination file, e.g. `checkpoints/best_model.pth`.
        model: Model whose `state_dict` is saved.
        optimizer: Optimizer to save alongside, for resuming training.
        epoch: Epoch this checkpoint came from.
        metrics: Scalar metrics worth keeping with the weights, e.g. test loss.
        extra: Anything else the repo needs to reload the model, such as the
            vocabulary size or embedding dimension the architecture was built with.
        state_dict: Weights to save instead of the model's own. For an adapter on a
            frozen backbone, `model.state_dict()` includes the whole backbone: a
            LoRA-wrapped 0.6B model writes 2.4GB where the trained weights are a few
            MB, and every push then re-uploads the frozen base. Pass just the
            trainable weights here.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict() if state_dict is None else dict(state_dict),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "metrics": metrics or {},
        "extra": extra or {},
    }
    torch.save(payload, path)
    logger.info("Saved checkpoint to %s (%.1f MB)", path, path.stat().st_size / 1e6)


def load_checkpoint(
    path: str | Path,
    model: nn.Module | None = None,
    *,
    optimizer: Optimizer | None = None,
    map_location: str = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint written by `save_checkpoint`.

    Args:
        path: Checkpoint file to read.
        model: If given, load the saved weights into it in place.
        optimizer: If given, restore its state too.
        map_location: Where to materialise tensors. Defaults to `"cpu"` so that a
            checkpoint trained on MPS still loads on a CPU-only host.

    Returns:
        The full checkpoint payload, so callers can read `epoch`, `metrics` and
        `extra` even when they passed a model to populate.
    """
    path = Path(path)
    payload: dict[str, Any] = torch.load(path, map_location=map_location, weights_only=False)

    if model is not None:
        model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])

    logger.info("Loaded checkpoint from %s", path)
    return payload


@dataclass(frozen=True)
class Resumed:
    """Where a resumed run should pick up.

    Attributes:
        start_epoch: First epoch still to run. 1 for a fresh start.
        best_metric: The best score seen so far, so the "is this a new best?"
            test survives the restart and a worse epoch cannot overwrite a better
            checkpoint that a previous attempt already saved.
        resumed: Whether state was actually restored, rather than this being a
            fresh start or a refused mismatch.
    """

    start_epoch: int
    best_metric: float
    resumed: bool


def resume_path_for(checkpoint_path: str | Path) -> Path:
    """The resume file that sits beside a service's best-checkpoint path."""
    path = Path(checkpoint_path)
    return path.with_name(f"{path.stem}_resume{path.suffix}")


def save_resume(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    *,
    epoch: int,
    best_metric: float,
    fingerprint: Mapping[str, Any],
    metrics: Mapping[str, float] | None = None,
    state_dict: Mapping[str, Any] | None = None,
) -> None:
    """Write the state a run needs to continue, every epoch.

    Distinct from `save_checkpoint`, which every trainer calls only on an
    improvement because that file is what gets served and pushed. A run that dies
    after a worse epoch would otherwise restart from the last *better* one and
    repeat work -- or, where nothing had improved yet, from scratch.

    The optimizer goes in because AdamW's moments are part of the run: restoring
    weights alone gives a different trajectory, not the same one continued.

    Args:
        path: Where to write. Use `resume_path_for` to derive it.
        model: Model whose weights to save.
        optimizer: Optimizer whose state to save.
        epoch: The epoch that just *finished*, so resuming starts at epoch + 1.
        best_metric: Best score so far, by whatever measure the caller ranks on.
        fingerprint: What makes this the same run -- see `resume_from`.
        metrics: Extra scalars worth keeping.
        state_dict: Weights to save instead of the model's own, for a small
            trainable module on a frozen backbone.
    """
    save_checkpoint(
        path,
        model,
        optimizer=optimizer,
        epoch=epoch,
        metrics=dict(metrics) if metrics else None,
        extra={"fingerprint": dict(fingerprint), "best_metric": best_metric},
        state_dict=state_dict,
    )


def resume_from(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    *,
    fingerprint: Mapping[str, Any],
    default_best: float,
    map_location: str = "cpu",
) -> Resumed:
    """Restore a run in progress, or report that there is nothing to restore.

    Written once here rather than per trainer: every service in this repo trains on
    a machine that can die mid-run, and a resume that each trainer re-implements is
    a resume that each trainer gets subtly wrong.

    `fingerprint` is what makes two runs the same run -- corpus size, model shape,
    batch size, learning rate. Without it a resume file from a different experiment
    loads cleanly and silently averages two runs together. On a mismatch this warns
    and starts fresh instead.

    The payload is read before anything is loaded into `model`, so a refused resume
    leaves the freshly built model untouched rather than half-overwritten.

    Args:
        path: The resume file, from `resume_path_for`.
        model: Model to restore in place.
        optimizer: Optimizer to restore in place.
        fingerprint: The current run's settings, compared for equality.
        default_best: What `best_metric` should be for a fresh start -- `0.0` when
            ranking on accuracy, `float("inf")` when ranking on loss.
        map_location: Device to map tensors onto, as a string.

    Returns:
        Where to pick up, and whether anything was restored.
    """
    resume = Path(path)
    if not resume.exists():
        return Resumed(start_epoch=1, best_metric=default_best, resumed=False)

    peek = load_checkpoint(resume, map_location=map_location)
    saved = peek.get("extra") or {}
    if saved.get("fingerprint") != dict(fingerprint):
        logger.warning(
            "Ignoring %s: written by a run with different settings (%s, now %s). Starting fresh.",
            resume,
            saved.get("fingerprint"),
            dict(fingerprint),
        )
        return Resumed(start_epoch=1, best_metric=default_best, resumed=False)

    load_checkpoint(resume, model, optimizer=optimizer, map_location=map_location)
    start_epoch = int(peek.get("epoch") or 0) + 1
    best_metric = float(saved.get("best_metric", default_best))
    logger.info(
        "Resuming from %s: starting at epoch %d, best metric so far %.4f",
        resume,
        start_epoch,
        best_metric,
    )
    return Resumed(start_epoch=start_epoch, best_metric=best_metric, resumed=True)
