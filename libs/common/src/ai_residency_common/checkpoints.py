"""Checkpoint saving and loading.

Two things this fixes portfolio-wide. First, the repos disagreed on what a checkpoint
file contains: some wrote a bare `state_dict`, others a dict with assorted metadata
keys, so nothing could be loaded generically. Second, `load_checkpoint` defaults to
`map_location="cpu"`, because checkpoints trained on Apple Silicon carry `mps:0`
tensor metadata and fail to deserialise inside the CPU-only Linux containers this
portfolio deploys to.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from ai_residency_common.logging import get_logger

__all__ = ["load_checkpoint", "save_checkpoint"]

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
