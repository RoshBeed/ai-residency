"""Device selection.

Every repo in this portfolio picked its own accelerator in its own way, and each
version was subtly different: one omitted CUDA entirely, one returned bare strings
instead of `torch.device`, one repeated the same block in three files. This is the
single implementation they all use now.
"""

import os

import torch

from ai_residency_common.logging import get_logger

__all__ = ["DEVICE_VAR", "get_device"]

logger = get_logger(__name__)

# Picking the best available device is the right default and the wrong answer often
# enough to need an escape hatch. The case that forced it: PPO on this Mac allocates
# past MPS's high-watermark ceiling and dies with `MPS backend out of memory
# (allocated 42.45 GiB, max allowed 42.43 GiB)` partway through a step, while the same
# work on CPU has 32GB of real RAM and no watermark to hit. Being able to say
# `AI_RESIDENCY_DEVICE=cpu` beats editing a library to get a run to finish.
DEVICE_VAR = "AI_RESIDENCY_DEVICE"


def get_device() -> torch.device:
    """Return the best available device, preferring CUDA, then MPS, then CPU.

    `AI_RESIDENCY_DEVICE` overrides the choice, matching how `AI_RESIDENCY_<FIELD>`
    overrides any `Config` field. An unavailable choice is honoured rather than
    silently corrected -- a run asked for a device explicitly, and torch's own error
    says more than a fallback that quietly trains somewhere else would.

    Returns:
        The selected device. On an Apple Silicon host this is normally `mps`; inside
        the Linux containers this portfolio runs in, it is `cpu`, since Docker Desktop
        exposes neither CUDA nor MPS to the VM.
    """
    requested = os.environ.get(DEVICE_VAR, "").strip()
    if requested:
        logger.info("Device %s requested via %s", requested, DEVICE_VAR)
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
