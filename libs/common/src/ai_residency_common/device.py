"""Device selection.

Every repo in this portfolio picked its own accelerator in its own way, and each
version was subtly different: one omitted CUDA entirely, one returned bare strings
instead of `torch.device`, one repeated the same block in three files. This is the
single implementation they all use now.
"""

import torch

__all__ = ["get_device"]


def get_device() -> torch.device:
    """Return the best available device, preferring CUDA, then MPS, then CPU.

    Returns:
        The selected device. On an Apple Silicon host this is normally `mps`; inside
        the Linux containers this portfolio runs in, it is `cpu`, since Docker Desktop
        exposes neither CUDA nor MPS to the VM.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
