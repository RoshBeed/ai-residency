"""Device selection, including the environment override."""

import pytest
import torch

from ai_residency_common.device import DEVICE_VAR, get_device


def test_the_default_is_a_real_torch_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """Whatever host this runs on, the answer is a device torch will accept.

    Three earlier copies of this function returned bare strings, which works until
    something compares against a `torch.device` and silently never matches.
    """
    monkeypatch.delenv(DEVICE_VAR, raising=False)
    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in {"cuda", "mps", "cpu"}
    torch.zeros(1, device=device)


def test_the_device_can_be_overridden_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An accelerator that cannot fit the run has to be refusable without a code edit.

    PPO on this Mac dies partway through a step with `MPS backend out of memory
    (allocated 42.45 GiB, max allowed 42.43 GiB)`, while the same work on CPU has real
    RAM and no watermark ceiling to hit.
    """
    monkeypatch.setenv(DEVICE_VAR, "cpu")
    assert get_device() == torch.device("cpu")


def test_an_empty_override_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    """`AI_RESIDENCY_DEVICE=` from an unset shell variable must not become a device.

    `torch.device("")` raises, so without the strip-and-check an exported-but-empty
    variable would take down every run on the machine that had it set.
    """
    monkeypatch.delenv(DEVICE_VAR, raising=False)
    default = get_device()
    for blank in ("", "   "):
        monkeypatch.setenv(DEVICE_VAR, blank)
        assert get_device() == default


def test_an_unavailable_override_is_not_quietly_corrected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run that named a device gets that device, or an error -- never a substitute.

    Falling back would mean a run that asked for an accelerator trains on CPU and
    reports nothing, which is the shape of several hours quietly wasted.
    """
    monkeypatch.setenv(DEVICE_VAR, "cuda")
    assert get_device() == torch.device("cuda")
