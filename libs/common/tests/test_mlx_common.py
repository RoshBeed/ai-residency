"""Tests for the shared helpers.

These cover the behaviours the six repos actually depend on, including the two that
exist specifically to fix real bugs: seeding making a run reproducible, and loading
an MPS-era checkpoint onto CPU.
"""

from pathlib import Path

import torch
from torch import nn

from ai_residency_common import (
    BaseConfig,
    get_device,
    get_logger,
    load_checkpoint,
    save_checkpoint,
    set_seed,
)


def test_get_device_returns_a_real_device() -> None:
    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in {"cuda", "mps", "cpu"}
    # A tensor must actually be placeable on whatever we selected.
    assert torch.zeros(1, device=device).device.type == device.type


def test_get_logger_is_namespaced_and_reusable() -> None:
    logger = get_logger("ai_residency_common.tests")
    assert logger.name == "ai_residency_common.tests"
    assert get_logger("ai_residency_common.tests") is logger


def test_set_seed_makes_initialisation_reproducible() -> None:
    set_seed(42)
    first = nn.Linear(8, 4).weight.detach().clone()
    set_seed(42)
    second = nn.Linear(8, 4).weight.detach().clone()
    assert torch.equal(first, second)

    set_seed(43)
    third = nn.Linear(8, 4).weight.detach().clone()
    assert not torch.equal(first, third)


def test_checkpoint_round_trip_restores_weights(tmp_path: Path) -> None:
    torch.manual_seed(0)
    saved = nn.Linear(4, 2)
    optimizer = torch.optim.Adam(saved.parameters(), lr=1e-3)
    path = tmp_path / "nested" / "model.pth"

    save_checkpoint(
        path,
        saved,
        optimizer=optimizer,
        epoch=5,
        metrics={"test_loss": 1.25},
        extra={"input_dim": 4},
    )
    assert path.exists(), "save_checkpoint should create parent directories"

    loaded = nn.Linear(4, 2)
    assert not torch.equal(saved.weight, loaded.weight)

    payload = load_checkpoint(path, loaded)
    assert torch.equal(saved.weight, loaded.weight)
    assert payload["epoch"] == 5
    assert payload["metrics"]["test_loss"] == 1.25
    assert payload["extra"]["input_dim"] == 4


def test_load_checkpoint_defaults_to_cpu(tmp_path: Path) -> None:
    """Checkpoints must load on a host without the accelerator they were trained on.

    This is the failure that motivated the `map_location="cpu"` default: weights
    trained on MPS carry `mps:0` metadata and blow up inside the CPU-only container.
    """
    path = tmp_path / "model.pth"
    save_checkpoint(path, nn.Linear(4, 2))

    payload = load_checkpoint(path)
    for tensor in payload["model_state_dict"].values():
        assert tensor.device.type == "cpu"


def test_base_config_defaults_and_subclassing() -> None:
    config = BaseConfig()
    assert config.seed == 42
    assert config.checkpoint_dir == Path("checkpoints")
    assert isinstance(config.device, torch.device)

    from dataclasses import dataclass

    @dataclass
    class RepoConfig(BaseConfig):
        embedding_dim: int = 128

    repo_config = RepoConfig(num_epochs=3)
    assert repo_config.embedding_dim == 128
    assert repo_config.num_epochs == 3
    assert repo_config.seed == 42


def test_save_checkpoint_can_store_only_the_trained_weights(tmp_path: Path) -> None:
    """An adapter on a frozen backbone must not drag the backbone into the file."""
    model = nn.Linear(64, 64)
    trained_only = {"weight": model.weight.detach()}

    full = tmp_path / "full.pth"
    partial = tmp_path / "partial.pth"
    save_checkpoint(full, model)
    save_checkpoint(partial, model, state_dict=trained_only)

    assert set(load_checkpoint(partial)["model_state_dict"]) == {"weight"}
    assert partial.stat().st_size < full.stat().st_size
