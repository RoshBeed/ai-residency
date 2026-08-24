"""Tests for `.env` loading and `AI_RESIDENCY_*` config overrides.

These two exist so a training run can be shortened for a smoke test and lengthened on
a rented GPU box without editing source, and so credentials reach the process without
being exported by hand.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import pytest

from ai_residency_common import BaseConfig, load_env


@dataclass
class _RepoConfig(BaseConfig):
    """Stands in for a service's own Config, with its own extra fields."""

    num_samples: int = 100
    train_on_cpu: bool = True
    dataset_keys: tuple[str, ...] = ("crema_d", "emodb")


def test_no_overrides_leaves_defaults_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("AI_RESIDENCY_NUM_EPOCHS", "AI_RESIDENCY_SEED", "AI_RESIDENCY_NUM_SAMPLES"):
        monkeypatch.delenv(name, raising=False)

    config = _RepoConfig()
    assert config.num_epochs == 10
    assert config.seed == 42
    assert config.num_samples == 100


def test_overrides_base_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_RESIDENCY_NUM_EPOCHS", "1")
    monkeypatch.setenv("AI_RESIDENCY_LEARNING_RATE", "0.05")
    monkeypatch.setenv("AI_RESIDENCY_CHECKPOINT_DIR", "/tmp/ckpts")

    config = BaseConfig()
    assert config.num_epochs == 1
    assert config.learning_rate == 0.05
    assert config.checkpoint_dir == Path("/tmp/ckpts")


def test_overrides_subclass_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    """A repo's own hyperparameters are covered without it opting in."""
    monkeypatch.setenv("AI_RESIDENCY_NUM_SAMPLES", "8")
    monkeypatch.setenv("AI_RESIDENCY_TRAIN_ON_CPU", "false")
    monkeypatch.setenv("AI_RESIDENCY_DATASET_KEYS", "emodb")

    config = _RepoConfig()
    assert config.num_samples == 8
    assert config.train_on_cpu is False
    assert config.dataset_keys == ("emodb",)


def test_optional_field_set_and_cleared(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_RESIDENCY_HF_REPO_ID", "roshbeed/somewhere-else")
    assert BaseConfig().hf_repo_id == "roshbeed/somewhere-else"

    # An empty value clears it, which is how a run opts out of pushing.
    monkeypatch.setenv("AI_RESIDENCY_HF_REPO_ID", "")
    assert BaseConfig().hf_repo_id is None


def test_invalid_override_names_the_field(monkeypatch: pytest.MonkeyPatch) -> None:
    """A typo must fail loudly rather than silently training on a default."""
    monkeypatch.setenv("AI_RESIDENCY_NUM_EPOCHS", "not-a-number")

    with pytest.raises(ValueError, match="AI_RESIDENCY_NUM_EPOCHS"):
        BaseConfig()


def test_load_env_reads_a_nearby_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AI_RESIDENCY_TEST_ONLY_TOKEN", raising=False)
    (tmp_path / ".env").write_text("AI_RESIDENCY_TEST_ONLY_TOKEN=from-the-file\n")
    monkeypatch.chdir(tmp_path)

    assert tmp_path / ".env" in load_env()
    assert os.environ["AI_RESIDENCY_TEST_ONLY_TOKEN"] == "from-the-file"


def test_a_partial_env_does_not_shadow_the_one_above_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real bug: a repo .env holding only its own settings hid the shared token.

    Loading stopped at the first file found, so the one repo with a Redis-only .env
    silently trained without Hub credentials while the other five picked them up.
    """
    monkeypatch.delenv("AI_RESIDENCY_TEST_ONLY_TOKEN", raising=False)
    monkeypatch.delenv("AI_RESIDENCY_TEST_ONLY_REDIS", raising=False)

    (tmp_path / ".env").write_text("AI_RESIDENCY_TEST_ONLY_TOKEN=shared-credential\n")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".env").write_text("AI_RESIDENCY_TEST_ONLY_REDIS=localhost\n")
    monkeypatch.chdir(repo)

    load_env()

    assert os.environ["AI_RESIDENCY_TEST_ONLY_REDIS"] == "localhost"
    assert os.environ["AI_RESIDENCY_TEST_ONLY_TOKEN"] == "shared-credential"


def test_the_nearest_env_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AI_RESIDENCY_TEST_ONLY_TOKEN", raising=False)

    (tmp_path / ".env").write_text("AI_RESIDENCY_TEST_ONLY_TOKEN=from-the-parent\n")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".env").write_text("AI_RESIDENCY_TEST_ONLY_TOKEN=from-the-repo\n")
    monkeypatch.chdir(repo)

    load_env()

    assert os.environ["AI_RESIDENCY_TEST_ONLY_TOKEN"] == "from-the-repo"


def test_load_env_never_overrides_real_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Containers get their values from compose or a Secret, not from a file."""
    (tmp_path / ".env").write_text("AI_RESIDENCY_TEST_ONLY_TOKEN=from-the-file\n")
    monkeypatch.setenv("AI_RESIDENCY_TEST_ONLY_TOKEN", "from-the-environment")
    monkeypatch.chdir(tmp_path)

    load_env()

    assert os.environ["AI_RESIDENCY_TEST_ONLY_TOKEN"] == "from-the-environment"
