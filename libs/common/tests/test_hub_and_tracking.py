"""Tests for artifact hosting and experiment tracking.

These concentrate on the no-credentials paths, because that is the behaviour the
portfolio depends on: someone cloning a repo without these accounts must still be
able to run training. The upload/download paths themselves are thin wrappers over
`huggingface_hub` and are not worth mocking out in detail here.
"""

from pathlib import Path

import pytest
from torch import nn

from ai_residency_common import (
    get_hf_token,
    init_tracking,
    log_metrics,
    push_checkpoint,
    resolve_checkpoint,
    save_checkpoint,
)
from ai_residency_common.tracking import finish_tracking


@pytest.fixture
def no_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neither an environment token nor a machine that is logged in.

    `get_hf_token` falls back to the credential `huggingface-cli login` writes, so
    clearing the environment alone is no longer enough to simulate an unconfigured
    machine -- and on a developer's laptop these tests would otherwise find a real
    token and push to the Hub for real.
    """
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    monkeypatch.setattr("ai_residency_common.hub.get_token", lambda: None)


def test_get_hf_token_reads_either_variable(
    monkeypatch: pytest.MonkeyPatch, no_credentials: None
) -> None:
    assert get_hf_token() is None

    monkeypatch.setenv("HUGGINGFACE_TOKEN", "legacy-name")
    assert get_hf_token() == "legacy-name"

    monkeypatch.setenv("HF_TOKEN", "preferred-name")
    assert get_hf_token() == "preferred-name"


def test_push_checkpoint_skips_without_token(tmp_path: Path, no_credentials: None) -> None:
    """Training must survive having no Hub credentials, not crash."""
    path = tmp_path / "model.pth"
    save_checkpoint(path, nn.Linear(2, 2))

    assert push_checkpoint("roshbeed/does-not-matter", path) is False
    assert path.exists(), "the local checkpoint must survive a skipped push"


def test_push_checkpoint_survives_a_rejected_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A finished training run must not be thrown away over a credentials problem.

    This is the failure that motivated the behaviour: the Hub rejects repo creation
    under a namespace whose case does not match the token's, and it does so after
    training has already completed.
    """
    monkeypatch.setenv("HF_TOKEN", "a-token-that-will-be-rejected")

    path = tmp_path / "model.pth"
    save_checkpoint(path, nn.Linear(2, 2))

    def reject(*args: object, **kwargs: object) -> None:
        raise RuntimeError("403 Forbidden")

    monkeypatch.setattr("ai_residency_common.hub.HfApi.create_repo", reject)

    assert push_checkpoint("roshbeed/does-not-matter", path) is False
    assert path.exists(), "the local checkpoint must survive a failed push"


def test_init_tracking_disabled_when_no_project() -> None:
    assert init_tracking(None) is False


def test_log_metrics_is_safe_without_a_run() -> None:
    """Training code calls this unconditionally, so it must no-op when untracked."""
    finish_tracking()
    log_metrics({"train_loss": 1.0})
    log_metrics({"train_loss": 1.0}, step=3)


def test_resolve_checkpoint_prefers_a_local_file(tmp_path: Path) -> None:
    path = tmp_path / "model.pth"
    save_checkpoint(path, nn.Linear(2, 2))

    resolved = resolve_checkpoint(path, "roshbeed/should-not-be-consulted")
    assert resolved == (path, "local")


def test_resolve_checkpoint_reports_nothing_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A host with neither a local file nor a repo must say so, not pretend."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)

    assert resolve_checkpoint(tmp_path / "absent.pth", None) == (None, "none")


def test_resolve_checkpoint_falls_back_to_the_hub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The inference-host path: no local weights, so fetch them."""
    downloaded = tmp_path / "pulled.pth"
    save_checkpoint(downloaded, nn.Linear(2, 2))

    def fake_download(**kwargs: object) -> str:
        assert kwargs["filename"] == "absent.pth"
        return str(downloaded)

    monkeypatch.setattr("ai_residency_common.hub.hf_hub_download", fake_download)

    resolved = resolve_checkpoint(tmp_path / "absent.pth", "roshbeed/somewhere")
    assert resolved == (downloaded, "hub")


def test_resolve_checkpoint_can_read_a_dataset_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Training inputs live in dataset repos, weights in model repos."""
    downloaded = tmp_path / "posts.json"
    downloaded.write_text("[]")
    seen: dict[str, object] = {}

    def fake_download(**kwargs: object) -> str:
        seen.update(kwargs)
        return str(downloaded)

    monkeypatch.setattr("ai_residency_common.hub.hf_hub_download", fake_download)

    resolved = resolve_checkpoint(
        tmp_path / "absent.json", "roshbeed/some-data", repo_type="dataset"
    )
    assert resolved.source == "hub"
    assert seen["repo_type"] == "dataset"


def test_get_hf_token_falls_back_to_the_local_hub_login(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`huggingface-cli login` writes a token that every other Hub call already
    picks up. Ours did not, so a logged-in machine trained for hours and then
    warned that it was keeping the checkpoint local."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    monkeypatch.setattr("ai_residency_common.hub.get_token", lambda: "cached-login")

    assert get_hf_token() == "cached-login"
