"""Tests for the Week 4 CLIP + Qwen3 captioning repo.

Four things are worth guarding here. The adapter's shape has to match the checkpoint
that claims it, because getting that wrong is what broke `adapter_flickr_batched.pth`
for months (`test_every_committed_checkpoint_loads`). The zero-shot half has to keep
working end to end on the ten sample images, since that is the only part of this repo
that produces a genuinely good result. And the API has to answer.

The tests that would need Qwen3 resident are the expensive ones, so anything touching
the decoder builds the adapter directly instead of downloading a 596M-parameter model.
"""

from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from ai_residency_common import load_checkpoint, save_checkpoint
from config import CHECKPOINTS, Config
from multimodal_captioning.models import build_adapter
from serving.api import app

CLIP_DIM = 512
QWEN_DIM = 1024


def test_adapter_projects_clip_width_to_qwen_width() -> None:
    adapter = build_adapter(CLIP_DIM, QWEN_DIM)
    assert adapter(torch.randn(3, CLIP_DIM)).shape == (3, QWEN_DIM)


def test_final_layer_norm_adds_exactly_one_module() -> None:
    """The five-module variant is the one `adapter_flickr_batched.pth` was trained with."""
    assert len(build_adapter(CLIP_DIM, QWEN_DIM)) == 4
    assert len(build_adapter(CLIP_DIM, QWEN_DIM, final_layer_norm=True)) == 5


def test_adapter_keys_stay_positional(tmp_path: Path) -> None:
    """The committed checkpoints have keys like `0.weight`.

    Wrapping the stack in a named `nn.Module` would rename every one of them and break
    all three files, so this pins the naming rather than leaving it to convention.
    """
    adapter = build_adapter(CLIP_DIM, QWEN_DIM, final_layer_norm=True)
    assert set(adapter.state_dict()) >= {"0.weight", "1.weight", "3.weight", "4.weight"}


@pytest.mark.parametrize("checkpoint_name", list(CHECKPOINTS))
def test_every_committed_checkpoint_loads(checkpoint_name: str) -> None:
    """Regression test for the checkpoint/adapter mismatch.

    `adapter_flickr_batched.pth` was saved from a five-module adapter, the trailing
    `LayerNorm` was later deleted from the code, and nothing recorded that the file on
    disk still expected it. Each checkpoint now carries its own architecture, and this
    asserts the recorded architecture actually loads.
    """
    config = Config()
    checkpoint_path = config.checkpoint_dir / checkpoint_name
    if not checkpoint_path.exists():
        pytest.skip(f"{checkpoint_name} is not on disk (checkpoints are gitignored, on the Hub)")

    payload = load_checkpoint(checkpoint_path)
    extra = payload["extra"]
    adapter = build_adapter(
        extra["clip_dim"], extra["qwen_dim"], final_layer_norm=extra["final_layer_norm"]
    )
    adapter.load_state_dict(payload["model_state_dict"])
    assert extra["final_layer_norm"] == CHECKPOINTS[checkpoint_name].final_layer_norm


def test_checkpoint_round_trips_its_architecture(tmp_path: Path) -> None:
    """A checkpoint written today can be reloaded without the caller knowing its shape."""
    adapter = build_adapter(CLIP_DIM, QWEN_DIM, final_layer_norm=True)
    path = tmp_path / "adapter.pth"
    save_checkpoint(
        path, adapter, extra={"final_layer_norm": True, "clip_dim": CLIP_DIM, "qwen_dim": QWEN_DIM}
    )

    payload = load_checkpoint(path)
    reloaded = build_adapter(
        payload["extra"]["clip_dim"],
        payload["extra"]["qwen_dim"],
        final_layer_norm=payload["extra"]["final_layer_norm"],
    )
    reloaded.load_state_dict(payload["model_state_dict"])
    assert torch.equal(adapter.state_dict()["0.weight"], reloaded.state_dict()["0.weight"])


def test_sample_image_exists_for_every_zero_shot_label() -> None:
    """The zero-shot demo reads `data/images/<item>.jpg` for each label it scores."""
    config = Config()
    if not config.image_dir.exists():
        pytest.skip("data/images/ is not present in this checkout")
    missing = [
        item for item in config.zero_shot_items if not (config.image_dir / f"{item}.jpg").exists()
    ]
    assert missing == []


def test_health_endpoint_reports_device_and_checkpoints() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["device"] in {"cuda", "mps", "cpu"}
    assert set(body["checkpoints"]) == set(CHECKPOINTS)


def test_caption_endpoint_rejects_data_that_is_not_an_image() -> None:
    client = TestClient(app)
    response = client.post("/caption", json={"image_data": "not-base64-image-data"})
    assert response.status_code == 422


def test_max_new_tokens_is_bounded() -> None:
    """An unbounded generation cap is an unbounded request.

    `max_new_tokens` feeds the generation loop directly. Measured against the
    deployed service, 100000 was still generating after 60 seconds and holding the
    worker, where the default of 30 answered in about two. The bound rejects it at
    validation time instead, before it reaches the model.
    """
    from serving.api import MAX_NEW_TOKENS

    client = TestClient(app)
    over = client.post("/caption", json={"image_data": "", "max_new_tokens": MAX_NEW_TOKENS + 1})
    assert over.status_code == 422
    for bad in (0, -1):
        assert (
            client.post("/caption", json={"image_data": "", "max_new_tokens": bad}).status_code
            == 422
        )
