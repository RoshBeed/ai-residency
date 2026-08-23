"""Tests for ai_residency_common.serving."""

import base64
from pathlib import Path

import torch
from fastapi import HTTPException
from torch import nn

from ai_residency_common.serving import (
    decode_audio,
    decode_image,
    load_model,
    resolve_required,
    run_app,
)


class TinyModel(nn.Module):
    """A minimal model for exercising checkpoint load."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)  # type: ignore[no-any-return]


def test_resolve_required_returns_a_local_file(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.pth"
    artifact.write_text("x")

    path, source = resolve_required(artifact, None)
    assert path == artifact
    assert source == "local"


def test_resolve_required_raises_when_missing(tmp_path: Path) -> None:
    missing = tmp_path / "missing.pth"

    try:
        resolve_required(missing, None, rebuild_hint="Build it with `make train`.")
    except FileNotFoundError as error:
        message = str(error)
        assert "missing.pth" in message
        assert "Build it with `make train`." in message
    else:
        raise AssertionError("expected FileNotFoundError")


def test_load_model_populates_weights_and_evals(tmp_path: Path) -> None:
    model = TinyModel()
    original_bias = float(model.linear.bias[0].item())

    checkpoint = tmp_path / "model.pth"
    state = TinyModel()
    with torch.no_grad():
        state.linear.bias[0] = 99.0
    torch.save(
        {
            "model_state_dict": state.state_dict(),
            "optimizer_state_dict": None,
            "epoch": None,
            "metrics": {},
            "extra": {},
        },
        checkpoint,
    )

    loaded, source = load_model(checkpoint, None, model)
    assert source == "local"
    assert loaded is model
    assert float(model.linear.bias[0].item()) == 99.0
    assert not model.training
    assert model.linear.bias[0].item() != original_bias


def test_run_app_is_importable() -> None:
    assert callable(run_app)


def test_decode_audio_round_trips_bytes() -> None:
    raw = b"\x00\x01\x02\xff"
    decoded = decode_audio(base64.b64encode(raw).decode())
    assert decoded == raw


def test_decode_audio_raises_on_garbage() -> None:
    try:
        decode_audio("not base64!!")
    except HTTPException as error:
        assert error.status_code == 400
    else:
        raise AssertionError("expected HTTPException")


def test_decode_image_round_trips_a_png() -> None:
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), "red").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()

    decoded = decode_image(encoded)
    assert decoded.size == (4, 4)


def test_decode_image_raises_on_garbage() -> None:
    try:
        decode_image("not an image!!")
    except HTTPException as error:
        assert error.status_code == 422
    else:
        raise AssertionError("expected HTTPException")
