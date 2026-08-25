"""Tests for ai_residency_common.serving."""

import base64
import io
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from PIL import Image
from torch import nn

from ai_residency_common.serving import (
    add_health_routes,
    decode_audio,
    decode_image,
    load_model,
    resolve_required,
    retrying,
    run_app,
)


class TinyModel(nn.Module):
    """A minimal model for exercising checkpoint load."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Bound to a typed local rather than returned straight out: nn.Linear.__call__
        # is Any to mypy, and returning that from a Tensor-annotated function is what
        # the one `type: ignore` in this repo used to suppress.
        out: torch.Tensor = self.linear(x)
        return out


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
        assert error.status_code == 422
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


def test_decode_image_rejects_a_truncated_png() -> None:
    """A header-valid but truncated image must fail here, not inside the model.

    `Image.open` is lazy, so this payload used to sail through `decode_image` and
    raise several frames deeper -- turning a client error into a 500.
    """
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), "red").save(buffer, format="PNG")
    truncated = buffer.getvalue()[: len(buffer.getvalue()) // 2]

    try:
        decode_image(base64.b64encode(truncated).decode())
    except HTTPException as error:
        assert error.status_code == 422
    else:
        raise AssertionError("expected HTTPException")


def _app_with_health(*, ready: bool) -> FastAPI:
    app = FastAPI()
    add_health_routes(app, ready=lambda: ready, detail=lambda: {"checkpoint_source": "hub"})
    return app


def test_health_answers_200_even_when_not_ready() -> None:
    """Liveness must not depend on the model: a slow load is not a hung process."""
    response = TestClient(_app_with_health(ready=False)).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ready_answers_503_when_not_ready() -> None:
    """The whole point: an unusable instance must fail its readiness probe.

    Answering 200 here is what silently put checkpoint-less pods into a Service's
    endpoints, where every request they were routed reached a backend that could
    only 503.
    """
    response = TestClient(_app_with_health(ready=False)).get("/ready")
    assert response.status_code == 503
    assert response.json()["ready"] is False
    assert response.json()["status"] == "not_ready"


def test_ready_answers_200_when_ready() -> None:
    response = TestClient(_app_with_health(ready=True)).get("/ready")
    assert response.status_code == 200
    assert response.json()["ready"] is True


def test_detail_is_reported_by_both_endpoints() -> None:
    client = TestClient(_app_with_health(ready=True))
    assert client.get("/health").json()["checkpoint_source"] == "hub"
    assert client.get("/ready").json()["checkpoint_source"] == "hub"


def test_root_mirrors_health_by_default() -> None:
    """`make status` curls `/`; a backend that 404s there was reported as down."""
    assert TestClient(_app_with_health(ready=True)).get("/").status_code == 200


def test_root_can_be_left_to_the_service() -> None:
    app = FastAPI()

    @app.get("/")
    def root() -> dict[str, str]:
        return {"message": "mine"}

    add_health_routes(app, ready=lambda: True, include_root=False)
    assert TestClient(app).get("/").json() == {"message": "mine"}


def test_retrying_stops_calling_once_it_succeeds() -> None:
    """The common case: load once, then answer from memory."""
    calls = []

    def load() -> bool:
        calls.append(1)
        return True

    ready = retrying(load, min_interval_seconds=0)
    assert ready() is True
    assert ready() is True
    assert len(calls) == 1


def test_retrying_recovers_from_a_transient_failure() -> None:
    """The reason this exists.

    A checkpoint resolved once at startup makes a transient network failure permanent:
    the pod answers 503 for the rest of its life, and liveness cannot rescue it because
    the process is fine. Readiness retries instead.
    """
    attempts = {"n": 0}

    def load() -> bool:
        attempts["n"] += 1
        return attempts["n"] >= 3

    ready = retrying(load, min_interval_seconds=0)
    assert ready() is False
    assert ready() is False
    assert ready() is True
    assert ready() is True
    assert attempts["n"] == 3


def test_retrying_rate_limits_attempts_after_the_first() -> None:
    """A readiness probe runs every ten seconds; re-downloading a multi-gigabyte
    checkpoint that often would be worse than staying down.

    The *first* call must still attempt. An earlier version tracked the last attempt
    as 0.0 and compared against `time.monotonic()`, which counts from an arbitrary
    origin -- so on a freshly booted machine the first attempt was rate-limited away
    before it happened. It passed on a laptop with days of uptime and failed on a CI
    runner, which is the wrong way round for a bug that only bites new nodes.
    """
    calls = []

    def load() -> bool:
        calls.append(1)
        return False

    ready = retrying(load, min_interval_seconds=3600)
    ready()
    ready()
    ready()
    assert len(calls) == 1


def test_retrying_treats_a_raising_loader_as_not_ready() -> None:
    """A probe must never 500. Whatever the loader does, /ready answers."""

    def load() -> bool:
        raise RuntimeError("hub unreachable")

    ready = retrying(load, min_interval_seconds=0)
    assert ready() is False
    assert ready() is False


def test_ready_flips_from_503_to_200_when_the_load_recovers() -> None:
    """The whole point, through the endpoint a probe actually calls.

    The unit tests above cover `retrying` directly. This covers the wiring: a service
    whose checkpoint was unreachable at startup answers 503, and starts answering 200
    once the load succeeds -- without the pod being restarted, which is what would
    otherwise be required.
    """
    attempts = {"n": 0}
    reachable = {"yes": False}

    def load() -> bool:
        attempts["n"] += 1
        return reachable["yes"]

    app = FastAPI()
    add_health_routes(app, ready=retrying(load, min_interval_seconds=0))
    client = TestClient(app)

    assert client.get("/ready").status_code == 503
    assert client.get("/health").status_code == 200, "liveness must not depend on the load"

    reachable["yes"] = True

    assert client.get("/ready").status_code == 200
    assert client.get("/ready").status_code == 200
    # Three probes, but only the two before success called the loader.
    assert attempts["n"] == 2
