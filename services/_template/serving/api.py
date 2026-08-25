"""FastAPI service.

Canonical location for the HTTP backend in every repo: `serving/api.py`, exposing
`app`. The portfolio frontend calls these services by their route, so keep the
request and response models explicit.

Binary inputs (images, audio) go base64-encoded in a pydantic field rather than as
multipart uploads, matching the convention every other service here uses.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from pydantic import BaseModel

from ai_residency_common import (
    add_health_routes,
    get_device,
    get_logger,
    load_checkpoint,
    resolve_checkpoint,
    retrying,
)
from config import Config
from template.models import Predictor

__all__ = ["app"]

logger = get_logger(__name__)

_config = Config()
_device = get_device()
_model: Predictor | None = None
_checkpoint_source = "none"


class PredictRequest(BaseModel):
    """Input for `/predict`."""

    features: list[float]


class PredictResponse(BaseModel):
    """Output of `/predict`."""

    prediction: float


def _get_model() -> Predictor:
    """Return the served model, loading it from the checkpoint on first request."""
    global _model, _checkpoint_source
    if _model is None:
        model = Predictor(_config).to(_device)
        # Falls back to the Hub when this host has no local weights, which is what
        # lets the service run somewhere other than where training happened.
        path, _checkpoint_source = resolve_checkpoint(_config.checkpoint_path, _config.hf_repo_id)
        if path is not None:
            load_checkpoint(path, model)
        else:
            logger.warning("No checkpoint locally or on the Hub; serving an untrained model")
        model.eval()
        _model = model
    return _model


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Resolve the checkpoint once at startup rather than on the first request.

    In a lifespan, not at module scope: importing this module must not touch the
    network, or the service cannot be imported by a test, a smoke check, or anything
    else that only wants to inspect the app. A failure here is logged and left for
    `/ready` to report as 503, rather than killing the process.
    """
    ensure_loaded()
    yield


def ensure_loaded() -> bool:
    """Load the checkpoint if it is not loaded, and report whether it is.

    Wired into `/ready` through `retrying`, so a resolve that failed at startup is
    attempted again instead of leaving the pod not-ready for good. Copy this shape
    into a new service.
    """
    if _checkpoint_source != "none":
        return True
    try:
        _get_model()
    except Exception as error:
        logger.error("Could not load the model, /ready will report 503: %s", error)
        return False
    return _checkpoint_source != "none"


app = FastAPI(title="AI Residency template service", lifespan=lifespan)


# The two-endpoint contract every backend here exposes. `/health` is liveness and
# always answers 200; `/ready` answers 503 while there are no usable weights, which is
# what keeps a Kubernetes Service from routing to a pod that would only 503 anyway.
# Copy this shape into a new service rather than writing a bespoke health endpoint.
add_health_routes(
    app,
    # Retried rather than decided once. A resolve that failed at startup -- a DNS blip
    # during a mass restart, say -- would otherwise leave this pod not-ready for the
    # rest of its life, with liveness unable to help because the process itself is
    # perfectly healthy. `retrying` re-attempts, at most once a minute, from /ready.
    ready=retrying(ensure_loaded),
    detail=lambda: {
        "device": str(_device),
        "checkpoint_loaded": _checkpoint_source != "none",
        "checkpoint_source": _checkpoint_source,
    },
)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Score a single example.

    Args:
        request: One feature vector of length `Config.input_dim`.

    Returns:
        The model's scalar prediction.
    """
    model = _get_model()
    features = torch.tensor([request.features], dtype=torch.float32, device=_device)
    with torch.no_grad():
        prediction = model(features).item()
    return PredictResponse(prediction=prediction)


def main() -> None:
    """Entry point for `make serve`."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
