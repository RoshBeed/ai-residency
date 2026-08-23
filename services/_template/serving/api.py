"""FastAPI service.

Canonical location for the HTTP backend in every repo: `serving/api.py`, exposing
`app`. The portfolio frontend calls these services by their route, so keep the
request and response models explicit.

Binary inputs (images, audio) go base64-encoded in a pydantic field rather than as
multipart uploads, matching the convention the existing week repos already use.
"""

from typing import Any

import torch
from ai_residency_common import get_device, get_logger, load_checkpoint, resolve_checkpoint
from fastapi import FastAPI
from pydantic import BaseModel

from config import Config
from template.models import Predictor

__all__ = ["app"]

logger = get_logger(__name__)

app = FastAPI(title="AI Residency template service")

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


@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness check used by compose and the Helm chart.

    Reports where the weights came from, so a service that quietly fell back to an
    untrained model is visible rather than answering with plausible nonsense.
    """
    _get_model()
    return {
        "status": "ok",
        "device": str(_device),
        "checkpoint_loaded": _checkpoint_source != "none",
        "checkpoint_source": _checkpoint_source,
    }


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
