"""FastAPI service for the single-digit Vision Transformer classifier.

Canonical location for the HTTP backend in every service in this portfolio:
`serving/api.py`, exposing `app`.

Images arrive base64-encoded in a pydantic field rather than as multipart uploads,
matching the convention every other service uses.
"""

import base64
import io
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from PIL.Image import Image as PILImage
from pydantic import BaseModel

from ai_residency_common import (
    add_health_routes,
    get_device,
    get_logger,
    load_checkpoint,
    resolve_checkpoint,
    retrying,
    run_app,
)
from config import Config
from vit_classifier.classifier import VisionTransformer

__all__ = ["app"]


logger = get_logger(__name__)


MNIST_MEAN = 0.1307


MNIST_STD = 0.3081


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Load the checkpoints when the service starts. /ready retries if this fails."""
    ensure_loaded()
    yield


app = FastAPI(title="Vision Transformer API", version="1.0.0", lifespan=lifespan)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


_config = Config()


_device = get_device()


_classifier: VisionTransformer | None = None


_checkpoint_source = "none"


class ImageRequest(BaseModel):
    """A single image to classify."""

    image_data: str
    # Ignored: this service has one model. Kept so payloads written against the
    # combined service still validate.
    model_type: str = "vit"


class PredictionResponse(BaseModel):
    """One model's answer, plus a trace of how it got there."""

    pred: str
    confidence: float
    model_type: str
    class_probabilities: dict[str, float] | None = None
    img_size: int = 0
    patch_size: int = 0


def load_models() -> None:
    """Load the classifier checkpoint, pulling it from the Hub if it is not on disk.

    The fallback is what lets this service run on a host that never trained here: the
    checkpoint is gitignored, so a fresh clone or a separate inference box would
    otherwise start with no model and answer 503 to everything.
    """
    global _classifier, _checkpoint_source

    logger.info("Serving on %s", _device)

    classifier_path, _checkpoint_source = resolve_checkpoint(
        _config.classifier_checkpoint_path, _config.hf_repo_id
    )
    if classifier_path is not None:
        classifier = VisionTransformer(_config.classifier_arch)
        load_checkpoint(classifier_path, classifier)
        _classifier = classifier.to(_device).eval()
    else:
        logger.warning("No classifier checkpoint locally or on the Hub")


def ensure_loaded() -> bool:
    """Load the checkpoint if it is not already loaded, and report whether it is.

    Idempotent, and safe to call from a readiness probe: it returns immediately once
    the model is in memory, and re-attempts the resolve otherwise.
    """
    if _classifier is not None:
        return True
    try:
        load_models()
    except (FileNotFoundError, KeyError, RuntimeError) as error:
        logger.error("Could not load the checkpoint, /ready stays 503: %s", error)
        return False
    return _classifier is not None


def preprocess_image(image_data: str, img_size: int) -> torch.Tensor:
    """Decode a base64 image and resize it for one specific model.

    Decoding has to end in exactly the tensor the training loaders produce, which
    means the MNIST normalisation as well as the resize. Skipping it does not raise;
    it just makes the served model wrong.

    `img_size` is required rather than defaulted because the two models take
    different resolutions: 28 for the classifier, 84 for the seq2seq model. Sending a
    28x28 image to the seq2seq model does not raise — its patch convolution
    (kernel 16, stride 16) just reads one corner and silently returns noise — so
    getting this wrong is invisible unless the caller is forced to state the size.

    Args:
        image_data: Base64-encoded image bytes.
        img_size: Side length the target model expects.

    Returns:
        Normalised tensor of shape `(1, 1, img_size, img_size)` on the serving device.

    Raises:
        HTTPException: If the payload is not a decodable image.
    """
    try:
        image: PILImage = Image.open(io.BytesIO(base64.b64decode(image_data)))
    except Exception as error:
        raise HTTPException(
            status_code=400, detail=f"Image preprocessing failed: {error}"
        ) from error

    if image.mode != "L":
        image = image.convert("L")
    image = image.resize((img_size, img_size))

    array = np.array(image, dtype=np.float32) / 255.0
    array = (array - MNIST_MEAN) / MNIST_STD
    return torch.from_numpy(array).unsqueeze(0).unsqueeze(0).to(_device)


def predict_vit(image_tensor: torch.Tensor) -> tuple[str, float, dict[str, float]]:
    """Classify a single digit.

    Args:
        image_tensor: Preprocessed 28x28 image.

    Returns:
        The predicted digit, the model's confidence in it, and the full ten-way
        probability distribution it was chosen from (`{"0": p, ..., "9": p}`).

    Raises:
        HTTPException: If the classifier checkpoint was never loaded.
    """
    if _classifier is None:
        raise HTTPException(status_code=503, detail="ViT model not loaded")

    with torch.no_grad():
        logits = _classifier(image_tensor)
        probabilities = F.softmax(logits, dim=-1)
        predicted_class = int(torch.argmax(logits, dim=-1).item())
        confidence = float(probabilities.max().item())
        class_probabilities = {
            str(class_index): float(probabilities[0, class_index].item())
            for class_index in range(probabilities.shape[-1])
        }

    return str(predicted_class), confidence, class_probabilities


add_health_routes(
    app,
    # Retried rather than decided once. A checkpoint that could not be resolved at
    # startup -- a DNS blip during a mass restart, say -- would otherwise leave this
    # pod not-ready for the rest of its life, with liveness unable to help because the
    # process itself is perfectly healthy. `retrying` re-attempts, at most once a
    # minute, from /ready.
    ready=retrying(ensure_loaded),
    detail=lambda: {
        "message": "Vision Transformer classifier API",
        "model_loaded": _classifier is not None,
        "checkpoint_loaded": _classifier is not None,
        "checkpoint_source": _checkpoint_source,
        "device": str(_device),
    },
)


@app.get("/models")
async def get_models() -> dict[str, Any]:
    """Describe the model this service exposes."""
    arch = _config.classifier_arch
    return {
        "id": "vit",
        "name": "Vision Transformer",
        "type": "classification",
        "loaded": _classifier is not None,
        "img_size": arch.img_size,
        "patch_size": arch.patch_size,
        "device": str(_device),
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: ImageRequest) -> PredictionResponse:
    """Classify the digit in an image.

    Args:
        request: The base64-encoded image.

    Returns:
        The predicted digit, its confidence, and the ten-way distribution.
    """
    arch = _config.classifier_arch
    image_tensor = preprocess_image(request.image_data, arch.img_size)
    prediction, confidence, class_probabilities = predict_vit(image_tensor)
    return PredictionResponse(
        pred=prediction,
        confidence=confidence,
        model_type="vit",
        class_probabilities=class_probabilities,
        img_size=arch.img_size,
        patch_size=arch.patch_size,
    )


@app.post("/predict/vit", response_model=PredictionResponse)
async def predict_vit_endpoint(request: ImageRequest) -> PredictionResponse:
    """Alias of `/predict`, kept because callers were written against the route name
    this service used when it also served the seq2seq model.

    Args:
        request: The base64-encoded image.

    Returns:
        The classifier's prediction.
    """
    return await predict(request)


def main() -> None:
    """Entry point for `make serve`."""
    run_app(app, host=_config.api_host, port=_config.api_port)


if __name__ == "__main__":
    main()
