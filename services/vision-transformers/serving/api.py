"""FastAPI service for both Vision Transformer models.

Canonical location for the HTTP backend in every repo in this portfolio:
`serving/api.py`, exposing `app`. The portfolio hub calls these services by
route, so the request and response models stay explicit.

`/predict/seq2seq/stream` does the same digit-by-digit generation as `/predict/seq2seq`
but relays each decoded digit as a Server-Sent Event the moment the seq2seq model
emits it, so the frontend can show the number being read digit by digit.

Images arrive base64-encoded in a pydantic field rather than as multipart uploads,
matching the convention the other week repos use.
"""

import base64
import io
import queue
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image
from PIL.Image import Image as PILImage
from pydantic import BaseModel

from ai_residency_common import (
    get_device,
    get_logger,
    load_checkpoint,
    resolve_checkpoint,
    run_app,
    stream_sse,
)
from config import Config
from models.classifier.classifier import VisionTransformer
from models.seq2seq.seq2seq import Seq2SeqVisionTransformer

__all__ = ["app"]

logger = get_logger(__name__)

# The normalisation both dataset wrappers apply during training. Serving has to apply
# it too: without it the classifier drops from 95.3% to 15.6% on the same MNIST test
# batch, because the model is being shown values it was never trained on.
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Load the checkpoints once, when the service starts."""
    load_models()
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
_seq2seq: Seq2SeqVisionTransformer | None = None
_checkpoint_sources: dict[str, str] = {"vit": "none", "seq2seq": "none"}


class ImageRequest(BaseModel):
    """A single image to classify."""

    image_data: str
    model_type: str = "vit"


class DigitTrace(BaseModel):
    """One decoded digit's own confidence, as the seq2seq model emitted it."""

    digit: str
    confidence: float


class PredictionResponse(BaseModel):
    """One model's answer, plus a trace of how it got there."""

    pred: str
    confidence: float
    model_type: str
    class_probabilities: dict[str, float] | None = None
    digit_trace: list[DigitTrace] | None = None
    img_size: int = 0
    patch_size: int = 0


class BothModelsResponse(BaseModel):
    """Both models' answers for the same image."""

    vit_prediction: dict[str, Any]
    seq2seq_prediction: dict[str, Any]
    processing_time: float


def load_models() -> None:
    """Load both checkpoints, pulling either from the Hub if it is not on disk.

    The fallback is what lets this service run on a host that never trained here:
    both checkpoints are gitignored, so a fresh clone or a separate inference box
    would otherwise start with neither model and answer 503 to everything.
    """
    global _classifier, _seq2seq

    logger.info("Serving on %s", _device)

    classifier_path, _checkpoint_sources["vit"] = resolve_checkpoint(
        _config.classifier_checkpoint_path, _config.hf_repo_id
    )
    if classifier_path is not None:
        classifier = VisionTransformer(_config.classifier_arch)
        load_checkpoint(classifier_path, classifier)
        _classifier = classifier.to(_device).eval()
    else:
        logger.warning("No classifier checkpoint locally or on the Hub")

    seq2seq_path, _checkpoint_sources["seq2seq"] = resolve_checkpoint(
        _config.seq2seq_checkpoint_path, _config.hf_repo_id
    )
    if seq2seq_path is not None:
        seq2seq = Seq2SeqVisionTransformer(_config.seq2seq_arch)
        load_checkpoint(seq2seq_path, seq2seq)
        _seq2seq = seq2seq.to(_device).eval()
    else:
        logger.warning("No seq2seq checkpoint locally or on the Hub")


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


def predict_seq2seq(image_tensor: torch.Tensor) -> tuple[str, float, list[DigitTrace]]:
    """Read a multi-digit number off an image.

    Confidence is the mean probability the model assigned to each digit it emitted,
    recovered by re-running the decoder in teacher-forcing mode over the sequence it
    just generated. That is the same idea as the classifier's softmax confidence,
    extended across a sequence.

    Args:
        image_tensor: Preprocessed 84x84 image.

    Returns:
        The predicted number, the mean per-digit confidence, and the per-digit trace
        (each decoded digit alongside its own confidence, in decoding order).

    Raises:
        HTTPException: If the seq2seq checkpoint was never loaded.
    """
    if _seq2seq is None:
        raise HTTPException(status_code=503, detail="Seq2Seq model not loaded")

    arch = _config.seq2seq_arch
    with torch.no_grad():
        generated = _seq2seq.generate(
            image_tensor,
            start_token=arch.start_token,
            max_length=arch.max_seq_length,
        )

        digits: list[str] = []
        digit_positions: list[int] = []
        for position, token in enumerate(generated[0].tolist()):
            if token == arch.start_token:
                continue
            if token == arch.eos_token:
                break
            digits.append(str(token))
            digit_positions.append(position)

        prediction = "".join(digits) if digits else "0"

        if not digit_positions:
            return prediction, 0.0, []

        logits = _seq2seq.decode(_seq2seq.encode(image_tensor), generated[:, :-1])
        probabilities = F.softmax(logits, dim=-1)
        # logits[:, i, :] predicts the token at generated[:, i + 1], so the
        # probability of the digit at `position` lives at logits[:, position - 1, :].
        token_probabilities = [
            float(probabilities[0, position - 1, generated[0, position]].item())
            for position in digit_positions
        ]
        confidence = sum(token_probabilities) / len(token_probabilities)
        digit_trace = [
            DigitTrace(digit=digit, confidence=token_confidence)
            for digit, token_confidence in zip(digits, token_probabilities, strict=True)
        ]

    return prediction, confidence, digit_trace


@app.get("/health")
@app.get("/")
async def root() -> dict[str, Any]:
    """Health check, also reporting which models are live and where they came from."""
    return {
        "message": "Vision Transformer API",
        "status": "ok",
        "models_loaded": {"vit": _classifier is not None, "seq2seq": _seq2seq is not None},
        "checkpoint_loaded": _classifier is not None and _seq2seq is not None,
        "checkpoint_source": dict(_checkpoint_sources),
        "device": str(_device),
    }


@app.get("/models")
async def get_models() -> dict[str, Any]:
    """List the models this service exposes."""
    return {
        "available_models": [
            {
                "id": "vit",
                "name": "Vision Transformer",
                "type": "classification",
                "loaded": _classifier is not None,
            },
            {
                "id": "seq2seq",
                "name": "Seq2Seq Vision Transformer",
                "type": "sequence",
                "loaded": _seq2seq is not None,
            },
        ],
        "device": str(_device),
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: ImageRequest) -> PredictionResponse:
    """Predict with whichever model the request names.

    Args:
        request: The image and the model to use.

    Returns:
        That model's prediction.
    """
    if request.model_type == "seq2seq":
        seq2seq_arch = _config.seq2seq_arch
        image_tensor = preprocess_image(request.image_data, seq2seq_arch.img_size)
        prediction, confidence, digit_trace = predict_seq2seq(image_tensor)
        return PredictionResponse(
            pred=prediction,
            confidence=confidence,
            model_type="seq2seq",
            digit_trace=digit_trace,
            img_size=seq2seq_arch.img_size,
            patch_size=seq2seq_arch.patch_size,
        )

    classifier_arch = _config.classifier_arch
    image_tensor = preprocess_image(request.image_data, classifier_arch.img_size)
    prediction, confidence, class_probabilities = predict_vit(image_tensor)
    return PredictionResponse(
        pred=prediction,
        confidence=confidence,
        model_type="vit",
        class_probabilities=class_probabilities,
        img_size=classifier_arch.img_size,
        patch_size=classifier_arch.patch_size,
    )


@app.post("/predict/vit", response_model=PredictionResponse)
async def predict_vit_endpoint(request: ImageRequest) -> PredictionResponse:
    """Predict with the classifier specifically.

    Args:
        request: The image to classify.

    Returns:
        The classifier's prediction.
    """
    request.model_type = "vit"
    return await predict(request)


@app.post("/predict/seq2seq", response_model=PredictionResponse)
async def predict_seq2seq_endpoint(request: ImageRequest) -> PredictionResponse:
    """Predict with the seq2seq model specifically.

    Args:
        request: The image to read.

    Returns:
        The seq2seq model's prediction.
    """
    request.model_type = "seq2seq"
    return await predict(request)


@app.post("/predict/both", response_model=BothModelsResponse)
async def predict_both_models(request: ImageRequest) -> BothModelsResponse:
    """Run both models over the same image.

    The image is preprocessed twice, once per model, because the two take different
    input resolutions and a single shared tensor cannot feed both correctly.

    Args:
        request: The image to run.

    Returns:
        Both predictions and how long they took.
    """
    start_time = time.time()

    vit_response = await predict_vit_endpoint(request.model_copy())
    seq2seq_response = await predict_seq2seq_endpoint(request.model_copy())

    return BothModelsResponse(
        vit_prediction={
            **vit_response.model_dump(),
            "model_name": "Vision Transformer (Classification)",
        },
        seq2seq_prediction={
            **seq2seq_response.model_dump(),
            "model_name": "Seq2Seq Vision Transformer",
        },
        processing_time=time.time() - start_time,
    )


@app.post("/predict/seq2seq/stream")
def predict_seq2seq_stream(request: ImageRequest) -> StreamingResponse:
    """Same pipeline as `/predict/seq2seq`, streamed as SSE per decoded digit.

    Runs the real seq2seq autoregressive generation on a background thread and relays
    each digit through a queue the moment the model emits it, so a caller sees the
    number being read digit by digit instead of waiting for the full sequence.

    Args:
        request: The base64-encoded image.

    Returns:
        A `text/event-stream` response. Each digit is
        `{"stage": "digit", "step": N, "digit": "...", "confidence": ...}`; the final
        event is `{"stage": "done", ...same shape as /predict/seq2seq...}`, or
        `{"stage": "error", "detail": ...}` if the pipeline raised.
    """
    if _seq2seq is None:
        raise HTTPException(status_code=503, detail="Seq2Seq model not loaded")

    arch = _config.seq2seq_arch
    image_tensor = preprocess_image(request.image_data, arch.img_size)

    def _stream_worker(events: queue.Queue[dict[str, Any] | None]) -> None:
        try:
            generated = _seq2seq.generate_with_callback(
                image_tensor,
                start_token=arch.start_token,
                max_length=arch.max_seq_length,
                on_digit=lambda step, digit_str, prob: events.put(
                    {"stage": "digit", "step": step, "digit": digit_str, "confidence": prob}
                ),
            )

            digits: list[str] = []
            digit_positions: list[int] = []
            for position, token in enumerate(generated[0].tolist()):
                if token == arch.start_token:
                    continue
                if token == arch.eos_token:
                    break
                digits.append(str(token))
                digit_positions.append(position)

            prediction = "".join(digits) if digits else "0"

            if not digit_positions:
                events.put(
                    {
                        "stage": "done",
                        "pred": prediction,
                        "confidence": 0.0,
                        "model_type": "seq2seq",
                        "digit_trace": [],
                        "img_size": arch.img_size,
                        "patch_size": arch.patch_size,
                    }
                )
            else:
                logits = _seq2seq.decode(_seq2seq.encode(image_tensor), generated[:, :-1])
                probabilities = F.softmax(logits, dim=-1)
                token_probabilities = [
                    float(probabilities[0, position - 1, generated[0, position]].item())
                    for position in digit_positions
                ]
                confidence = sum(token_probabilities) / len(token_probabilities)
                digit_trace = [
                    {"digit": digit, "confidence": conf}
                    for digit, conf in zip(digits, token_probabilities, strict=True)
                ]
                events.put(
                    {
                        "stage": "done",
                        "pred": prediction,
                        "confidence": confidence,
                        "model_type": "seq2seq",
                        "digit_trace": digit_trace,
                        "img_size": arch.img_size,
                        "patch_size": arch.patch_size,
                    }
                )
        except Exception as error:
            events.put({"stage": "error", "detail": str(error)})

    return stream_sse(_stream_worker)


def main() -> None:
    """Entry point for `make serve`."""
    run_app(app, host=_config.api_host, port=_config.api_port)


if __name__ == "__main__":
    main()
