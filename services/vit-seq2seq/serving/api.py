"""FastAPI service for the multi-digit encoder-decoder Vision Transformer.

Canonical location for the HTTP backend in every service in this portfolio:
`serving/api.py`, exposing `app`.

`/predict/stream` does the same digit-by-digit generation as `/predict` but relays
each decoded digit as a Server-Sent Event the moment the model emits it, so the
frontend can show the number being read digit by digit.

Images arrive base64-encoded in a pydantic field rather than as multipart uploads,
matching the convention every other service uses.
"""

import queue
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ai_residency_common import (
    add_health_routes,
    decode_image,
    get_device,
    get_logger,
    load_checkpoint,
    resolve_checkpoint,
    retrying,
    run_app,
    stream_sse,
)
from config import Config
from vit_seq2seq.seq2seq import Seq2SeqVisionTransformer

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


_seq2seq: Seq2SeqVisionTransformer | None = None


_checkpoint_source = "none"


class ImageRequest(BaseModel):
    """A single image to classify."""

    image_data: str
    # Ignored: this service has one model. Kept so payloads written against the
    # combined service still validate.
    model_type: str = "seq2seq"


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


def load_models() -> None:
    """Load the seq2seq checkpoint, pulling it from the Hub if it is not on disk.

    The fallback is what lets this service run on a host that never trained here: the
    checkpoint is gitignored, so a fresh clone or a separate inference box would
    otherwise start with no model and answer 503 to everything.
    """
    global _seq2seq, _checkpoint_source

    logger.info("Serving on %s", _device)

    seq2seq_path, _checkpoint_source = resolve_checkpoint(
        _config.seq2seq_checkpoint_path, _config.hf_repo_id
    )
    if seq2seq_path is not None:
        seq2seq = Seq2SeqVisionTransformer(_config.seq2seq_arch)
        load_checkpoint(seq2seq_path, seq2seq)
        _seq2seq = seq2seq.to(_device).eval()
    else:
        logger.warning("No seq2seq checkpoint locally or on the Hub")


def ensure_loaded() -> bool:
    """Load the checkpoint if it is not already loaded, and report whether it is.

    Idempotent, and safe to call from a readiness probe: it returns immediately once
    the model is in memory, and re-attempts the resolve otherwise.
    """
    if _seq2seq is not None:
        return True
    try:
        load_models()
    except (FileNotFoundError, KeyError, RuntimeError) as error:
        logger.error("Could not load the checkpoint, /ready stays 503: %s", error)
        return False
    return _seq2seq is not None


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
        HTTPException: 422 if the payload is not a decodable image.
    """
    # `decode_image` rather than a local `Image.open`. The local version passed a
    # truncated PNG straight through, because `Image.open` only reads the header --
    # the pixel data is decoded lazily, so the failure surfaced later, outside the
    # try block, as a 500. The shared helper calls `.load()` inside its own guard,
    # which is the whole reason it exists.
    image = decode_image(image_data)

    if image.mode != "L":
        image = image.convert("L")
    image = image.resize((img_size, img_size))

    array = np.array(image, dtype=np.float32) / 255.0
    array = (array - MNIST_MEAN) / MNIST_STD
    return torch.from_numpy(array).unsqueeze(0).unsqueeze(0).to(_device)


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


add_health_routes(
    app,
    # Retried rather than decided once. A checkpoint that could not be resolved at
    # startup -- a DNS blip during a mass restart, say -- would otherwise leave this
    # pod not-ready for the rest of its life, with liveness unable to help because the
    # process itself is perfectly healthy. `retrying` re-attempts, at most once a
    # minute, from /ready.
    ready=retrying(ensure_loaded),
    detail=lambda: {
        "message": "Vision Transformer seq2seq API",
        "model_loaded": _seq2seq is not None,
        "checkpoint_loaded": _seq2seq is not None,
        "checkpoint_source": _checkpoint_source,
        "device": str(_device),
    },
)


@app.get("/models")
async def get_models() -> dict[str, Any]:
    """Describe the model this service exposes."""
    arch = _config.seq2seq_arch
    return {
        "id": "seq2seq",
        "name": "Seq2Seq Vision Transformer",
        "type": "sequence",
        "loaded": _seq2seq is not None,
        "img_size": arch.img_size,
        "patch_size": arch.patch_size,
        "device": str(_device),
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: ImageRequest) -> PredictionResponse:
    """Read the multi-digit number off an image.

    Args:
        request: The base64-encoded image.

    Returns:
        The predicted number, the mean per-digit confidence, and the per-digit trace.
    """
    arch = _config.seq2seq_arch
    image_tensor = preprocess_image(request.image_data, arch.img_size)
    prediction, confidence, digit_trace = predict_seq2seq(image_tensor)
    return PredictionResponse(
        pred=prediction,
        confidence=confidence,
        model_type="seq2seq",
        digit_trace=digit_trace,
        img_size=arch.img_size,
        patch_size=arch.patch_size,
    )


@app.post("/predict/seq2seq", response_model=PredictionResponse)
async def predict_seq2seq_endpoint(request: ImageRequest) -> PredictionResponse:
    """Alias of `/predict`, kept because callers were written against the route name
    this service used when the classifier shared the same process.

    Args:
        request: The base64-encoded image.

    Returns:
        The seq2seq model's prediction.
    """
    return await predict(request)


@app.post("/predict/stream")
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
