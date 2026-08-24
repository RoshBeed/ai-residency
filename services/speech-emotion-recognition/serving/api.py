"""FastAPI service predicting the emotion in a clip.

`serving/api.py` exposing `app` is the canonical backend location in every service in
this portfolio, which is what lets the frontend talk to all of them the same way.

Audio arrives base64-encoded in a pydantic field rather than as a multipart upload,
matching the convention every other service uses.

The checkpoint is resolved locally first, then from the Hugging Face Hub, so this runs
on a host that has never trained anything. When neither has it, the service still
starts and reports that plainly through `/health` rather than refusing to boot -- a
missing checkpoint is a fact to report, not a reason for the pod to crash-loop.
"""

import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import librosa
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ai_residency_common import (
    add_health_routes,
    decode_audio,
    get_logger,
    resolve_checkpoint,
    run_app,
)
from config import Config
from speech_emotion_recognition.models import load_model_and_processor

__all__ = ["app"]

logger = get_logger(__name__)

_config = Config()
_model: Any = None
_processor: Any = None
_device: torch.device | None = None
_checkpoint_source = "none"


class AudioRequest(BaseModel):
    """A single clip to score."""

    audio_data: str


class EmotionResponse(BaseModel):
    """The predicted emotion and the score behind it."""

    emotion: str
    confidence: float
    scores: dict[str, float]
    checkpoint_source: str


def load_model() -> None:
    """Load the model once, at startup, from disk or the Hub."""
    global _model, _processor, _device, _checkpoint_source

    checkpoint_path, _checkpoint_source = resolve_checkpoint(
        _config.checkpoint_dir / _config.checkpoint_name, _config.hf_repo_id
    )
    if checkpoint_path is None:
        logger.warning("No emotion checkpoint locally or on the Hub; /predict will answer 503")
        return

    _model, _processor, _device = load_model_and_processor(_config, checkpoint_path)
    logger.info("Serving on %s (checkpoint from %s)", _device, _checkpoint_source)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Load the checkpoint once, when the service starts."""
    load_model()
    yield


app = FastAPI(
    title="Speech Emotion Recognition",
    description="Whisper fine-tuned to emit an emotion token for a clip",
    lifespan=lifespan,
)


add_health_routes(
    app,
    ready=lambda: _model is not None,
    detail=lambda: {
        "model_loaded": _model is not None,
        "checkpoint_loaded": _model is not None,
        "checkpoint_source": _checkpoint_source,
        "emotions": list(_config.emotions),
        "device": str(_device) if _device is not None else "none",
    },
)


@app.post("/predict", response_model=EmotionResponse)
async def predict(request: AudioRequest) -> EmotionResponse:
    """Predict the emotion in a clip.

    Args:
        request: Base64-encoded audio.

    Returns:
        The predicted emotion, its probability, and the probability of every emotion.

    Raises:
        HTTPException: 503 if no checkpoint was ever loaded, 400 if the audio is
            undecodable.
    """
    if _model is None or _processor is None or _device is None:
        raise HTTPException(status_code=503, detail="No emotion checkpoint loaded")

    audio_bytes = decode_audio(request.audio_data)
    with tempfile.NamedTemporaryFile(suffix=".wav") as handle:
        handle.write(audio_bytes)
        handle.flush()
        try:
            waveform, _ = librosa.load(
                handle.name,
                sr=_config.sample_rate,
                duration=_config.max_duration_seconds,
            )
        except Exception as error:
            raise HTTPException(
                status_code=400, detail=f"Could not decode audio: {error}"
            ) from error

    features = _processor.feature_extractor(
        np.asarray(waveform), sampling_rate=_config.sample_rate, return_tensors="pt"
    ).input_features.to(_device)

    # Score the emotion tokens directly rather than generating text: the label is one
    # token, so the decoder's first step already holds the whole answer, and reading it
    # off the logits gives every emotion's probability instead of only the winner.
    token_ids = [
        _processor.tokenizer.convert_tokens_to_ids(_config.token_for(emotion))
        for emotion in _config.emotions
    ]
    with torch.no_grad():
        decoder_start = torch.tensor([[_model.config.decoder_start_token_id]], device=_device)
        logits = _model(input_features=features, decoder_input_ids=decoder_start).logits
        emotion_logits = logits[0, -1, token_ids]
        probabilities = torch.softmax(emotion_logits, dim=-1)

    scores = {
        emotion: float(probability)
        for emotion, probability in zip(_config.emotions, probabilities, strict=True)
    }
    best = max(scores, key=lambda emotion: scores[emotion])
    return EmotionResponse(
        emotion=best,
        confidence=scores[best],
        scores=scores,
        checkpoint_source=_checkpoint_source,
    )


def main() -> None:
    """Entry point for `make serve`."""
    run_app(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
