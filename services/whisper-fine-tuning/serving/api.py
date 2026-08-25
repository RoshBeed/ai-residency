"""FastAPI backend for the base-vs-fine-tuned Whisper transcription comparison.

A thin HTTP wrapper over the same functions the CLI demo and the Streamlit UI call —
`load_whisper_model`, `get_tokenizer`, `decode_audio` and `fine_tune`. It does not
reimplement Whisper loading or inference.

Binary audio arrives base64-encoded inside a JSON body rather than as a multipart
upload, which is the convention every backend in this portfolio uses. It is written to
a temp file because the audio readers take a path rather than raw bytes.

Each request loads a fresh copy of Whisper-tiny and fine-tunes it in place on the
audio it was sent, exactly as the CLI demo does. There is no persisted fine-tuned
checkpoint to serve instead: `make train` writes one, but it is fitted to the single
bundled clip, so loading it for arbitrary uploaded audio would be misleading rather
than useful. `TARGET_TEXT` is only the request's *default* target transcript now — a
caller can override it via `target_text`, since fine-tuning always toward one fixed
demo sentence regardless of what the audio actually says was misleading for anything
that wasn't that exact clip.
"""

import os
import queue
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ai_residency_common import add_health_routes, get_logger, retrying, run_app, stream_sse
from ai_residency_common import decode_audio as decode_base64_audio
from config import Config
from whisper_fine_tuning.models import (
    decode_audio,
    load_whisper_model,
    prepare_training_data,
)
from whisper_fine_tuning.training import evaluate, fine_tune

__all__ = ["app"]

logger = get_logger(__name__)

config = Config()

# The same ground-truth text the CLI demo fine-tunes toward.
TARGET_TEXT = config.target_text

_weights_ready = False
"""Whether Whisper-tiny's weights resolved at startup. Reported by `/ready`."""


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Resolve Whisper-tiny once at boot, so readiness means something here.

    This service has no checkpoint of its own to load: `/transcribe` deliberately
    builds a *fresh* copy per request, because fine-tuning mutates it in place. That
    left nothing for a readiness probe to check. Loading the weights once at startup
    both proves they are fetchable -- on a cold pod they come over the network -- and
    populates the on-disk cache every later request reads from, so the first caller
    no longer pays the download.
    """
    ensure_loaded()
    yield


def ensure_loaded() -> bool:
    """Resolve the weights if they are not resolved, and report whether they are."""
    global _weights_ready
    if _weights_ready:
        return True
    try:
        load_whisper_model(config)
        _weights_ready = True
        logger.info("Whisper weights resolved; ready to transcribe")
    except Exception as error:
        logger.error("Could not resolve the Whisper weights, /ready will report 503: %s", error)
        return False
    return _weights_ready


app = FastAPI(
    title="Whisper Fine-Tuning API",
    description="Base-vs-fine-tuned Whisper transcription comparison.",
    version="1.0.0",
    lifespan=lifespan,
)


class TranscribeRequest(BaseModel):
    """Input for `/transcribe`."""

    audio_data: str
    """Base64-encoded audio bytes: mp3, wav, or anything else ffmpeg can decode."""

    target_text: str = TARGET_TEXT
    """Ground truth to fine-tune toward. Defaults to the CLI demo's own sentence."""


class FineTuneStep(BaseModel):
    """One gradient step of the live fine-tuning loop."""

    step: int
    loss: float
    text: str
    """What the model transcribes the clip as, right after this step's update."""


class TranscribeResponse(BaseModel):
    """Output of `/transcribe`."""

    base_output: str
    ft_output: str
    baseline_loss: float = 0.0
    fine_tune_trace: list[FineTuneStep] = []


add_health_routes(
    app,
    # Retried rather than decided once. A resolve that failed at startup -- a DNS blip
    # during a mass restart, say -- would otherwise leave this pod not-ready for the
    # rest of its life, with liveness unable to help because the process itself is
    # perfectly healthy. `retrying` re-attempts, at most once a minute, from /ready.
    ready=retrying(ensure_loaded),
    detail=lambda: {
        "device": str(config.device),
        "checkpoint_loaded": _weights_ready,
    },
)


@app.post("/transcribe", response_model=TranscribeResponse)
def transcribe(request: TranscribeRequest) -> TranscribeResponse:
    """Transcribe audio with the base model and with a copy fine-tuned on it.

    Args:
        request: Base64-encoded audio.

    Returns:
        The base transcript and the fine-tuned transcript.

    Raises:
        HTTPException: If `audio_data` is not valid, non-empty base64.
    """
    audio_bytes = decode_base64_audio(request.audio_data)

    if not audio_bytes:
        raise HTTPException(status_code=400, detail="audio_data decoded to empty bytes")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        whisper = load_whisper_model(config)

        try:
            base_output = decode_audio(whisper, tmp_path)
        except RuntimeError as error:
            raise HTTPException(
                status_code=422, detail=f"Failed to transcribe audio: {error}"
            ) from error

        features, labels = prepare_training_data(whisper, tmp_path, request.target_text)
        baseline_loss = evaluate(whisper, features, labels)

        steps: list[FineTuneStep] = []

        def on_step(step: int, loss: float) -> None:
            """Decode the clip with the just-updated weights, between gradient steps.

            `speech.transcribe` runs under `no_grad` and restores whatever train/eval
            mode it found, so this no longer has to bracket the call by hand -- getting
            that wrong would leave dropout active during decoding, or leave the model
            in eval for the rest of the loop.
            """
            steps.append(FineTuneStep(step=step, loss=loss, text=decode_audio(whisper, tmp_path)))

        whisper = fine_tune(
            whisper,
            tmp_path,
            request.target_text,
            learning_rate=config.learning_rate,
            num_steps=5,
            on_step=on_step,
        )
        ft_output = decode_audio(whisper, tmp_path)

        return TranscribeResponse(
            base_output=base_output,
            ft_output=ft_output,
            baseline_loss=baseline_loss,
            fine_tune_trace=steps,
        )
    finally:
        if tmp_path is not None and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/transcribe/stream")
def transcribe_stream(request: TranscribeRequest) -> StreamingResponse:
    """Same pipeline as `/transcribe`, streamed as Server-Sent Events per stage.

    Runs the real work (audio decode, baseline eval, five fine-tuning steps) on a
    background thread and relays each stage through a queue as it completes, so a
    caller sees live progress instead of waiting on one blocking response -- the
    fine-tuning loop already runs step by step, this just stops hiding that.

    Args:
        request: Base64-encoded audio and the target transcript.

    Returns:
        A `text/event-stream` response. Each event is `{"stage": ..., ...}`; the
        final event is `{"stage": "done", ...same shape as /transcribe's body...}`,
        or `{"stage": "error", "detail": ...}` if the pipeline raised.
    """
    audio_bytes = decode_base64_audio(request.audio_data)
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="audio_data decoded to empty bytes")

    def _stream_worker(events: queue.Queue[dict[str, Any] | None]) -> None:
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            whisper = load_whisper_model(config)

            base_output = decode_audio(whisper, tmp_path)
            events.put({"stage": "base_transcript", "text": base_output})

            features, labels = prepare_training_data(whisper, tmp_path, request.target_text)
            baseline_loss = evaluate(whisper, features, labels)
            events.put({"stage": "baseline_loss", "loss": baseline_loss})

            steps: list[dict[str, Any]] = []

            def on_step(step: int, loss: float) -> None:
                # transcribe restores the train/eval mode it found, so the loop is
                # handed back exactly the state it was running in.
                step_event = {"step": step, "loss": loss, "text": decode_audio(whisper, tmp_path)}
                steps.append(step_event)
                events.put({"stage": "step", **step_event})

            fine_tune(
                whisper,
                tmp_path,
                request.target_text,
                learning_rate=config.learning_rate,
                num_steps=5,
                on_step=on_step,
            )
            ft_output = decode_audio(whisper, tmp_path)

            events.put(
                {
                    "stage": "done",
                    "base_output": base_output,
                    "ft_output": ft_output,
                    "baseline_loss": baseline_loss,
                    "fine_tune_trace": steps,
                }
            )
        except Exception as error:
            events.put({"stage": "error", "detail": str(error)})
        finally:
            if tmp_path is not None and os.path.exists(tmp_path):
                os.remove(tmp_path)

    return stream_sse(_stream_worker)


def main() -> None:
    """Entry point for `make serve`."""
    run_app(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
