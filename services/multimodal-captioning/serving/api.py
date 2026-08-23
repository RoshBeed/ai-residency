"""FastAPI service.

The HTTP contract the portfolio hub talks to. Thin by design: it decodes the request's
base64 image and calls `captioning.generate_caption`, this repo's one inference path.

`/caption/stream` does the same generation as `/caption` but relays every token as a
Server-Sent Event the moment the model emits it (greedy decoding, autoregressive loop),
so the frontend can animate the caption being written rather than showing a spinner.

Binary input goes base64-encoded in a pydantic field rather than as a multipart upload,
matching the convention the other week repos already use.
"""

import queue
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ai_residency_common import decode_image, get_device, get_logger, run_app, stream_sse
from config import CHECKPOINTS, Config
from multimodal_captioning.captioning import (
    CaptionResult,
    caption_image_with_callback,
    generate_caption,
)
from multimodal_captioning.models import CHECKPOINT_SOURCES, load_adapter, load_backbones

__all__ = ["app"]

logger = get_logger(__name__)

app = FastAPI(title="Image Captioning API (CLIP + Qwen3)", version="1.0.0")

_config = Config()


class CaptionRequest(BaseModel):
    """Input for `/caption`."""

    image_data: str
    checkpoint_name: str = _config.checkpoint_name
    prompt: str = _config.prompt
    max_new_tokens: int = _config.max_new_tokens


class TokenTrace(BaseModel):
    """One generated token and how confident the model was in picking it."""

    token: str
    probability: float


class CaptionResponse(BaseModel):
    """Output of `/caption`."""

    caption: str
    checkpoint_used: str
    token_trace: list[TokenTrace] = []
    prompt_used: str = ""


@app.get("/health")
@app.get("/")
def health() -> dict[str, Any]:
    """Liveness check used by compose and the Helm chart.

    Registered on `/` as well, matching the other five backends. This one answered
    404 there, which made the hub's `make status` (which curls `/`) report it as
    down while it was serving captions perfectly well.

    Adapters load lazily on the first request, so `checkpoint_source` is empty until
    one has been captioned with. It then records whether those weights came off local
    disk or the Hub.
    """
    return {
        "status": "ok",
        "device": str(get_device()),
        "checkpoints": {name: info.description for name, info in CHECKPOINTS.items()},
        "checkpoint_loaded": bool(CHECKPOINT_SOURCES),
        "checkpoint_source": dict(CHECKPOINT_SOURCES),
    }


@app.post("/caption", response_model=CaptionResponse)
def caption(request: CaptionRequest) -> CaptionResponse:
    """Caption one base64-encoded image.

    Args:
        request: The image and the checkpoint to caption it with.

    Returns:
        The generated caption and the checkpoint that produced it.

    Raises:
        HTTPException: 422 if the payload is not a decodable image, or if the named
            checkpoint does not exist.
    """
    image = decode_image(request.image_data)

    try:
        generated = generate_caption(
            image,
            _config,
            checkpoint_name=request.checkpoint_name,
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
        )
    except FileNotFoundError as error:
        raise HTTPException(422, str(error)) from error

    return CaptionResponse(
        caption=generated.caption,
        checkpoint_used=request.checkpoint_name,
        token_trace=[
            TokenTrace(token=trace.token, probability=trace.probability)
            for trace in generated.token_trace
        ],
        prompt_used=request.prompt,
    )


class StreamToken(BaseModel):
    """One generated token, streamed as an SSE event."""

    step: int
    token: str
    probability: float


@app.post("/caption/stream")
def caption_stream(request: CaptionRequest) -> StreamingResponse:
    """Same pipeline as `/caption`, streamed as Server-Sent Events per token.

    Runs the real inference (CLIP encode, adapter project, Qwen3 autoregressive
    generation) on a background thread and relays each token through a queue the
    moment the model emits it, so a caller sees the caption being written live token
    by token instead of waiting for the full generation.

    Args:
        request: The image, checkpoint, prompt and generation cap.

    Returns:
        A `text/event-stream` response. Each `token` event is
        `{"stage": "token", "step": N, "token": ..., "probability": ...}`; the final
        event is `{"stage": "done", ...same shape as /caption's body...}`, or
        `{"stage": "error", "detail": ...}` if the pipeline raised.
    """
    image = decode_image(request.image_data)

    def _stream_worker(events: queue.Queue[dict[str, Any] | None]) -> None:
        try:
            backbones = load_backbones(_config)
            adapter = load_adapter(_config, request.checkpoint_name, backbones)
            events.put({"stage": "model_loaded"})

            def on_token(step: int, token: str, probability: float) -> None:
                events.put(
                    {"stage": "token", "step": step, "token": token, "probability": probability}
                )

            result: CaptionResult = caption_image_with_callback(
                image,
                adapter,
                backbones,
                _config,
                prompt=request.prompt,
                max_new_tokens=request.max_new_tokens,
                on_token=on_token,
            )

            events.put(
                {
                    "stage": "done",
                    "caption": result.caption,
                    "checkpoint_used": request.checkpoint_name,
                    "token_trace": [
                        {"token": t.token, "probability": t.probability} for t in result.token_trace
                    ],
                    "prompt_used": request.prompt,
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
