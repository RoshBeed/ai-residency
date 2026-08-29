"""Serving-time helpers shared across the portfolio's backends.

Three things the six repos used to each write themselves:

* resolving a trained artifact and failing loudly when it is nowhere (the
  ``resolve_checkpoint`` + ``if path is None: raise FileNotFoundError`` guard
  that appeared in a dozen places, each with its own phrasing),
* loading a torch model from a resolved checkpoint and putting it in eval mode,
* the uvicorn bootstrap every ``make serve`` entry point ends in.

Plus the base64 binary-input decoding every backend that takes an image or
audio clip repeats: decode, wrap the failure as an HTTP error.

And ``add_health_routes``, which owns the liveness/readiness contract for every
backend. Splitting those two is not cosmetic. Every service used to expose one
``/health`` that answered 200 unconditionally and reported the checkpoint state
in the body, which meant a Kubernetes ``readinessProbe`` pointed at it could
never fail: a pod with no weights was marked Ready, joined the Service
endpoints, and 503'd every request it was sent -- the exact failure the probe
exists to prevent. A liveness probe on the same endpoint had the mirror problem,
since a wedged model still answered 200 and so could never be restarted.
"""

import base64
import io
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from torch import nn

from ai_residency_common.checkpoints import load_checkpoint
from ai_residency_common.hub import RepoType, resolve_checkpoint
from ai_residency_common.logging import get_logger

__all__ = [
    "add_health_routes",
    "decode_audio",
    "decode_image",
    "load_model",
    "resolve_required",
    "retrying",
    "run_app",
]

logger = get_logger(__name__)


def retrying(
    load: Callable[[], bool],
    *,
    min_interval_seconds: float = 60.0,
) -> Callable[[], bool]:
    """Wrap a loader so readiness retries it, rather than giving up at startup.

    Resolving a checkpoint happens once, in the app's lifespan, and a transient
    failure there is permanent: the pod answers 503 for the rest of its life even
    after the problem clears. That is not hypothetical -- eleven pods restarting at
    once saturated DNS on a single-node cluster, several failed to resolve
    huggingface.co, and they stayed not-ready long after resolution recovered.

    Kubernetes will not rescue it either, by design: liveness points at `/health`,
    which answers 200 because the process is fine. Nothing is wrong with the process.
    What is missing is a retry.

    So `/ready` performs one. Rate-limited, because a readiness probe runs every ten
    seconds and re-attempting a multi-gigabyte download that often would be worse than
    the problem.

    A service with genuinely nothing to load -- no trained checkpoint yet -- simply
    keeps returning False, which is the honest answer and costs one cheap call per
    interval.

    Args:
        load: Attempts to load, returning whether the service can now serve. Must be
            safe to call repeatedly, and must not raise.
        min_interval_seconds: Shortest gap between attempts.

    Returns:
        A `ready` callable for `add_health_routes`.
    """
    # `None`, not 0.0. `time.monotonic()` counts from an arbitrary origin -- on a
    # freshly booted machine it is a small number, so `now - 0.0` can be less than the
    # interval and the *first* attempt gets rate-limited away before it ever happens.
    # That is invisible on a host with days of uptime and silently costs an hour on a
    # new node, which is exactly when a service is starting for the first time.
    ready_now = False
    last_attempt: float | None = None

    def check() -> bool:
        nonlocal ready_now, last_attempt
        if ready_now:
            return True
        now = time.monotonic()
        if last_attempt is not None and now - last_attempt < min_interval_seconds:
            return False
        last_attempt = now
        try:
            ready_now = bool(load())
        except Exception as error:
            logger.warning("Readiness retry failed, will try again: %s", error)
            ready_now = False
        return ready_now

    return check


def add_health_routes(
    app: FastAPI,
    *,
    ready: Callable[[], bool],
    detail: Callable[[], dict[str, Any]] | None = None,
    include_root: bool = True,
) -> None:
    """Register this portfolio's `/health` and `/ready` contract on `app`.

    `/health` is liveness: the process is up and serving HTTP. It answers 200
    unconditionally, so a slow or absent checkpoint never gets a pod restarted.

    `/ready` is readiness: this instance can answer a real request. It answers
    200 when `ready()` is true and **503 when it is not**, which is what lets a
    Service take an unusable pod out of its endpoints instead of routing to it.

    Point `livenessProbe` and `startupProbe` at `/health`, and `readinessProbe`
    at `/ready`. Both bodies carry `detail()`, so whatever a service wants to
    disclose -- device, checkpoint source, loaded adapters -- shows up in both.

    Args:
        app: The FastAPI application to register the routes on.
        ready: Returns whether this instance can serve a real request now.
            Called on every probe, so it must be cheap: read cached state, do
            not touch the network or the disk.
        detail: Extra fields to merge into both bodies. Also called per probe,
            so the same constraint applies.
        include_root: Also serve the liveness body at `/`. On by default because
            `make status` curls `/`, and a backend that 404s there was reported
            as down while it was serving perfectly well.
    """

    def _body() -> dict[str, Any]:
        return dict(detail()) if detail is not None else {}

    @app.get("/health")
    def health() -> dict[str, Any]:
        """Liveness: the process is up. Always 200."""
        return {"status": "ok", **_body()}

    if include_root:
        app.get("/")(health)

    @app.get("/ready")
    def readiness() -> JSONResponse:
        """Readiness: this instance can answer a real request. 503 when it cannot."""
        is_ready = ready()
        return JSONResponse(
            status_code=200 if is_ready else 503,
            content={
                "status": "ready" if is_ready else "not_ready",
                "ready": is_ready,
                **_body(),
            },
        )


def decode_image(image_data: str) -> Image.Image:
    """Decode a base64-encoded image into a PIL image, raising 422 on failure.

    `Image.open` is lazy: it reads the header and returns, so a truncated or
    corrupt payload used to pass this guard cleanly and then blow up deep inside
    the model call, turning a client error into a 500. `load()` forces the pixel
    data to be read here, where the failure can still be reported as a 422.

    Args:
        image_data: Base64-encoded image bytes.

    Returns:
        The decoded image.

    Raises:
        HTTPException: 422 if the payload is not a decodable image.
    """
    try:
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        image.load()
    except Exception as error:
        raise HTTPException(422, f"Could not decode image_data as an image: {error}") from error
    return image


def decode_audio(audio_data: str) -> bytes:
    """Decode base64-encoded audio into raw bytes, raising 422 on failure.

    422 rather than 400, matching `decode_image`: the request is well-formed
    JSON, it is the *content* of a field that cannot be processed, which is what
    422 means. The two used to disagree, so the same class of failure produced a
    different status depending on which backend the caller reached.

    Args:
        audio_data: Base64-encoded audio bytes.

    Returns:
        The decoded bytes.

    Raises:
        HTTPException: 422 if the payload is not valid base64.
    """
    try:
        return base64.b64decode(audio_data, validate=True)
    except ValueError as error:
        raise HTTPException(422, f"Invalid base64 audio_data: {error}") from error


def resolve_required(
    artifact_path: str | Path,
    repo_id: str | None,
    *,
    repo_type: RepoType = "model",
    rebuild_hint: str = "",
) -> tuple[Path, str]:
    """Resolve an artifact locally or on the Hub, raising if it is nowhere.

    `resolve_checkpoint` degrades to `(None, "none")` on purpose, so a service
    can answer with an untrained model rather than crash. The loaders here need
    the opposite behaviour: a missing vocabulary or tower is a hard failure.

    Args:
        artifact_path: Where the artifact lives locally, and where a downloaded
            copy is placed.
        repo_id: Repo to fall back to. None means local-only.
        repo_type: `"model"` or `"dataset"`.
        rebuild_hint: Appended to the error, e.g. which `make` target recreates
            the artifact. Ends up as " ... Build it with `make train`."

    Returns:
        `(path, source)` where `source` is `"local"` or `"hub"`.

    Raises:
        FileNotFoundError: If the artifact is neither on disk nor on the Hub.
    """
    path, source = resolve_checkpoint(artifact_path, repo_id, repo_type=repo_type)
    if path is None:
        hint = f" {rebuild_hint}" if rebuild_hint else ""
        raise FileNotFoundError(f"{artifact_path} not found locally or in {repo_id}.{hint}")
    return path, source


def load_model(
    checkpoint_path: str | Path,
    repo_id: str | None,
    model: nn.Module,
    *,
    device: torch.device | None = None,
) -> tuple[nn.Module, str]:
    """Resolve a checkpoint and load its weights into `model`, ready for inference.

    Args:
        checkpoint_path: Where the checkpoint lives locally.
        repo_id: Repo to pull from when the local file is absent.
        model: Model to populate in place. Its `state_dict` keys must match the
            checkpoint's.
        device: Optional device to move the model onto before `eval()`.

    Returns:
        `(model, source)` where `source` is `"local"` or `"hub"`.

    Raises:
        FileNotFoundError: If the checkpoint is neither on disk nor on the Hub.
    """
    path, source = resolve_required(checkpoint_path, repo_id)
    load_checkpoint(path, model)
    if device is not None:
        model = model.to(device)
    model.eval()
    return model, source


def run_app(app: Any, *, host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run an ASGI app under uvicorn — the standard `make serve` entry point."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)
