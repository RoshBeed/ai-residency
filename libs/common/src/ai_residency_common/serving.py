"""Serving-time helpers shared across the portfolio's backends.

Three things the six repos used to each write themselves:

* resolving a trained artifact and failing loudly when it is nowhere (the
  ``resolve_checkpoint`` + ``if path is None: raise FileNotFoundError`` guard
  that appeared in a dozen places, each with its own phrasing),
* loading a torch model from a resolved checkpoint and putting it in eval mode,
* the uvicorn bootstrap every ``make serve`` entry point ends in.

Plus the base64 binary-input decoding every backend that takes an image or
audio clip repeats: decode, wrap the failure as an HTTP error.
"""

import base64
import io
from pathlib import Path
from typing import Any

import torch
from fastapi import HTTPException
from PIL import Image
from torch import nn

from ai_residency_common.checkpoints import load_checkpoint
from ai_residency_common.hub import RepoType, resolve_checkpoint
from ai_residency_common.logging import get_logger

__all__ = ["decode_audio", "decode_image", "load_model", "resolve_required", "run_app"]

logger = get_logger(__name__)


def decode_image(image_data: str) -> Image.Image:
    """Decode a base64-encoded image into a PIL image, raising 422 on failure.

    Args:
        image_data: Base64-encoded image bytes.

    Returns:
        The decoded image.

    Raises:
        HTTPException: 422 if the payload is not a decodable image.
    """
    try:
        return Image.open(io.BytesIO(base64.b64decode(image_data)))
    except Exception as error:
        raise HTTPException(422, f"Could not decode image_data as an image: {error}") from error


def decode_audio(audio_data: str) -> bytes:
    """Decode base64-encoded audio into raw bytes, raising 400 on failure.

    Args:
        audio_data: Base64-encoded audio bytes.

    Returns:
        The decoded bytes.

    Raises:
        HTTPException: 400 if the payload is not valid base64.
    """
    try:
        return base64.b64decode(audio_data, validate=True)
    except ValueError as error:
        raise HTTPException(400, f"Invalid base64 audio_data: {error}") from error


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
            the artifact. Ends up as " ... Build it with `make train-cbow`."

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
