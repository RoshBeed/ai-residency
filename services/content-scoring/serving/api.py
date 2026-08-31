"""FastAPI service predicting a Hacker News score from a post.

A thin HTTP wrapper over the same functions `predict.py` uses. All four architectures
are loaded at startup -- they are small heads sharing one embedding matrix, so holding
four costs a few megabytes over holding one -- which is what lets `/compare` answer the
question this service exists to ask: does fusing the title with the metadata beat either
on its own?

`/predict` scores with the configured architecture. `/compare` runs all four.
`/predict/stream` relays the same pipeline as Server-Sent Events, a stage at a time.

Requests carry the title and, optionally, the URL, the timestamp and the author. A
title-only request still works and is scored as a text post submitted now, so the older
client contract did not break when the features arrived.
"""

from collections.abc import AsyncIterator, Generator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ai_residency_common import add_health_routes, format_sse, get_logger, retrying, run_app
from config import Config, Fusion
from content_scoring.model import describe, graph_dot
from content_scoring.predict import (
    LoadedPredictor,
    Post,
    load_embeddings,
    load_predictor,
    score_post,
)

__all__ = ["app"]

logger = get_logger(__name__)

FUSIONS: tuple[Fusion, ...] = (
    "text_only",
    "features_only",
    "early",
    "block_diagonal",
    "late",
)
MAX_TITLE_LENGTH = 512
MAX_URL_LENGTH = 2048
MAX_AUTHOR_LENGTH = 64

_predictors: dict[str, LoadedPredictor] = {}
_config = Config()


def ensure_loaded() -> bool:
    """Load every architecture that has a checkpoint, and report whether any did.

    Returns:
        True once at least the configured architecture is loaded.

    Idempotent and safe to call from a readiness probe. A missing checkpoint for one
    architecture is logged and skipped rather than failing the service: three of four
    available is a service that can still answer, and `/compare` says which ones it
    used rather than silently returning fewer.
    """
    if _config.fusion in _predictors:
        return True
    try:
        # Loaded once and shared by all four. The matrix is 104MB and frozen, so
        # four independent loads would be 415MB of a pod to hold one table.
        embeddings = load_embeddings(_config)
    except (FileNotFoundError, KeyError, RuntimeError) as error:
        logger.warning("No extended embeddings: %s", error)
        return False
    for fusion in FUSIONS:
        if fusion in _predictors:
            continue
        try:
            _predictors[fusion] = load_predictor(Config(fusion=fusion), embeddings)
        except (FileNotFoundError, KeyError, RuntimeError) as error:
            logger.warning("No %s predictor: %s", fusion, error)
    return _config.fusion in _predictors


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Load models on startup. /ready retries if that fails."""
    ensure_loaded()
    yield


app = FastAPI(
    title="Hacker News Score Predictor",
    description="Predict a post's score from its title and metadata, four ways",
    version="2.0.0",
    lifespan=lifespan,
)


class PredictionRequest(BaseModel):
    """One post to score.

    Only `title` is required. Every other field narrows the estimate: without a URL the
    post is treated as a text post, and without a timestamp as submitted now.

    Every field is bounded. An unbounded string field on a request that reaches a model
    is an unbounded request, and the bound belongs here so validation rejects it before
    any work starts.
    """

    title: str = Field(min_length=1, max_length=MAX_TITLE_LENGTH)
    url: str = Field(default="", max_length=MAX_URL_LENGTH)
    posted_at: datetime | None = None
    author: str = Field(default="", max_length=MAX_AUTHOR_LENGTH)
    fusion: Fusion | None = Field(
        default=None, description="Which architecture to score with. Defaults to the served one."
    )


class WordTrace(BaseModel):
    """One title token and whether the vocabulary contains it."""

    word: str
    known: bool


class PredictionResponse(BaseModel):
    """What `/predict` returns.

    `predicted_score` is an expected upvote count -- the model emits `log(lambda)` under
    a Poisson objective, so the exponential is the expectation directly.

    `viral_probability` is the more honest half of the answer. Hacker News scores are
    close to a lottery: the same URL posted twice correlates at only rho 0.26, and of
    posts whose first submission scored 2-5 the second reached a maximum of 2,645. A
    single expected value hides all of that, so the response carries the model's estimate
    of clearing 100 upvotes beside it.
    """

    title: str
    predicted_score: float
    viral_probability: float = 0.0
    fusion: str
    api_source: str = "FastAPI Service"
    model_info: str = "extended skip-gram + fusion head, Poisson objective"
    word_trace: list[WordTrace] = []
    domain: str = ""
    embedding_dim: int = 0
    checkpoint_source: str = ""


class ComparisonResponse(BaseModel):
    """What `/compare` returns: every loaded architecture's answer to one post."""

    title: str
    domain: str
    predictions: dict[str, float]
    # Read off each loaded model, so the UI can say what it is comparing without
    # repeating shapes that would go stale the moment one changed.
    details: dict[str, dict[str, Any]] = {}
    word_trace: list[WordTrace] = []
    missing: list[str] = []


def _post(request: PredictionRequest) -> Post:
    """Turn a request into the dataclass `predict.py` reads."""
    timestamp = (
        pd.Timestamp(request.posted_at).tz_convert("UTC")
        if request.posted_at is not None and request.posted_at.tzinfo is not None
        else (
            pd.Timestamp(request.posted_at).tz_localize("UTC")
            if request.posted_at is not None
            else None
        )
    )
    return Post(title=request.title, url=request.url, posted_at=timestamp, author=request.author)


def _resolve(fusion: Fusion | None) -> LoadedPredictor:
    """Pick a loaded predictor, or say why there is not one.

    Args:
        fusion: The requested architecture, or None for the configured default.

    Returns:
        The loaded predictor.

    Raises:
        HTTPException: 503 when no checkpoint for it could be resolved. Not 500 -- the
            service is fine, it simply has nothing to serve yet, and 503 is what keeps
            an unusable pod out of its Service's endpoints.
    """
    name = fusion or _config.fusion
    if not ensure_loaded() or name not in _predictors:
        raise HTTPException(status_code=503, detail=f"No checkpoint loaded for fusion={name}")
    return _predictors[name]


def _trace(post: Post, loaded: LoadedPredictor, known: list[bool]) -> list[WordTrace]:
    """Pair a title's tokens with whether each was in the vocabulary."""
    from content_scoring.etl import normalise_text, tokenise

    tokens = tokenise(normalise_text(post.title))
    return [WordTrace(word=word, known=flag) for word, flag in zip(tokens, known, strict=True)]


add_health_routes(
    app,
    # Retried rather than decided once. A resolve that failed at startup -- a DNS blip
    # during a mass restart, say -- would otherwise leave this pod not-ready for the
    # rest of its life, with nothing wrong that a retry would not fix.
    ready=retrying(ensure_loaded),
    detail=lambda: {
        "device": str(_config.device),
        "checkpoint_loaded": _config.fusion in _predictors,
        "loaded_architectures": sorted(_predictors),
        "served_architecture": _config.fusion,
    },
)


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Score one post with a single architecture."""
    loaded = _resolve(request.fusion)
    post = _post(request)
    score, known = score_post(loaded, post, Config(fusion=loaded.fusion))
    return PredictionResponse(
        title=request.title,
        predicted_score=round(score, 2),
        viral_probability=round(viral_probability(loaded, score), 4),
        fusion=loaded.fusion,
        word_trace=_trace(post, loaded, known),
        domain=_domain(request.url),
        embedding_dim=loaded.embeddings.embed_size,
        checkpoint_source=loaded.source,
    )


@app.post("/compare", response_model=ComparisonResponse)
def compare(request: PredictionRequest) -> ComparisonResponse:
    """Score one post with every architecture that has a checkpoint.

    This is the endpoint the write-up is about. The four numbers it returns are the
    same comparison the README tables, made on a post the caller chose rather than on
    the test split.
    """
    if not ensure_loaded():
        raise HTTPException(status_code=503, detail="No checkpoints loaded")
    post = _post(request)
    predictions: dict[str, float] = {}
    known: list[bool] = []
    for fusion in FUSIONS:
        loaded = _predictors.get(fusion)
        if loaded is None:
            continue
        score, known = score_post(loaded, post, Config(fusion=fusion))
        predictions[fusion] = round(score, 2)
    reference = _predictors[next(iter(predictions))]
    return ComparisonResponse(
        title=request.title,
        domain=_domain(request.url),
        predictions=predictions,
        details={
            name: {
                **describe(_predictors[name].model),
                "graph": graph_dot(_predictors[name].model),
            }
            for name in predictions
        },
        word_trace=_trace(post, reference, known),
        missing=[fusion for fusion in FUSIONS if fusion not in predictions],
    )


def viral_probability(loaded: LoadedPredictor, expected: float) -> float:
    """P(score >= 100) for a post the model expects to score `expected`.

    Args:
        loaded: The predictor, carrying the calibration fitted during its training run.
        expected: The predicted upvote count.

    Returns:
        The empirical probability of clearing 100 upvotes, interpolated between the
        validation buckets, or 0.0 for a checkpoint predating the calibration.

    Read off held-out data rather than from the Poisson the model was trained under. The
    analytic tail of a Poisson with a mean of 15 is about 1e-25, so serving it reported
    0.0% for every post regardless of the title -- Hacker News is overdispersed enough
    that the parametric tail is meaningless.
    """
    curve = loaded.viral_calibration
    if not curve or not curve.get("edges"):
        return 0.0
    return float(np.interp(expected, curve["edges"], curve["rates"]))


def _domain(url: str) -> str:
    """The domain a request's URL points at, for display."""
    from content_scoring.etl import registered_domain

    return registered_domain(url)


@app.post("/predict/stream")
def predict_stream(request: PredictionRequest) -> StreamingResponse:
    """The same prediction, relayed a stage at a time.

    Stages are `word_lookup`, `features`, `fusion` and `done`, plus `error`. Each is
    what the pipeline actually just finished, so the UI's progress display is a report
    rather than an animation.
    """

    def events() -> Generator[str, None, None]:
        try:
            loaded = _resolve(request.fusion)
            post = _post(request)
            score, known = score_post(loaded, post, Config(fusion=loaded.fusion))
            trace = _trace(post, loaded, known)

            yield format_sse(
                {"stage": "word_lookup", "word_trace": [word.model_dump() for word in trace]}
            )
            yield format_sse(
                {
                    "stage": "features",
                    "domain": _domain(request.url),
                    "has_url": bool(request.url),
                    "posted_at": (request.posted_at or datetime.now()).isoformat(),
                }
            )
            yield format_sse({"stage": "fusion", "fusion": loaded.fusion})

            payload: dict[str, Any] = {
                "stage": "done",
                "title": request.title,
                "predicted_score": round(score, 2),
                "viral_probability": round(viral_probability(loaded, score), 4),
                "fusion": loaded.fusion,
                "word_trace": [word.model_dump() for word in trace],
                "domain": _domain(request.url),
                "embedding_dim": loaded.embeddings.embed_size,
                "checkpoint_source": loaded.source,
            }
            yield format_sse(payload)
        except HTTPException as error:
            yield format_sse({"stage": "error", "detail": str(error.detail)})
        except Exception as error:
            yield format_sse({"stage": "error", "detail": str(error)})

    return StreamingResponse(events(), media_type="text/event-stream")


if __name__ == "__main__":
    run_app(app)
