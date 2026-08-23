"""FastAPI service predicting a content score from a post title.

A thin HTTP wrapper over the same functions `predict.py` uses: load the models
once at startup, average a title's word vectors, run them through the scorer.
`/predict/stream` relays the same pipeline as Server-Sent Events.
"""

from collections.abc import AsyncIterator, Generator
from contextlib import asynccontextmanager
from typing import Any

import torch
from ai_residency_common import format_sse, get_logger, run_app
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config import Config
from content_scoring.predict import ModelState, create_title_embedding, load_models, predict_score

__all__ = ["app"]

logger = get_logger(__name__)

state = ModelState()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Load models on startup, leaving the service degraded if they are missing."""
    try:
        loaded = load_models(Config())
        state.predictor = loaded.predictor
        state.embeddings = loaded.embeddings
        state.word_to_index = loaded.word_to_index
        state.embedding_dim = loaded.embedding_dim
        state.cbow_source = loaded.cbow_source
        state.predictor_source = loaded.predictor_source
    except (FileNotFoundError, KeyError, RuntimeError) as error:
        logger.error("Could not load models, /predict will return 503: %s", error)
    yield


app = FastAPI(
    title="Content Score Predictor",
    description="Predict an engagement score from a title",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictionRequest(BaseModel):
    """Input for `/predict`."""

    title: str


class WordTrace(BaseModel):
    """Whether a single title word was found in the CBOW vocabulary."""

    word: str
    known: bool


class PredictionResponse(BaseModel):
    """Output of `/predict`."""

    title: str
    predicted_score: float
    api_source: str = "FastAPI Service"
    model_info: str = "CBOW + ScorePredictor"
    word_trace: list[WordTrace] = []
    embedding_dim: int = 0
    title_embedding_norm: float = 0.0


@app.get("/")
async def root() -> dict[str, str]:
    """Describe the service."""
    return {
        "message": "Content Score Predictor API",
        "endpoint": "/predict",
        "usage": 'POST /predict with {"title": "your title here"}',
    }


@app.get("/health")
async def health() -> dict[str, Any]:
    """Liveness check used by compose and the portfolio hub."""
    return {
        "status": "ok",
        "model_loaded": state.is_loaded,
        "checkpoint_loaded": state.is_loaded,
        "checkpoint_source": {
            "cbow": state.cbow_source,
            "predictor": state.predictor_source,
        },
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(request: PredictionRequest) -> PredictionResponse:
    """Predict the score for a single title."""
    embeddings = state.embeddings
    if embeddings is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    title_vector = create_title_embedding(
        request.title, state.word_to_index, embeddings, state.embedding_dim
    )
    score = predict_score(state, request.title)

    word_trace = [
        WordTrace(
            word=word,
            known=state.word_to_index.get(word) is not None
            and state.word_to_index[word] < embeddings.shape[0],
        )
        for word in request.title.lower().split()
    ]

    logger.info("Predicted %.2f for %r", score, request.title)
    return PredictionResponse(
        title=request.title,
        predicted_score=round(score, 2),
        word_trace=word_trace,
        embedding_dim=state.embedding_dim,
        title_embedding_norm=float(title_vector.norm().item()),
    )


@app.post("/predict/stream")
def predict_stream(request: PredictionRequest) -> StreamingResponse:
    """Same pipeline as `/predict`, streamed as Server-Sent Events per stage."""
    embeddings = state.embeddings
    predictor = state.predictor
    if embeddings is None or predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    def event_generator() -> Generator[str, None, None]:
        try:
            word_trace = [
                WordTrace(
                    word=word,
                    known=state.word_to_index.get(word) is not None
                    and state.word_to_index[word] < embeddings.shape[0],
                )
                for word in request.title.lower().split()
            ]
            yield format_sse(
                {"stage": "word_lookup", "word_trace": [w.model_dump() for w in word_trace]}
            )

            title_vector = create_title_embedding(
                request.title, state.word_to_index, embeddings, state.embedding_dim
            )
            yield format_sse(
                {
                    "stage": "embedding",
                    "embedding_dim": state.embedding_dim,
                    "title_embedding_norm": float(title_vector.norm().item()),
                }
            )

            with torch.no_grad():
                score = float(predictor(title_vector).item())

            yield format_sse(
                {
                    "stage": "done",
                    "title": request.title,
                    "predicted_score": round(score, 2),
                    "word_trace": [w.model_dump() for w in word_trace],
                    "embedding_dim": state.embedding_dim,
                    "title_embedding_norm": float(title_vector.norm().item()),
                }
            )
        except Exception as error:
            yield format_sse({"stage": "error", "detail": str(error)})

    return StreamingResponse(event_generator(), media_type="text/event-stream")


def main() -> None:
    """Entry point for `make serve`."""
    run_app(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
