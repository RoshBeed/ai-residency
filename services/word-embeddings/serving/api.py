"""FastAPI service exposing the trained word embeddings.

The vectors themselves are the artifact this repo produces; the API lets a caller
inspect them — find the nearest neighbours of a word, or complete an analogy
(`king - man + woman`) — which is the intrinsic evidence that the training run
learned structure. It is a thin wrapper over `word_embeddings.evaluate`, not a
second implementation.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import torch
import torch.nn.functional as F
from ai_residency_common import get_logger, load_checkpoint, resolve_required, run_app
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from config import Config

__all__ = ["app"]

logger = get_logger(__name__)


class EmbeddingState:
    """The loaded embedding matrix and its vocabulary, or nothing at all."""

    def __init__(self) -> None:
        self.embeddings: torch.Tensor | None = None
        self.word_to_index: dict[str, int] = {}
        self.source: str = "none"

    @property
    def is_loaded(self) -> bool:
        """Whether the checkpoint was loaded successfully."""
        return self.embeddings is not None

    @property
    def index_to_word(self) -> dict[int, str]:
        """Inverse of the vocabulary, for turning a nearest neighbour back into a word."""
        return {index: word for word, index in self.word_to_index.items()}


state = EmbeddingState()


def _nearest_words(
    embeddings: torch.Tensor, vector: torch.Tensor, index_to_word: dict[int, str], k: int
) -> list[tuple[str, float]]:
    """Return the `k` vocabulary words closest to `vector` by cosine similarity."""
    similarities = F.cosine_similarity(embeddings, vector.unsqueeze(0)).cpu()
    top = similarities.topk(k)
    return [
        (index_to_word[int(index)], float(similarity))
        for index, similarity in zip(top.indices, top.values, strict=True)
    ]


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Load the checkpoint on startup, leaving the service degraded if it is absent."""
    try:
        config = Config()
        path, state.source = resolve_required(config.cbow_checkpoint, config.hf_repo_id)
        checkpoint = load_checkpoint(path)
        state.embeddings = checkpoint["model_state_dict"]["in_embed.weight"].cpu()
        state.word_to_index = checkpoint["extra"]["word_to_lemma_index"]
        logger.info("Loaded %s embeddings from %s", tuple(state.embeddings.shape), state.source)
    except (FileNotFoundError, KeyError, RuntimeError) as error:
        logger.error("Could not load embeddings, /nearest will return 503: %s", error)
    yield


app = FastAPI(
    title="Word Embeddings API",
    description="Nearest-neighbour and analogy lookup over CBOW embeddings trained from scratch",
    version="1.0.0",
    lifespan=lifespan,
)


class NearestRequest(BaseModel):
    """Input for `/nearest`."""

    word: str
    k: int = 5


class AnalogyRequest(BaseModel):
    """Input for `/analogy` — find the vector closest to `a - b + c`."""

    a: str
    b: str
    c: str
    k: int = 1


class NearestResponse(BaseModel):
    """Output for `/nearest` and `/analogy`."""

    query: str
    nearest: list[tuple[str, float]]
    checkpoint_source: str


@app.get("/")
async def root() -> dict[str, str]:
    """Describe the service."""
    return {
        "message": "Word Embeddings API",
        "endpoint": "/nearest",
        "usage": 'POST /nearest with {"word": "python"} or /analogy with '
        '{"a": "king", "b": "man", "c": "woman"}',
    }


@app.get("/health")
async def health() -> dict[str, Any]:
    """Liveness check used by compose and the portfolio hub."""
    return {
        "status": "ok",
        "model_loaded": state.is_loaded,
        "checkpoint_loaded": state.is_loaded,
        "checkpoint_source": state.source,
    }


@app.post("/nearest", response_model=NearestResponse)
async def nearest(request: NearestRequest) -> NearestResponse:
    """Return the `k` words closest to the given word's vector."""
    embeddings = state.embeddings
    if embeddings is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    index = state.word_to_index.get(request.word)
    if index is None:
        raise HTTPException(status_code=404, detail=f"word {request.word!r} not in vocabulary")

    vector = embeddings[index]
    nearest = _nearest_words(
        embeddings, vector, state.index_to_word, k=min(request.k + 1, embeddings.shape[0])
    )
    # The closest word is always the query itself; drop it.
    nearest = [pair for pair in nearest if pair[0] != request.word][: request.k]
    logger.info("Nearest to %r: %s", request.word, [word for word, _ in nearest])
    return NearestResponse(query=request.word, nearest=nearest, checkpoint_source=state.source)


@app.post("/analogy", response_model=NearestResponse)
async def analogy(request: AnalogyRequest) -> NearestResponse:
    """Return the word closest to `a - b + c` (the canonical `king - man + woman`)."""
    embeddings = state.embeddings
    if embeddings is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    for word in (request.a, request.b, request.c):
        if word not in state.word_to_index:
            raise HTTPException(status_code=404, detail=f"word {word!r} not in vocabulary")

    vector = (
        embeddings[state.word_to_index[request.a]]
        - embeddings[state.word_to_index[request.b]]
        + embeddings[state.word_to_index[request.c]]
    )
    nearest = _nearest_words(embeddings, vector, state.index_to_word, k=request.k)
    logger.info("Analogy %r - %r + %r -> %s", request.a, request.b, request.c, nearest)
    return NearestResponse(
        query=f"{request.a} - {request.b} + {request.c}",
        nearest=nearest,
        checkpoint_source=state.source,
    )


def main() -> None:
    """Entry point for `make serve`."""
    run_app(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
