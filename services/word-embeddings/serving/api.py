"""FastAPI service exposing the trained word embeddings.

The vectors themselves are the artifact this repo produces; the API lets a caller
inspect them — find the nearest neighbours of a word, or complete an analogy
(`king - man + woman`) — which is the intrinsic evidence that the training run
learned structure. It is a thin wrapper over `word_embeddings.evaluate`, not a
second implementation.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ai_residency_common import (
    add_health_routes,
    get_logger,
    load_checkpoint,
    resolve_required,
    run_app,
)
from config import Config

__all__ = ["app"]

logger = get_logger(__name__)


class EmbeddingState:
    """The loaded embedding matrix and its vocabulary, or nothing at all."""

    def __init__(self) -> None:
        self.embeddings: torch.Tensor | None = None
        self.word_to_index: dict[str, int] = {}
        self.index_to_word: dict[int, str] = {}
        self.source: str = "none"

    def load(self, embeddings: torch.Tensor, word_to_index: dict[str, int], source: str) -> None:
        """Adopt a loaded checkpoint, inverting the vocabulary once.

        `index_to_word` used to be a property, which rebuilt the whole inverse dict on
        every read -- and it is read on the hot path of both `/nearest` and `/analogy`,
        so every request paid a full pass over the vocabulary to answer a top-k query.
        """
        self.embeddings = embeddings
        self.word_to_index = word_to_index
        self.index_to_word = {index: word for word, index in word_to_index.items()}
        self.source = source

    @property
    def is_loaded(self) -> bool:
        """Whether the checkpoint was loaded successfully."""
        return self.embeddings is not None


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
        path, source = resolve_required(config.cbow_checkpoint, config.hf_repo_id)
        checkpoint = load_checkpoint(path)
        embeddings = checkpoint["model_state_dict"]["in_embed.weight"].cpu()
        state.load(embeddings, checkpoint["extra"]["word_to_lemma_index"], source)
        logger.info("Loaded %s embeddings from %s", tuple(embeddings.shape), source)
    except (FileNotFoundError, KeyError, RuntimeError) as error:
        logger.error("Could not load embeddings, /nearest will return 503: %s", error)
    yield


app = FastAPI(
    title="Word Embeddings API",
    description="Nearest-neighbour and analogy lookup over CBOW embeddings trained from scratch",
    version="1.0.0",
    lifespan=lifespan,
)


# `k` is bounded rather than free. Unbounded, a request for k=10**9 reached `topk` and
# came back as a 500 for what is a client error.
MAX_K = 100


class NearestRequest(BaseModel):
    """Input for `/nearest`."""

    word: str
    k: int = Field(default=5, ge=1, le=MAX_K)


class AnalogyRequest(BaseModel):
    """Input for `/analogy` — find the vector closest to `a - b + c`."""

    a: str
    b: str
    c: str
    k: int = Field(default=1, ge=1, le=MAX_K)


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


add_health_routes(
    app,
    ready=lambda: state.is_loaded,
    detail=lambda: {
        "model_loaded": state.is_loaded,
        "checkpoint_loaded": state.is_loaded,
        "checkpoint_source": state.source,
    },
    include_root=False,
)


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
