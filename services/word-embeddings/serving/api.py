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
    retrying,
    run_app,
)
from config import Config
from word2vec import nearest_words, valid_row_mask
from word_embeddings.evaluate import checkpoint_vocabulary

__all__ = ["app"]

logger = get_logger(__name__)


class EmbeddingState:
    """The loaded embedding matrix and its vocabulary, or nothing at all."""

    def __init__(self) -> None:
        self.embeddings: torch.Tensor | None = None
        self.unit_embeddings: torch.Tensor | None = None
        self.word_to_index: dict[str, int] = {}
        self.index_to_word: dict[int, str] = {}
        self.valid_rows: torch.Tensor | None = None
        self.source: str = "none"

    def load(self, embeddings: torch.Tensor, word_to_index: dict[str, int], source: str) -> None:
        """Adopt a loaded checkpoint, inverting the vocabulary once.

        `index_to_word` used to be a property, which rebuilt the whole inverse dict on
        every read -- and it is read on the hot path of both `/nearest` and `/analogy`,
        so every request paid a full pass over the vocabulary to answer a top-k query.
        The unit-normalised copy is precomputed here for the same reason: `/analogy`
        needs it on every call and it depends only on the checkpoint.
        """
        self.embeddings = embeddings
        self.unit_embeddings = F.normalize(embeddings, dim=1)
        self.word_to_index = word_to_index
        self.index_to_word = {index: word for word, index in word_to_index.items()}
        # Which rows can be named at all. Current checkpoints have a dense vocabulary
        # and no holes, but the one this service shipped for months had 258,196 rows
        # against 230,701 nameable words, so this is not hypothetical --
        # searching the other 27,495 raised KeyError on `/nearest`.
        self.valid_rows = valid_row_mask(embeddings.shape[0], self.index_to_word)
        self.source = source

    @property
    def is_loaded(self) -> bool:
        """Whether the checkpoint was loaded successfully."""
        return self.embeddings is not None


state = EmbeddingState()


def load_embeddings() -> bool:
    """Resolve and load the checkpoint. Returns whether the service can now serve.

    Called at startup and, if that fails, again from `/ready` on a rate limit -- so a
    checkpoint that was unreachable at boot is picked up when it becomes reachable,
    instead of leaving the pod not-ready for the rest of its life.
    """
    if state.is_loaded:
        return True
    try:
        config = Config()
        path, source = resolve_required(config.embedding_checkpoint, config.hf_repo_id)
        checkpoint = load_checkpoint(path)
        embeddings = checkpoint["model_state_dict"]["in_embed.weight"].cpu()
        state.load(embeddings, checkpoint_vocabulary(checkpoint["extra"]), source)
        logger.info("Loaded %s embeddings from %s", tuple(embeddings.shape), source)
    except (FileNotFoundError, KeyError, RuntimeError) as error:
        logger.error("Could not load embeddings, /nearest will return 503: %s", error)
        return False
    return True


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Load the checkpoint on startup, leaving the service degraded if it is absent."""
    load_embeddings()
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
    # Retried rather than decided once: a checkpoint that could not be resolved at
    # startup -- a DNS blip during a mass restart, say -- would otherwise leave this
    # pod not-ready permanently, with liveness unable to help because the process is
    # perfectly healthy.
    ready=retrying(load_embeddings),
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

    neighbours = nearest_words(
        embeddings,
        embeddings[index],
        state.index_to_word,
        k=request.k,
        # The closest word is always the query itself.
        exclude=request.word,
        mask=state.valid_rows,
    )
    logger.info("Nearest to %r: %s", request.word, [word for word, _ in neighbours])
    return NearestResponse(query=request.word, nearest=neighbours, checkpoint_source=state.source)


@app.post("/analogy", response_model=NearestResponse)
async def analogy(request: AnalogyRequest) -> NearestResponse:
    """Return the word closest to `a - b + c` (the canonical `king - man + woman`)."""
    embeddings = state.embeddings
    if embeddings is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    for word in (request.a, request.b, request.c):
        if word not in state.word_to_index:
            raise HTTPException(status_code=404, detail=f"word {word!r} not in vocabulary")

    # Unit vectors, not raw ones. 3CosAdd is defined on normalised embeddings, and
    # that is how `benchmarks.py` scores the analogy set -- so computing it on raw
    # vectors here meant the endpoint answered a slightly different question from the
    # accuracy the README reports. Word vector norms grow with frequency, so the raw
    # sum is tilted towards whichever of the three words is commonest rather than
    # towards the relation being asked about.
    unit = state.unit_embeddings
    assert unit is not None  # set together with `embeddings`, checked above
    vector = (
        unit[state.word_to_index[request.a]]
        - unit[state.word_to_index[request.b]]
        + unit[state.word_to_index[request.c]]
    )
    neighbours = nearest_words(
        embeddings,
        vector,
        state.index_to_word,
        k=request.k,
        exclude=(request.a, request.b, request.c),
        mask=state.valid_rows,
    )
    logger.info("Analogy %r - %r + %r -> %s", request.a, request.b, request.c, neighbours)
    return NearestResponse(
        query=f"{request.a} - {request.b} + {request.c}",
        nearest=neighbours,
        checkpoint_source=state.source,
    )


def main() -> None:
    """Entry point for `make serve`."""
    run_app(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
