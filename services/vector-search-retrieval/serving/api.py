"""FastAPI retrieval service.

A thin wrapper over `query_engine.run_query`. It owns no model, vocabulary or Redis
loading of its own, so the API, the CLI and the Streamlit UI cannot drift apart in what
they actually search with.

Its only real job beyond that is translating the two failure modes `run_query` documents
into HTTP: a missing checkpoint and an unreachable Redis both mean "not ready yet", which
is a 503 rather than a traceback.

`/query/stream` relays the same pipeline as Server-Sent Events: vocabulary check, then
the query tower's encoding, then Redis KNN search, then the ranked results.
"""

import base64
from collections.abc import Generator
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import redis
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ai_residency_common import add_health_routes, format_sse, get_logger, retrying, run_app
from config import Config
from retrieval import query_engine
from retrieval.query_engine import (
    embed_query,
    load_query_tower,
    load_vocabulary,
    run_query,
)

__all__ = ["app"]

logger = get_logger(__name__)

_config = Config()


# Bounded, not free. `top_k` goes straight into the Redis query, and a negative
# value made Redis reject the *query* -- which this then reported as
# "Redis unreachable: Syntax error", a 503 blaming the datastore for the caller's
# input. Validation catches it first and answers 422.
MAX_TOP_K = 100


class QueryRequest(BaseModel):
    """Input for `/query`."""

    text: str
    top_k: int = Field(default=5, ge=1, le=MAX_TOP_K)


class QueryResult(BaseModel):
    """One retrieved document."""

    rank: int
    score: float
    text: str
    embedding: str | None = None  # base64-encoded float32 vector for pairwise similarity


class WordTrace(BaseModel):
    """Whether one word of the query was in the trained vocabulary."""

    word: str
    known: bool


class QueryResponse(BaseModel):
    """Output of `/query`."""

    query: str
    results: list[QueryResult]
    word_trace: list[WordTrace] = []
    embedding_norm: float = 0.0


@asynccontextmanager
async def lifespan(_: FastAPI) -> Any:
    """Warm the tower and vocabulary caches at startup.

    Deliberately non-fatal. `load_query_tower` is `lru_cache`d and does not cache
    exceptions, so if the model is not trained yet the process still starts, `/` still
    reports healthy, and the first `/query` retries the load and gets a clean 503.
    """
    ensure_loaded()
    yield


def ensure_loaded() -> bool:
    """Warm the tower and vocabulary if they are not warm, and report readiness.

    `load_query_tower` is lru_cached and does not cache exceptions, so re-attempting
    after a failure genuinely retries rather than replaying the same error.
    """
    if query_engine.query_tower_source != "none":
        return True
    try:
        load_vocabulary()
        load_query_tower()
        logger.info("Query tower and vocabulary loaded")
    except FileNotFoundError as error:
        logger.warning("Model not ready, will retry: %s", error)
        return False
    except RuntimeError as error:
        # A tower checkpoint carries the frozen embedding matrix, so it is only loadable
        # against the vocabulary that produced it. Rebuild the vocabulary without
        # retraining the towers and `load_state_dict` raises a size mismatch -- an
        # unusable checkpoint, which is a not-ready condition, not a reason to take the
        # process down on startup. Logged with the shapes, because the message is the
        # only thing that says the two artifacts are from different runs.
        logger.error("Checkpoint does not fit this vocabulary, /query will 503: %s", error)
        return False
    return query_engine.query_tower_source != "none"


app = FastAPI(
    title="Dual-Encoder Document Retrieval API",
    description="Query-tower encoding plus Redis KNN search over MS MARCO",
    version="1.0.0",
    lifespan=lifespan,
)


add_health_routes(
    app,
    # Retried rather than decided once. A resolve that failed at startup -- a DNS blip
    # during a mass restart, say -- would otherwise leave this pod not-ready for the
    # rest of its life, with liveness unable to help because the process itself is
    # perfectly healthy. `retrying` re-attempts, at most once a minute, from /ready.
    ready=retrying(ensure_loaded),
    detail=lambda: {
        "checkpoint_loaded": query_engine.query_tower_source != "none",
        "checkpoint_source": query_engine.query_tower_source,
    },
)


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """Retrieve the documents nearest to a query.

    Args:
        request: The query text and how many results to return.

    Returns:
        Ranked results, closest first.

    Raises:
        HTTPException: 503 if the model is untrained or Redis is unreachable.
    """
    try:
        vocabulary = load_vocabulary()
        word_trace = [
            WordTrace(word=word, known=word in vocabulary) for word in request.text.strip().split()
        ]
        embedding = embed_query(request.text)
        raw_results = run_query(request.text, top_k=request.top_k)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=f"Model not ready: {error}") from error
    except RuntimeError as error:
        # A checkpoint that does not fit the current vocabulary is unusable in the same
        # way a missing one is, and must answer the same 503 rather than a 500.
        raise HTTPException(status_code=503, detail=f"Model not ready: {error}") from error
    except redis.exceptions.RedisError as error:
        raise HTTPException(status_code=503, detail=f"Redis unreachable: {error}") from error

    return QueryResponse(
        query=request.text,
        results=[
            QueryResult(
                rank=rank,
                score=score,
                text=text,
                embedding=base64.b64encode(emb.tobytes()).decode() if emb is not None else None,
            )
            for rank, (score, text, emb) in enumerate(raw_results, start=1)
        ],
        word_trace=word_trace,
        embedding_norm=float(np.linalg.norm(embedding)),
    )


@app.post("/query/stream")
def query_stream(request: QueryRequest) -> StreamingResponse:
    """Same pipeline as `/query`, streamed as Server-Sent Events per stage.

    The retrieval itself is fast (one encoding + one Redis KNN search), so these
    stages are fine-grained milestones within it. This endpoint exists so every
    service's API has a consistent SSE contract.

    Args:
        request: The query text and how many results to return.

    Returns:
        A `text/event-stream` response. Events:
        `{"stage": "vocabulary", "word_trace": [...]}`,
        `{"stage": "encoded", "embedding_norm": ...}`,
        `{"stage": "done", ...}` or `{"stage": "error", "detail": ...}`.
    """
    try:
        vocabulary = load_vocabulary()
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=f"Model not ready: {error}") from error

    word_trace = [
        WordTrace(word=word, known=word in vocabulary) for word in request.text.strip().split()
    ]

    def event_generator() -> Generator[str, None, None]:
        try:
            yield format_sse(
                {"stage": "vocabulary", "word_trace": [w.model_dump() for w in word_trace]}
            )

            embedding = embed_query(request.text)
            yield format_sse(
                {"stage": "encoded", "embedding_norm": float(np.linalg.norm(embedding))}
            )

            raw_results = run_query(request.text, top_k=request.top_k)
            yield format_sse(
                {
                    "stage": "done",
                    "query": request.text,
                    "results": [
                        {
                            "rank": rank,
                            "score": score,
                            "text": text,
                            "embedding": (
                                base64.b64encode(emb.tobytes()).decode()
                                if emb is not None
                                else None
                            ),
                        }
                        for rank, (score, text, emb) in enumerate(raw_results, start=1)
                    ],
                    "word_trace": [w.model_dump() for w in word_trace],
                    "embedding_norm": float(np.linalg.norm(embedding)),
                }
            )
        except redis.exceptions.RedisError as error:
            yield format_sse({"stage": "error", "detail": f"Redis unreachable: {error}"})
        except Exception as error:
            yield format_sse({"stage": "error", "detail": str(error)})

    return StreamingResponse(event_generator(), media_type="text/event-stream")


def main() -> None:
    """Entry point for `make api`."""
    run_app(app, host=_config.api_host, port=_config.api_port)


if __name__ == "__main__":
    main()
