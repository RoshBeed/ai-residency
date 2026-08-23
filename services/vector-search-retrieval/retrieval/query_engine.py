"""Query-tower encoding plus Redis KNN search.

The one place retrieval actually happens. The CLI, the FastAPI service and the Streamlit
UI all call `run_query` rather than each loading a tower of their own, which is what they
used to do — three copies of the loading code meant three chances to load the wrong
weights, and at one point the query side was being encoded with the *document* tower.

Loading is lazy and cached rather than done at import. The CLI would rather fail fast,
but Streamlit needs to catch a missing checkpoint and render an error instead of dying
at process start, and a module that loads a model on import cannot do that.
"""

import pickle
from contextlib import suppress
from functools import lru_cache
from typing import cast

import numpy as np
import redis
import torch

from ai_residency_common import get_logger, load_model, resolve_required
from cbow.vocabulary import build_embedding_placeholder
from config import Config
from models.towers import GruTower, encode

__all__ = [
    "SearchResult",
    "embed_query",
    "ensure_index",
    "get_redis_client",
    "load_query_tower",
    "load_vocabulary",
    "run_query",
    "tokenize",
]

logger = get_logger(__name__)

# One result: cosine distance from the query, and the document text. Lower is closer.
SearchResult = tuple[float, str, np.ndarray | None]

_config = Config()


def get_redis_client(config: Config | None = None) -> redis.Redis:
    """Return a Redis client.

    Deliberately not cached. Streamlit reruns its script top to bottom on every
    interaction, and a cached connection object there is a reliable source of "works
    once, then fails". Constructing the client is cheap; it connects lazily.

    Args:
        config: Run configuration. Defaults to the module-level one.

    Returns:
        A binary-safe Redis client.
    """
    config = config or _config
    return redis.Redis(
        host=config.redis_host,
        port=config.redis_port,
        password=config.redis_password,
        decode_responses=False,
    )


def ensure_index(client: redis.Redis, config: Config | None = None) -> bool:
    """Create the RediSearch vector index if it does not exist.

    Args:
        client: Redis client.
        config: Run configuration. Defaults to the module-level one.

    Returns:
        True if the index already existed, False if it was just created.
    """
    config = config or _config
    try:
        client.execute_command(f"FT.INFO {config.index_name}")
        return True
    except redis.ResponseError:
        client.execute_command(
            f"FT.CREATE {config.index_name} ON HASH PREFIX 1 doc: SCHEMA "
            f"embedding VECTOR HNSW 6 TYPE FLOAT32 DIM {config.hidden_size} "
            f"DISTANCE_METRIC COSINE text TEXT doc_id TAG"
        )
        logger.info("Created RediSearch index %s", config.index_name)
        return False


# Where load_query_tower got its weights: "local", "hub", or "none" before it runs.
# The API reports this, so a service quietly running on an untrained tower is visible.
query_tower_source = "none"


@lru_cache(maxsize=1)
def load_vocabulary() -> dict[str, int]:
    """Load and cache the word-to-token-id vocabulary.

    Pulled from the Hub when absent, like the towers. It lives under `data/` rather
    than `checkpoints/` but it is just as much a trained artifact: without the exact
    vocabulary the towers were trained against, token ids mean nothing and retrieval
    silently returns rubbish.

    Raises:
        FileNotFoundError: If the vocabulary is neither on disk nor on the Hub.
    """
    path, _ = resolve_required(
        _config.words_to_ids_path,
        _config.hf_dataset_repo_id,
        repo_type="dataset",
        rebuild_hint="Build the vocabulary with `make train-cbow`.",
    )

    with path.open("rb") as handle:
        vocabulary: dict[str, int] = pickle.load(handle)
    return vocabulary


@lru_cache(maxsize=1)
def load_query_tower() -> GruTower:
    """Load and cache the trained query tower.

    The tower is gitignored, so a host that did not train it here has no local file
    and falls back to the Hub. Its source is recorded in `query_tower_source` for
    the API's health check to report.

    The embedding layer is a placeholder: the checkpoint about to be loaded carries
    the trained word vectors itself, so serving needs no CBOW checkpoint of its own.

    Raises:
        FileNotFoundError: If the tower or the vocabulary cannot be obtained.
    """
    global query_tower_source

    embedding_layer = build_embedding_placeholder(_config)
    tower = GruTower(embedding_layer, _config.hidden_size)
    loaded_tower, query_tower_source = load_model(
        _config.query_tower_path, _config.hf_repo_id, tower
    )
    return cast(GruTower, loaded_tower)


def tokenize(text: str) -> list[int]:
    """Map text to token ids, sending unknown words to the padding id.

    Args:
        text: Raw query text.

    Returns:
        Token ids, one per whitespace-separated word.
    """
    vocabulary = load_vocabulary()
    return [vocabulary.get(word, 0) for word in text.strip().split()]


def embed_query(text: str) -> np.ndarray:
    """Encode a query with the trained query tower.

    Args:
        text: Raw query text.

    Returns:
        A float32 vector of width `hidden_size`. A query whose words are all unknown
        encodes to zeros, since there is nothing for the GRU to read.
    """
    with torch.no_grad():
        embedding = encode(load_query_tower(), tokenize(text))
    if embedding is None:
        return np.zeros(_config.hidden_size, dtype=np.float32)
    return embedding.detach().cpu().numpy().astype(np.float32)


def _build_result(
    raw_score: object,
    raw_text: object,
    raw_embedding: object | None = None,
) -> SearchResult | None:
    """Turn one raw RediSearch field triple into a result.

    Args:
        raw_score: The `score` field as returned by Redis.
        raw_text: The `text` field as returned by Redis.
        raw_embedding: The `embedding` field as returned by Redis, or None.

    Returns:
        A `(score, text, embedding)` triple, or None if the score was absent or
        unparseable.
    """
    if not isinstance(raw_score, bytes | str | float | int):
        return None
    try:
        score = float(raw_score)
    except ValueError:
        return None
    text = raw_text.decode("utf-8", errors="ignore") if isinstance(raw_text, bytes) else ""
    embedding: np.ndarray | None = None
    if raw_embedding is not None and isinstance(raw_embedding, bytes):
        with suppress(ValueError):
            embedding = np.frombuffer(raw_embedding, dtype=np.float32)
    return score, text or "[No text found]", embedding


def _parse_search_results(response: object) -> list[SearchResult]:
    """Read `(score, text)` pairs out of an `FT.SEARCH` reply.

    Two shapes are handled because redis-py changed one. Version 5 and later parse the
    reply into `{b"results": [{b"extra_attributes": {...}}, ...]}`; before that it was a
    flat `[count, doc_id, [field, value, ...], ...]` array. Confirmed against redis-py
    8.0.1 and the `redis/redis-stack-server` image this repo runs.

    Args:
        response: Whatever `execute_command` returned.

    Returns:
        Parsed results, unsorted.
    """
    raw_pairs: list[tuple[object, object, object | None]] = []

    if isinstance(response, dict):
        documents = response.get(b"results", [])
        for document in documents if isinstance(documents, list) else []:
            attributes = document.get(b"extra_attributes", {}) if isinstance(document, dict) else {}
            raw_pairs.append(
                (
                    attributes.get(b"score"),
                    attributes.get(b"text"),
                    attributes.get(b"embedding"),
                )
            )
    elif isinstance(response, list) and len(response) > 1:
        for index in range(1, len(response) - 1, 2):
            fields = response[index + 1]
            if not isinstance(fields, list) or len(fields) < 2:
                continue
            attributes = dict(zip(fields[::2], fields[1::2], strict=False))
            raw_pairs.append(
                (
                    attributes.get(b"score"),
                    attributes.get(b"text"),
                    attributes.get(b"embedding"),
                )
            )

    parsed = (_build_result(score, text, emb) for score, text, emb in raw_pairs)
    return [result for result in parsed if result is not None]


def run_query(text: str, top_k: int | None = None) -> list[SearchResult]:
    """Embed a query and return its nearest documents from the index.

    Args:
        text: Raw query text.
        top_k: How many results to return. Defaults to `Config.top_k`.

    Returns:
        `(cosine_distance, document_text)` pairs, closest first. Empty if the index has
        nothing in it.

    Raises:
        FileNotFoundError: If the vocabulary or the trained tower is missing.
        redis.exceptions.RedisError: If Redis is unreachable. Callers decide how to
            present that; the CLI prints it and the API turns it into a 503.
    """
    top_k = top_k if top_k is not None else _config.top_k
    client = get_redis_client()
    ensure_index(client)

    response = client.execute_command(
        "FT.SEARCH",
        _config.index_name,
        f"*=>[KNN {top_k} @embedding $vec as score]",
        "RETURN",
        3,
        "text",
        "score",
        "embedding",
        "PARAMS",
        2,
        "vec",
        embed_query(text).tobytes(),
        "DIALECT",
        2,
    )

    results = _parse_search_results(response)
    results.sort(key=lambda result: result[0])
    return results
