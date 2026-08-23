"""Stage 4: embed the corpus with the trained document tower and index it in Redis.

This is the half of retrieval that happens offline. Every document is encoded once, here,
so that answering a query later only costs one encode plus an approximate
nearest-neighbour lookup — which is the entire reason a dual encoder is worth training
instead of scoring every (query, document) pair with a cross-encoder.
"""

import json
from typing import cast

import numpy as np
import redis
import torch
from ai_residency_common import get_logger, load_model, resolve_required
from tqdm import tqdm

from cbow.vocabulary import build_embedding_placeholder
from config import Config
from models.towers import GruTower, encode
from retrieval.query_engine import (
    ensure_index,
    get_redis_client,
)

__all__ = ["index_documents", "load_document_tower", "main", "unique_documents"]

logger = get_logger(__name__)


def load_document_tower(config: Config) -> GruTower:
    """Load the trained document tower, pulling it from the Hub if absent locally.

    `query_engine.load_query_tower` already resolves its side this way. This one
    checked `Path.exists()` and gave up, so indexing failed on any host without a
    local copy even though the tower is published -- the container case.

    The embedding layer is a placeholder, for the same reason as on the query side:
    the tower checkpoint carries the trained word vectors.

    Args:
        config: Supplies the checkpoint and embedding paths and `hf_repo_id`.

    Returns:
        The tower, in eval mode.

    Raises:
        FileNotFoundError: If the tower is neither on disk nor on the Hub.
    """
    tower = GruTower(build_embedding_placeholder(config), config.hidden_size)
    loaded, source = load_model(config.document_tower_path, config.hf_repo_id, tower)
    logger.info("Loaded document tower from %s", source)
    loaded.eval()
    return cast(GruTower, loaded)


def unique_documents(config: Config) -> list[tuple[list[int], str]]:
    """Collect every distinct positive document across all splits.

    Deduplication is by token sequence rather than by text, because that is what
    actually determines the embedding, and MS MARCO repeats passages across queries.

    Args:
        config: Supplies `tokenized_triples_path`.

    Returns:
        `(token_ids, text)` pairs.
    """
    # Derived here rather than downloaded, and not committed, so a container without
    # the sibling repo bind-mounted gets it from the dataset repo instead.
    path, _ = resolve_required(
        config.tokenized_triples_path,
        config.hf_dataset_repo_id,
        repo_type="dataset",
        rebuild_hint="Build it with `make train-triples`.",
    )

    with path.open() as handle:
        triples = json.load(handle)

    seen: set[tuple[int, ...]] = set()
    documents: list[tuple[list[int], str]] = []
    for split in ("train", "validation", "test"):
        for triple in triples[split]:
            tokens = tuple(triple["positive_document_tokens"])
            if tokens in seen:
                continue
            seen.add(tokens)
            documents.append((list(tokens), triple["positive_document"]))
    return documents


def index_documents(config: Config, client: redis.Redis) -> int:
    """Encode every document and write it into the RediSearch index.

    Args:
        config: Run configuration.
        client: Redis client to write through.

    Returns:
        How many documents were indexed.
    """
    tower = load_document_tower(config)
    documents = unique_documents(config)
    logger.info("Indexing %d unique positive documents", len(documents))
    ensure_index(client, config)

    indexed = 0
    with torch.no_grad():
        for index, (tokens, text) in enumerate(tqdm(documents, desc="Indexing")):
            embedding = encode(tower, tokens)
            if embedding is None:
                continue
            vector = embedding.detach().cpu().numpy().astype(np.float32)
            document_id = f"doc:{index}"
            client.hset(
                document_id,
                mapping={
                    "embedding": vector.tobytes(),
                    "text": text,
                    "doc_id": document_id,
                },
            )
            indexed += 1

    logger.info("Indexed %d documents into %s", indexed, config.index_name)
    return indexed


def main() -> None:
    """Entry point for `make index`."""
    config = Config()
    index_documents(config, get_redis_client(config))


if __name__ == "__main__":
    main()
