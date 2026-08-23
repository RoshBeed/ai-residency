"""Mean-pool every triple with the trained CBOW embeddings and index the positives.

The average-pooling variant of the whole pipeline: no towers at all, documents
represented by the mean of their word vectors. It writes the same Redis index the GRU
path does, so the two can be pointed at the same query and compared.

It used to hardcode `REDIS_HOST = 'your-redis-host'` at module level and build a client
from it on import, which meant importing this file at all opened a connection to a
hostname that does not exist. Connection details come from `Config` now, the same as
every other module here.

No held-out evaluation exists for this variant. The GRU towers have one, in
`training.pairwise_accuracy`, and the two have still never been scored against each
other on the same triples — a real gap, recorded rather than papered over.
"""

import json

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from ai_residency_common import get_logger
from cbow.vocabulary import build_embedding_layer
from config import Config
from data_prep.generate_triples import SPLITS
from data_prep.tokenize_triples import Triple
from retrieval.query_engine import (
    ensure_index,
    get_redis_client,
)

__all__ = ["main", "pool_and_index"]

logger = get_logger(__name__)


def pool_and_index(config: Config, embedding_layer: nn.Embedding) -> int:
    """Mean-pool every triple and write the positive documents into Redis.

    Args:
        config: Run configuration.
        embedding_layer: Trained CBOW embeddings.

    Returns:
        How many documents were indexed.
    """
    with config.tokenized_triples_path.open() as handle:
        triples = json.load(handle)

    client = get_redis_client(config)
    ensure_index(client, config)

    processed: dict[str, list[Triple]] = {split: [] for split in SPLITS}
    indexed = 0
    with torch.no_grad():
        for split in SPLITS:
            for triple in tqdm(triples[split], desc=f"Pooling {split}"):
                vectors = {}
                for name, key in (
                    ("query_vector", "query_tokens"),
                    ("positive_document_vector", "positive_document_tokens"),
                    ("negative_document_vector", "negative_document_tokens"),
                ):
                    tokens = torch.tensor(triple[key], dtype=torch.long)
                    vectors[name] = embedding_layer(tokens).mean(dim=0).numpy()

                document_id = f"doc:{indexed}"
                client.hset(
                    document_id,
                    mapping={
                        "embedding": vectors["positive_document_vector"]
                        .astype(np.float32)
                        .tobytes(),
                        "text": triple["positive_document"],
                        "doc_id": document_id,
                    },
                )
                indexed += 1

                processed[split].append(
                    {name: [float(value) for value in vector] for name, vector in vectors.items()}
                    | {
                        "query": triple["query"],
                        "positive_document": triple["positive_document"],
                        "negative_document": triple["negative_document"],
                    }
                )

    output_path = config.data_dir / "triple_embeddings_cbow.json"
    with output_path.open("w") as handle:
        json.dump(processed, handle)
    logger.info("Wrote %s and indexed %d documents", output_path, indexed)
    return indexed


def main() -> None:
    """Entry point for the average-pooling variant."""
    config = Config()
    pool_and_index(config, build_embedding_layer(config))


if __name__ == "__main__":
    main()
