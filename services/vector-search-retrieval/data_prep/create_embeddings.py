"""Early experiment: mean-pool every triple with randomly initialised embeddings.

Kept because it is the control the rest of the pipeline is measured against. The
embeddings here are never trained, so any structure in the resulting vectors comes from
word overlap alone. `create_triple_embeddings` is the same computation with trained CBOW
weights, and the difference between the two is what those weights bought.
"""

import json

import torch
from ai_residency_common import get_logger, set_seed
from torch import nn
from tqdm import tqdm

from cbow.vocabulary import load_words_to_ids
from config import Config
from data_prep.generate_triples import SPLITS
from data_prep.tokenize_triples import Triple

__all__ = ["main", "mean_pool", "process_triples"]

logger = get_logger(__name__)


def mean_pool(token_ids: list[int], embedding_layer: nn.Embedding) -> list[float]:
    """Average the embeddings of a token sequence.

    Args:
        token_ids: Token ids to pool.
        embedding_layer: Embeddings to look up.

    Returns:
        The mean vector as a plain list, ready to serialise.
    """
    with torch.no_grad():
        tokens = torch.tensor(token_ids, dtype=torch.long)
        pooled = embedding_layer(tokens).mean(dim=0)
    return [float(value) for value in pooled]


def process_triples(
    triples: dict[str, list[Triple]],
    embedding_layer: nn.Embedding,
) -> dict[str, list[Triple]]:
    """Mean-pool every triple in every split.

    Args:
        triples: Tokenized triples keyed by split.
        embedding_layer: Embeddings to pool with.

    Returns:
        Vectors plus the original text, keyed by split.
    """
    processed: dict[str, list[Triple]] = {split: [] for split in SPLITS}
    for split in SPLITS:
        for triple in tqdm(triples[split], desc=f"Pooling {split}"):
            processed[split].append(
                {
                    "query_vector": mean_pool(triple["query_tokens"], embedding_layer),
                    "positive_document_vector": mean_pool(
                        triple["positive_document_tokens"], embedding_layer
                    ),
                    "negative_document_vector": mean_pool(
                        triple["negative_document_tokens"], embedding_layer
                    ),
                    "query": triple["query"],
                    "positive_document": triple["positive_document"],
                    "negative_document": triple["negative_document"],
                }
            )
    return processed


def main() -> None:
    """Pool all triples with untrained embeddings and write them out."""
    config = Config()
    set_seed(config.seed)

    words_to_ids = load_words_to_ids(config)
    embedding_layer = nn.Embedding(len(words_to_ids), config.embedding_dim)
    nn.init.xavier_uniform_(embedding_layer.weight)

    with config.tokenized_triples_path.open() as handle:
        triples = json.load(handle)

    processed = process_triples(triples, embedding_layer)
    output_path = config.data_dir / "triple_embeddings.json"
    with output_path.open("w") as handle:
        json.dump(processed, handle)
    logger.info("Wrote %s", output_path)


if __name__ == "__main__":
    main()
