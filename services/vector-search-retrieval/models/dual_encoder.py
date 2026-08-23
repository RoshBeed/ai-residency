"""The mean-pooling dual encoder, which came before the GRU towers.

Encode a query or a document by averaging its word vectors, and score a triple by how
much closer the query is to the positive document than to the negative one. There are no
learned parameters at all above the frozen embeddings, which is the point: it is the
floor the GRU towers in `towers.py` have to beat to justify existing.

This file had real dead code in it. The bottom third was a copy of the GRU training loop
from what is now `training.py`, pasted in without the tower definitions it depends on, so
`qryTower` and `docTower` were referenced and never defined — an unconditional
`NameError` on any run. That block is gone; nothing else called it and the working
half above it is what the file is named for.
"""

import json

import torch
import torch.nn.functional as F
from torch import nn

from ai_residency_common import get_logger
from cbow.vocabulary import build_embedding_layer
from config import Config

__all__ = ["main", "mean_pool", "triplet_loss"]

logger = get_logger(__name__)


def mean_pool(token_ids: list[int], embedding_layer: nn.Embedding) -> torch.Tensor | None:
    """Average the word vectors of a token sequence.

    Args:
        token_ids: Token ids, possibly empty.
        embedding_layer: Frozen word embeddings.

    Returns:
        The mean vector, or None if `token_ids` was empty.
    """
    if not token_ids:
        return None
    tokens = torch.tensor(token_ids, dtype=torch.long)
    pooled: torch.Tensor = embedding_layer(tokens).mean(dim=0)
    return pooled


def triplet_loss(
    query: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: float,
) -> tuple[torch.Tensor, float, float]:
    """Score a triple by how much closer the query is to the positive document.

    Args:
        query: Query encoding.
        positive: Encoding of the relevant document.
        negative: Encoding of the irrelevant document.
        margin: How much further the negative must be before the loss is zero.

    Returns:
        The loss and the two cosine similarities that produced it.
    """
    similarity_positive = F.cosine_similarity(query.unsqueeze(0), positive.unsqueeze(0))
    similarity_negative = F.cosine_similarity(query.unsqueeze(0), negative.unsqueeze(0))
    loss = torch.clamp(margin - (similarity_positive - similarity_negative), min=0.0)
    return loss, float(similarity_positive.item()), float(similarity_negative.item())


def main() -> None:
    """Score the first few training triples with mean-pooled embeddings."""
    config = Config()
    embedding_layer = build_embedding_layer(config)

    with config.tokenized_triples_path.open() as handle:
        triples = json.load(handle)

    for index, triple in enumerate(triples["train"][:5], start=1):
        query = mean_pool(triple["query_tokens"], embedding_layer)
        positive = mean_pool(triple["positive_document_tokens"], embedding_layer)
        negative = mean_pool(triple["negative_document_tokens"], embedding_layer)
        if query is None or positive is None or negative is None:
            logger.info("Example %d: an input tokenized to nothing, skipping", index)
            continue

        loss, similarity_positive, similarity_negative = triplet_loss(
            query, positive, negative, config.margin
        )
        logger.info(
            "Example %d  query=%r  sim(pos)=%.4f  sim(neg)=%.4f  loss=%.4f",
            index,
            triple["query"],
            similarity_positive,
            similarity_negative,
            loss.item(),
        )


if __name__ == "__main__":
    main()
