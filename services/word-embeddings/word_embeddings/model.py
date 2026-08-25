"""The word-embedding model: CBOW trained with negative sampling.

The algorithm lives in `libs/word2vec` now, shared with vector-search-retrieval,
which had grown its own copy. Read `word2vec/model.py` to understand word2vec; this
file is the service's view of it.

Three implementations of this existed across the portfolio, one per team that touched
it, and they were not rival CBOWs -- they were the same two matrices under different
objectives. This one used negative sampling with `in_embed`/`out_embed`; the retrieval
service used a full softmax with `emb`/`ffw`, where `ffw` is
`nn.Linear(embed, vocab, bias=False)` and its weight is the *same* `(vocab, embed)`
matrix under another name.

`load_cbow_embeddings` is how the rest of the repo reads the trained vectors back out
of a saved checkpoint, and the key it reads -- `in_embed.weight` -- is unchanged.
"""

from pathlib import Path

import torch

from ai_residency_common import get_logger
from word2vec import Word2Vec, Word2VecConfig

__all__ = ["build_cbow", "load_cbow_embeddings"]

logger = get_logger(__name__)


def build_cbow(
    vocab_size: int,
    embed_size: int,
    *,
    window_size: int = 4,
    num_negatives: int = 5,
) -> Word2Vec:
    """Build the CBOW model this service trains.

    Args:
        vocab_size: Tokens in the vocabulary.
        embed_size: Width of each word vector.
        window_size: Context words either side of the target.
        num_negatives: Negatives scored against each positive.

    Returns:
        An untrained model. Its `in_embed.weight` is what training produces and
        everything downstream reads.
    """
    return Word2Vec(
        Word2VecConfig(
            vocab_size=vocab_size,
            embed_size=embed_size,
            architecture="cbow",
            objective="negative_sampling",
            window_size=window_size,
            num_negatives=num_negatives,
        )
    )


def load_cbow_embeddings(model_path: str | Path) -> torch.Tensor:
    """Load the trained input-embedding matrix from a CBOW checkpoint.

    Args:
        model_path: Path to the checkpoint saved by `train_cbow`.

    Returns:
        The `in_embed` weight matrix, shape `(vocab_size, embed_size)`, on CPU.
    """
    logger.info("Loading CBOW embeddings from %s", model_path)
    # map_location='cpu': checkpoints trained on MPS carry mps:0 tensor metadata and
    # fail to deserialise on a CPU-only host unless loaded onto the CPU explicitly.
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    embeddings: torch.Tensor = checkpoint["model_state_dict"]["in_embed.weight"]
    embeddings = embeddings.cpu()
    logger.info("Loaded embeddings with shape %s", tuple(embeddings.shape))
    return embeddings
