"""The word-embedding model: CBOW or skip-gram, trained with negative sampling.

The algorithm lives in `libs/word2vec` now, shared with vector-search-retrieval,
which had grown its own copy. Read `word2vec/model.py` to understand word2vec; this
file is the service's view of it.

Three implementations of this existed across the portfolio, one per team that touched
it, and they were not rival CBOWs -- they were the same two matrices under different
objectives. This one used negative sampling with `in_embed`/`out_embed`; the retrieval
service used a full softmax with `emb`/`ffw`, where `ffw` is
`nn.Linear(embed, vocab, bias=False)` and its weight is the *same* `(vocab, embed)`
matrix under another name.

`build_cbow` became `build_word2vec` when the service started training skip-gram as
well: the architecture is a config field now, not a literal, so a sweep can vary it.

`load_cbow_embeddings` is how the rest of the repo reads the trained vectors back out
of a saved checkpoint, and the key it reads -- `in_embed.weight` -- is unchanged.
"""

from pathlib import Path

import torch

from ai_residency_common import get_logger
from config import Config
from word2vec import Word2Vec, Word2VecConfig

__all__ = ["build_word2vec", "load_cbow_embeddings"]

logger = get_logger(__name__)


def build_word2vec(vocab_size: int, config: Config) -> Word2Vec:
    """Build the model this service trains, for either architecture.

    Args:
        vocab_size: Tokens in the vocabulary, which is the embedding row count.
        config: Supplies the architecture and dimensions.

    Returns:
        An untrained model. Its `in_embed.weight` is what training produces and
        everything downstream reads.
    """
    return Word2Vec(
        Word2VecConfig(
            vocab_size=vocab_size,
            embed_size=config.embed_size,
            architecture=config.architecture,
            objective="negative_sampling",
            window_size=config.window_size,
            num_negatives=config.num_negatives,
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
