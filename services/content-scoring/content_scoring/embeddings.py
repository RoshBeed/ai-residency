"""Loading the extended word vectors this service's models read.

The base vectors are trained in the sibling `word-embeddings` service on text8;
`vocab.py` extends that vocabulary onto the Hacker News title corpus and publishes the
result as `hn_embeddings.pt`. This module is how everything downstream gets it back.

The vocabulary travels inside the checkpoint alongside the matrix, so the mapping cannot
drift from the rows it indexes. That is not defensive style -- this repo shipped a
version where the corpus was encoded with word ids while the checkpoint saved lemma ids,
and every neighbour the service returned for months was read out of the wrong index
space.
"""

from pathlib import Path

import torch

from ai_residency_common import get_logger, load_checkpoint

__all__ = ["ExtendedEmbeddings", "load_extended_embeddings"]

logger = get_logger(__name__)


class ExtendedEmbeddings:
    """The extended matrix and the vocabulary that indexes it.

    Attributes:
        matrix: `(vocab_size, embed_size)` word vectors on CPU.
        word_to_index: Word to row.
        num_pretrained: Rows inherited from text8; anything above came from Hacker News.
    """

    def __init__(
        self, matrix: torch.Tensor, word_to_index: dict[str, int], num_pretrained: int
    ) -> None:
        self.matrix = matrix
        self.word_to_index = word_to_index
        self.num_pretrained = num_pretrained

    @property
    def embed_size(self) -> int:
        """Width of one word vector."""
        return int(self.matrix.shape[1])


def load_extended_embeddings(path: str | Path) -> ExtendedEmbeddings:
    """Load the extended embedding checkpoint.

    Args:
        path: Path to `hn_embeddings.pt`.

    Returns:
        The matrix, its vocabulary and how much of it is inherited.

    Raises:
        KeyError: If the checkpoint carries no vocabulary, which makes the matrix
            unreadable rather than merely undocumented.
    """
    payload = load_checkpoint(path)
    extra = payload["extra"]
    if "word_to_index" not in extra:
        raise KeyError(f"{path} carries no vocabulary; the matrix cannot be indexed")
    matrix = payload["model_state_dict"]["in_embed.weight"].cpu()
    word_to_index: dict[str, int] = extra["word_to_index"]
    logger.info(
        "Loaded %d x %d extended embeddings (%d from text8, %d from Hacker News)",
        matrix.shape[0],
        matrix.shape[1],
        extra.get("num_pretrained", 0),
        matrix.shape[0] - extra.get("num_pretrained", 0),
    )
    return ExtendedEmbeddings(matrix, word_to_index, int(extra.get("num_pretrained", 0)))
