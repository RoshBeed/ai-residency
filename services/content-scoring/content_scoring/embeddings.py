"""Load the frozen CBOW embeddings this repo's scoring head consumes.

The word vectors are trained in the sibling `ai-residency-week1-word-embeddings`
repo and published to the Hugging Face Hub; this repo pulls them back down and reads
the vocabulary out of the checkpoint alongside the weights. The vocabulary travels
inside the checkpoint so the mapping cannot drift from the matrix it indexes.
"""

from pathlib import Path

import torch
from ai_residency_common import get_logger

__all__ = ["load_cbow_embeddings", "load_cbow_vocabulary"]

logger = get_logger(__name__)


def load_cbow_embeddings(model_path: str | Path) -> torch.Tensor:
    """Load the frozen input-embedding matrix from a CBOW checkpoint.

    Args:
        model_path: Path to the checkpoint saved by the word-embeddings repo's
            `train_cbow`.

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


def load_cbow_vocabulary(model_path: str | Path) -> dict[str, int]:
    """Load the word-to-lemma-index mapping a CBOW checkpoint was trained with.

    Args:
        model_path: Path to the same checkpoint passed to `load_cbow_embeddings`.

    Returns:
        The `word_to_lemma_index` mapping stored in the checkpoint's `extra`.
    """
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    vocabulary: dict[str, int] = checkpoint["extra"]["word_to_lemma_index"]
    logger.info("Loaded vocabulary with %d words", len(vocabulary))
    return vocabulary
