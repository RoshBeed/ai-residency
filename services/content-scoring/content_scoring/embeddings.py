"""Load the frozen CBOW embeddings this repo's scoring head consumes.

The word vectors are trained in the sibling `word-embeddings` service
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
            `train`.

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
    """Load the vocabulary a CBOW checkpoint's embedding rows are indexed by.

    Args:
        model_path: Path to the same checkpoint passed to `load_cbow_embeddings`.

    Returns:
        The `word_to_index` mapping stored in the checkpoint's `extra`, falling back
        to `word_to_lemma_index` for checkpoints written before the two were
        untangled.

    Raises:
        KeyError: If the checkpoint carries no vocabulary at all.

    The fallback exists because the word-embeddings service used to save a mapping
    from words to *lemma* ids while training the matrix on *word* ids -- two
    different index spaces -- so the vectors this function returned were being looked
    up through the wrong mapping. A checkpoint from before the fix still loads, and
    still scores badly; retraining is what fixes it.
    """
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    extra = checkpoint["extra"]
    for key in ("word_to_index", "word_to_lemma_index"):
        if key in extra:
            vocabulary: dict[str, int] = extra[key]
            if key != "word_to_index":
                logger.warning("Checkpoint predates the vocabulary fix; ids may not index its rows")
            logger.info("Loaded vocabulary with %d words", len(vocabulary))
            return vocabulary
    raise KeyError("checkpoint carries no vocabulary")
