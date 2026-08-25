"""Nearest-neighbour probe used during CBOW training.

Cheap qualitative check that the embedding matrix is going somewhere: take one word,
find the words closest to it by cosine similarity, and print them. It is not a metric
and nothing selects a checkpoint on it — it is there so a long training run shows
something more informative than a falling loss.
"""

import torch

from ai_residency_common import get_logger
from word2vec import Word2Vec
from word2vec import nearest_words as nearest_in_matrix

__all__ = ["nearest_words"]

logger = get_logger(__name__)


def nearest_words(
    model: Word2Vec,
    word: str,
    words_to_ids: dict[str, int],
    ids_to_words: dict[int, str],
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Return the words whose embeddings are closest to `word`'s.

    Args:
        model: Model whose embedding matrix is probed.
        word: Word to look up. Missing words return an empty list rather than raising,
            because this runs inside a training loop.
        words_to_ids: Vocabulary mapping word to token id.
        ids_to_words: Vocabulary mapping token id to word.
        top_k: How many neighbours to return, excluding the word itself.

    Returns:
        `(word, cosine_similarity)` pairs, most similar first.
    """
    if word not in words_to_ids:
        logger.warning("%r is not in the vocabulary; skipping the neighbour probe", word)
        return []

    with torch.no_grad():
        weights = model.embeddings.detach()
        return nearest_in_matrix(
            weights, weights[words_to_ids[word]], ids_to_words, k=top_k, exclude=word
        )
