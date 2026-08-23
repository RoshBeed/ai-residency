"""Data for the content-score predictor.

Loads the labelled posts and turns each title into one averaged word vector over the
frozen CBOW embeddings produced by the sibling word-embeddings repo. ``TitleScoreDataset``
pairs those title vectors with their scores for training.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ai_residency_common import get_logger
from torch.utils.data import Dataset

__all__ = ["TitleScoreDataset", "create_title_embeddings"]

logger = get_logger(__name__)


class TitleScoreDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Title embeddings and their scores.

    Attributes:
        embeddings: Title embeddings, shape `(num_posts, embedding_dim)`.
        scores: Score per post, shape `(num_posts,)`.
        posts: The original post records, kept so training can print real titles
            alongside sample predictions.
    """

    def __init__(
        self,
        embeddings: np.ndarray[Any, Any] | torch.Tensor,
        scores: np.ndarray[Any, Any] | torch.Tensor,
        posts: list[dict[str, Any]] | None = None,
    ) -> None:
        """Build the dataset.

        Args:
            embeddings: One embedding per post.
            scores: One score per post.
            posts: Original post records, used only for display.
        """
        self.embeddings = torch.as_tensor(embeddings, dtype=torch.float32)
        if self.embeddings.ndim == 1:
            self.embeddings = self.embeddings.unsqueeze(0)
        self.scores = torch.as_tensor(scores, dtype=torch.float32)
        self.posts = posts or []
        logger.info("Created dataset with embeddings shape %s", tuple(self.embeddings.shape))

    def __len__(self) -> int:
        """Return the number of posts."""
        return len(self.embeddings)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the embedding and score at `index`."""
        return self.embeddings[index], self.scores[index]


def create_title_embeddings(
    data_path: str | Path,
    word_to_ix: dict[str, int],
    embeddings: torch.Tensor,
    embedding_dim: int = 128,
    max_samples: int | None = None,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Pre-compute an averaged embedding for every post title.

    Args:
        data_path: Path to the cleaned posts JSON (each has `Title` and `score`).
        word_to_ix: The CBOW vocabulary, word to token id.
        embeddings: The trained CBOW input-embedding matrix.
        embedding_dim: Width of a single embedding (defaulted, overridden below).
        max_samples: Optional cap on the number of posts to process.

    Returns:
        `(title_embeddings, scores)` — one averaged vector and one score per post.
    """
    embedding_dim = embeddings.shape[1]  # override the default 128

    logger.info(
        "Creating title embeddings (max_samples=%s)",
        max_samples if max_samples else "all",
    )
    with open(data_path) as handle:
        data = json.load(handle)
    logger.info("Loaded %d total posts", len(data))

    if max_samples:
        data = data[:max_samples]

    title_embeddings: list[np.ndarray[Any, Any]] = []
    scores: list[float] = []

    for i, item in enumerate(data):
        if i % 1000 == 0:
            logger.info("Processing post %d/%d", i + 1, len(data))

        title = item["Title"].lower().split()
        raw_score = item.get("score")
        if raw_score is None:
            raw_score = item.get("Score", 0)
        score = float(raw_score)

        word_embeddings = [
            embeddings[word_to_ix[word]].cpu().numpy() for word in title if word in word_to_ix
        ]

        # A title with no known words becomes the zero vector.
        title_embedding = (
            np.mean(word_embeddings, axis=0) if word_embeddings else np.zeros(embedding_dim)
        )

        title_embeddings.append(title_embedding)
        scores.append(score)

    logger.info("Created %d title embeddings", len(title_embeddings))
    return np.array(title_embeddings), np.array(scores)
