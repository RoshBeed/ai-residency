"""CBOW model definitions."""

import torch
from torch import nn

__all__ = ["CBOW", "Regressor"]


class CBOW(nn.Module):
    """Continuous bag-of-words: predict a word from the average of its context.

    Only `emb` matters downstream. The projection back out to vocabulary size exists
    to produce a loss and is discarded once the embedding matrix is what the towers
    want.

    Attributes:
        emb: Embedding matrix of shape `(vocab_size, embedding_dim)`.
        ffw: Projection from an averaged context vector back to vocabulary logits.
    """

    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        """Build the model.

        Args:
            vocab_size: Number of tokens in the vocabulary.
            embedding_dim: Width of each word vector.
        """
        super().__init__()
        self.emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.ffw = nn.Linear(in_features=embedding_dim, out_features=vocab_size, bias=False)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Predict the centre word from its context.

        Args:
            context: Context token ids of shape `(batch_size, context_length)`.

        Returns:
            Vocabulary logits of shape `(batch_size, vocab_size)`.
        """
        embeddings = self.emb(context)
        pooled = embeddings.mean(dim=1)
        logits: torch.Tensor = self.ffw(pooled)
        return logits


class Regressor(nn.Module):
    """Small MLP mapping a pooled embedding to a single score.

    Carried over from Week 1's Hacker News score prediction. The retrieval pipeline
    does not use it; `train_regressor` is the only caller.

    Attributes:
        seq: The layer stack applied in `forward`.
    """

    def __init__(self, embedding_dim: int = 128) -> None:
        """Build the network.

        Args:
            embedding_dim: Width of the pooled embedding fed in.
        """
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=1),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Score a batch of pooled embeddings.

        Args:
            embeddings: Shape `(batch_size, embedding_dim)`.

        Returns:
            Scores of shape `(batch_size, 1)`.
        """
        scores: torch.Tensor = self.seq(embeddings)
        return scores
