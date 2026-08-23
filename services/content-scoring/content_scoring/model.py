"""The score-prediction model: a small MLP over frozen title embeddings."""

import torch
from torch import nn

__all__ = ["ScorePredictor"]


class ScorePredictor(nn.Module):
    """A small MLP predicting an engagement score from a title embedding.

    The input is a title's averaged word vector, produced from the frozen CBOW
    embeddings trained in the sibling word-embeddings repo; the output is an
    estimated upvote score.
    """

    def __init__(self, input_dim: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        elif len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        output: torch.Tensor = self.network(x)
        return output.squeeze()
