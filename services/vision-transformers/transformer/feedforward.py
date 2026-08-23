"""Position-wise feed-forward block."""

import torch
from torch import nn

__all__ = ["FeedForward"]


class FeedForward(nn.Module):
    """Two linear layers with a ReLU between them, applied to each position."""

    def __init__(self, d_model: int, ff_hidden_dim: int, dropout: float = 0.1) -> None:
        """Build the block.

        Args:
            d_model: Input and output width.
            ff_hidden_dim: Hidden width.
            dropout: Dropout probability.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, ff_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block.

        Args:
            x: Shape `(batch, seq_len, d_model)`.

        Returns:
            Shape `(batch, seq_len, d_model)`.
        """
        output: torch.Tensor = self.net(x)
        return output
