"""A single post-norm transformer encoder block."""

import torch
from torch import nn

from vision_transformer.attention import MultiHeadAttention

__all__ = ["Encoder"]


class Encoder(nn.Module):
    """Self-attention followed by a feed-forward block, each with a residual.

    Normalisation comes after the residual add, matching the original Transformer
    paper rather than the pre-norm arrangement ViT uses.
    """

    def __init__(
        self,
        heads: int,
        d_model: int,
        head_dim: int,
        ff_hidden_dim: int,
        dropout: float = 0.1,
    ) -> None:
        """Build the block.

        Args:
            heads: Attention heads.
            d_model: Model width.
            head_dim: Width of each attention head.
            ff_hidden_dim: Hidden width of the feed-forward block.
            dropout: Dropout probability.
        """
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(heads, d_model, head_dim, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, ff_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run one encoder block.

        Args:
            x: Shape `(batch, seq_len, d_model)`.

        Returns:
            Shape `(batch, seq_len, d_model)`.
        """
        x = self.norm1(x + self.dropout1(self.multi_head_attention(x)))
        output: torch.Tensor = self.norm2(x + self.dropout2(self.feed_forward(x)))
        return output
