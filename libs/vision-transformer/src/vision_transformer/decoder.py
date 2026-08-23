"""A single post-norm transformer decoder block."""

import torch
from torch import nn

from vision_transformer.attention import MultiHeadAttention

__all__ = ["Decoder"]


class Decoder(nn.Module):
    """Masked self-attention, cross-attention over the encoder, then feed-forward."""

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
        self.self_attention = MultiHeadAttention(heads, d_model, head_dim, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.cross_attention = MultiHeadAttention(heads, d_model, head_dim, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, ff_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        src_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run one decoder block.

        Args:
            x: Decoder input, shape `(batch, tgt_len, d_model)`.
            encoder_output: Encoder memory, shape `(batch, src_len, d_model)`.
            tgt_mask: Causal mask over the decoder's own positions.
            src_mask: Mask over the encoder positions.

        Returns:
            Shape `(batch, tgt_len, d_model)`.
        """
        x = self.norm1(x + self.dropout1(self.self_attention(x, mask=tgt_mask)))
        cross = self.cross_attention(x, encoder_output, encoder_output, mask=src_mask)
        x = self.norm2(x + self.dropout2(cross))
        output: torch.Tensor = self.norm3(x + self.dropout3(self.feed_forward(x)))
        return output
