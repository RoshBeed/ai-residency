"""Positional encodings.

The encoder uses the fixed sinusoidal encoding from the original Transformer paper;
the decoder uses a learned one, which is what the seq2seq model was trained with.
"""

import math

import torch
from torch import nn

__all__ = ["LearnablePositionalEncoding", "PositionalEncoding"]


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding, added to the input."""

    pe: torch.Tensor

    def __init__(self, d_model: int, max_seq_length: int) -> None:
        """Precompute the encoding table.

        Args:
            d_model: Model width.
            max_seq_length: Longest sequence to support.
        """
        super().__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional information.

        Args:
            x: Shape `(batch, seq_len, d_model)`.

        Returns:
            Same shape as `x`.
        """
        return x + self.pe[:, : x.size(1)]


class LearnablePositionalEncoding(nn.Module):
    """Learned positional embedding, added to the input."""

    def __init__(self, d_model: int, max_seq_length: int) -> None:
        """Create the embedding table.

        Args:
            d_model: Model width.
            max_seq_length: Longest sequence to support.
        """
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_length, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional information.

        Args:
            x: Shape `(batch, seq_len, d_model)`.

        Returns:
            Same shape as `x`.
        """
        return x + self.pos_embedding[:, : x.size(1), :]
