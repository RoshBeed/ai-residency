"""Scaled dot-product multi-head attention."""

import math

import torch
from torch import nn

__all__ = ["MultiHeadAttention"]


class MultiHeadAttention(nn.Module):
    """Multi-head attention over a batch of token sequences.

    Used for encoder self-attention, decoder masked self-attention and decoder cross
    attention, which is why `key` and `value` default to `query`: passing one tensor
    is self-attention, passing three is cross-attention.

    Attributes:
        heads: Number of parallel attention heads.
        d_model: Model width, split evenly across heads.
        head_dim: Width of a single head.
    """

    def __init__(self, heads: int, d_model: int, head_dim: int, dropout: float = 0.1) -> None:
        """Build the four projections attention needs.

        Args:
            heads: Number of attention heads.
            d_model: Model width; must divide evenly by `heads`.
            head_dim: Width of each head.
            dropout: Dropout applied to the attention probabilities.

        Raises:
            ValueError: If `d_model` is not divisible by `heads`.
        """
        super().__init__()
        if d_model % heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by heads ({heads})")

        self.heads = heads
        self.d_model = d_model
        self.head_dim = head_dim

        # Uppercase to match the paper's notation, and load-bearing: these names are
        # the state-dict keys the committed checkpoints were saved with, so renaming
        # them to snake_case would make every existing checkpoint unloadable.
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Attend `query` over `key`/`value`.

        Args:
            query: Shape `(batch, query_len, d_model)`.
            key: Shape `(batch, key_len, d_model)`. Defaults to `query`.
            value: Shape `(batch, key_len, d_model)`. Defaults to `query`.
            mask: Boolean mask where True means "do not attend", or a numeric mask
                where 0 means the same.

        Returns:
            Shape `(batch, query_len, d_model)`.
        """
        if key is None:
            key = query
        if value is None:
            value = query

        batch_size, q_len, _ = query.size()
        k_len = key.size(1)

        q = self.W_q(query).view(batch_size, q_len, self.heads, self.head_dim).transpose(1, 2)
        k = self.W_k(key).view(batch_size, k_len, self.heads, self.head_dim).transpose(1, 2)
        v = self.W_v(value).view(batch_size, k_len, self.heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # Softmax overflows to NaN on the extreme logits this model produced early in
        # training before the clamp was added.
        attn_scores = torch.clamp(attn_scores, min=-100, max=100)

        if mask is not None:
            if mask.dtype == torch.bool:
                attn_scores = attn_scores.masked_fill(mask, float("-inf"))
            else:
                attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        attn_probs = self.dropout(torch.softmax(attn_scores, dim=-1))

        out = torch.matmul(attn_probs, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, q_len, self.d_model)
        projected: torch.Tensor = self.W_o(out)
        return projected
