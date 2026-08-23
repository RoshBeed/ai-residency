"""Transformer building blocks, written from scratch rather than taken from `torch.nn`.

Composing these by hand is the point of the week: `nn.TransformerEncoderLayer` would
have hidden the attention maths this repo exists to demonstrate.
"""

from transformer.attention import MultiHeadAttention
from transformer.decoder import Decoder
from transformer.encoder import Encoder
from transformer.feedforward import FeedForward
from transformer.patch_embedding import PatchEmbedding
from transformer.positional import (
    LearnablePositionalEncoding,
    PositionalEncoding,
)

__all__ = [
    "Decoder",
    "Encoder",
    "FeedForward",
    "LearnablePositionalEncoding",
    "MultiHeadAttention",
    "PatchEmbedding",
    "PositionalEncoding",
]
