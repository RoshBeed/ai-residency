"""Transformer building blocks, written from scratch rather than taken from `torch.nn`.

Composing these by hand is the point: `nn.TransformerEncoderLayer` would
have hidden the attention maths this repo exists to demonstrate.
"""

from vision_transformer.attention import MultiHeadAttention
from vision_transformer.decoder import Decoder
from vision_transformer.encoder import Encoder
from vision_transformer.feedforward import FeedForward
from vision_transformer.patch_embedding import PatchEmbedding
from vision_transformer.positional import (
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
