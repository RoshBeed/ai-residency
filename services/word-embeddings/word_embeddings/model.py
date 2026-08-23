"""The word-embedding model: CBOW trained with negative sampling.

This is the file to read to understand word2vec. ``CBOWNS.forward`` is the whole
algorithm: average the context word vectors, then score the target word against a
handful of randomly sampled negative words with a logsigmoid loss.

``CBOWNS`` is kept as a ``PreTrainedModel`` so the trained vectors can be published
and loaded through the usual Hugging Face machinery rather than as a bare state dict.
``load_cbow_embeddings`` is how the rest of the repo reads those vectors back out of
a saved checkpoint.
"""

import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from ai_residency_common import get_logger
from transformers import PretrainedConfig, PreTrainedModel

__all__ = ["CBOWNS", "CBOWConfig", "load_cbow_embeddings"]

logger = get_logger(__name__)


class CBOWConfig(PretrainedConfig):
    """Hyperparameters for the CBOW model."""

    model_type = "cbow"

    def __init__(
        self,
        vocab_size: int = 10000,
        embed_size: int = 100,
        window_size: int = 4,
        num_negatives: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.window_size = window_size
        self.num_negatives = num_negatives


class CBOWNS(PreTrainedModel):
    """Continuous bag-of-words trained with negative sampling (Mikolov et al., 2013).

    Two embedding matrices: ``in_embed`` for context words, ``out_embed`` for
    targets and negatives. A forward pass averages the context vectors, dots
    them with the target vector (positive) and each negative vector, and returns
    the summed negative-sampling loss.
    """

    config_class = CBOWConfig

    def __init__(self, config: CBOWConfig) -> None:
        super().__init__(config)
        self.in_embed = nn.Embedding(config.vocab_size, config.embed_size)
        self.out_embed = nn.Embedding(config.vocab_size, config.embed_size)

    def forward(
        self,
        context_idxs: torch.Tensor,
        target_idx: torch.Tensor,
        negative_idxs: torch.Tensor,
    ) -> torch.Tensor:
        v_c = self.in_embed(context_idxs)  # (B, C, E)
        v_c = torch.mean(v_c, dim=1)  # (B, E)
        u_o = self.out_embed(target_idx)  # (B, E)
        u_k = self.out_embed(negative_idxs)  # (B, N, E)

        # Positive loss
        pos = torch.sum(v_c * u_o, dim=1)
        pos_loss = -F.logsigmoid(pos)

        # Negative loss
        neg = torch.bmm(u_k, v_c.unsqueeze(2)).squeeze()  # (B, N)
        neg_loss = -torch.sum(F.logsigmoid(-neg), dim=1)  # Sum over negative samples

        return (pos_loss + neg_loss).mean()

    def save_pretrained(self, save_directory: str | os.PathLike[Any], **kwargs: Any) -> None:
        # Save model weights
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))

        # Save config
        self.config.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str | Path, *model_args: Any, **kwargs: Any
    ) -> "CBOWNS":
        config = kwargs.pop("config", None)
        if config is None:
            config = cls.config_class.from_pretrained(pretrained_model_name_or_path, **kwargs)

        model = cls(config)
        model.load_state_dict(
            torch.load(os.path.join(pretrained_model_name_or_path, "pytorch_model.bin"))
        )
        return model


def load_cbow_embeddings(model_path: str | Path) -> torch.Tensor:
    """Load the trained input-embedding matrix from a CBOW checkpoint.

    Args:
        model_path: Path to the checkpoint saved by `train_cbow`.

    Returns:
        The `in_embed` weight matrix, shape `(vocab_size, embed_size)`, on CPU.
    """
    logger.info("Loading CBOW embeddings from %s", model_path)
    # map_location='cpu': checkpoints trained on MPS carry mps:0 tensor metadata and
    # fail to deserialise on a CPU-only host unless loaded onto the CPU explicitly.
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    embeddings: torch.Tensor = checkpoint["model_state_dict"]["in_embed.weight"]
    embeddings = embeddings.cpu()
    logger.info("Loaded embeddings with shape %s", tuple(embeddings.shape))
    return embeddings
