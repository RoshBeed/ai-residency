"""Encoder-only Vision Transformer for single-digit MNIST classification."""

import torch
from torch import nn

from config import ClassifierArch
from vision_transformer.encoder import Encoder
from vision_transformer.patch_embedding import PatchEmbedding
from vision_transformer.positional import PositionalEncoding

__all__ = ["VisionTransformer"]


class VisionTransformer(nn.Module):
    """ViT that classifies one 28x28 MNIST digit.

    A learned [CLS] token is prepended to the patch sequence and its final hidden
    state is what the classification head reads, following the original ViT recipe.
    """

    def __init__(self, arch: ClassifierArch) -> None:
        """Build the model.

        Args:
            arch: Architecture the weights are shaped by.
        """
        super().__init__()
        self.arch = arch
        self.n_patches = (arch.img_size // arch.patch_size) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, arch.d_model))

        self.patch_embedding = PatchEmbedding(
            img_size=arch.img_size,
            patch_size=arch.patch_size,
            in_channels=arch.in_channels,
            embed_dim=arch.d_model,
        )

        # The +1 is the [CLS] token, which occupies a position like any patch does.
        self.encoder_pos_encoding = PositionalEncoding(
            d_model=arch.d_model,
            max_seq_length=self.n_patches + 1,
        )

        self.encoder_layers = nn.ModuleList(
            [
                Encoder(
                    heads=arch.heads,
                    d_model=arch.d_model,
                    head_dim=arch.head_dim,
                    ff_hidden_dim=arch.ff_hidden_dim,
                    dropout=arch.dropout,
                )
                for _ in range(arch.num_encoder_layers)
            ]
        )

        self.layer_norm = nn.LayerNorm(arch.d_model)
        self.dropout = nn.Dropout(arch.dropout)
        self.classifier = nn.Linear(arch.d_model, arch.num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise the parameters the default schemes handle badly."""
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.patch_embedding.proj.weight, std=0.02)
        # Conv2d's bias is optional by construction, so narrow before touching it.
        patch_bias = self.patch_embedding.proj.bias
        if patch_bias is not None:
            nn.init.zeros_(patch_bias)
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify a batch of images.

        Args:
            x: Shape `(batch, in_channels, img_size, img_size)`.

        Returns:
            Class logits of shape `(batch, num_classes)`.

        Raises:
            ValueError: If the input is not 4D, or its size or channel count does not
                match the architecture.
        """
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input tensor, got {x.dim()}D")
        if x.size(-2) != self.arch.img_size or x.size(-1) != self.arch.img_size:
            raise ValueError(
                f"Expected input size {self.arch.img_size}x{self.arch.img_size}, "
                f"got {x.size(-2)}x{x.size(-1)}"
            )
        if x.size(1) != self.arch.in_channels:
            raise ValueError(f"Expected {self.arch.in_channels} channels, got {x.size(1)}")

        batch_size = x.size(0)

        x = self.patch_embedding(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.encoder_pos_encoding(x)

        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)

        x = self.layer_norm(x)
        cls_output = self.dropout(x[:, 0])
        logits: torch.Tensor = self.classifier(cls_output)
        return logits
