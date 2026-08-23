"""Image-to-patch-sequence embedding."""

import torch
from torch import nn

__all__ = ["PatchEmbedding"]


class PatchEmbedding(nn.Module):
    """Split an image into non-overlapping patches and project each one.

    The split is done with a strided convolution rather than an explicit reshape: a
    `kernel_size == stride == patch_size` convolution is exactly a per-patch linear
    projection, and it is faster.

    Attributes:
        n_patches: Number of patches an image is divided into.
    """

    def __init__(self, img_size: int, patch_size: int, in_channels: int, embed_dim: int) -> None:
        """Build the projection.

        Args:
            img_size: Side length of the square input image.
            patch_size: Side length of each square patch.
            in_channels: Input channels.
            embed_dim: Width each patch is projected to.
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed an image batch as a patch sequence.

        Args:
            x: Shape `(batch, in_channels, img_size, img_size)`.

        Returns:
            Shape `(batch, n_patches, embed_dim)`.
        """
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        output: torch.Tensor = self.norm(x)
        return output
