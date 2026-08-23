"""Run configuration for the single-digit classifier.

The architecture is a frozen dataclass rather than a dict so a wrong key is a type
error, not a `KeyError` at epoch 3. The checkpoint file name is derived from the
architecture itself (`vit_cls_28x28_patch4_d128_layers6_heads8_best.pth`), so the
trainer and the API cannot disagree about which file a given architecture writes.
"""

from dataclasses import dataclass, field
from pathlib import Path

from ai_residency_common import BaseConfig

__all__ = ["ClassifierArch", "Config"]


@dataclass(frozen=True)
class ClassifierArch:
    """Architecture of the single-digit encoder-only classifier.

    Attributes:
        img_size: Side length of the square input image.
        patch_size: Side length of each square patch.
        in_channels: Input channels; MNIST is grayscale.
        num_classes: Output classes, one per digit.
        d_model: Model width.
        num_encoder_layers: Stacked encoder blocks.
        heads: Attention heads per block.
        head_dim: Width of each attention head.
        ff_hidden_dim: Hidden width of the feed-forward block.
        dropout: Dropout probability.
    """

    img_size: int = 28
    patch_size: int = 4
    in_channels: int = 1
    num_classes: int = 10
    d_model: int = 128
    num_encoder_layers: int = 6
    heads: int = 8
    head_dim: int = 16
    ff_hidden_dim: int = 512
    dropout: float = 0.1

    @property
    def checkpoint_name(self) -> str:
        """File name the checkpoint for this architecture is stored under."""
        return (
            f"vit_cls_{self.img_size}x{self.img_size}"
            f"_patch{self.patch_size}"
            f"_d{self.d_model}"
            f"_layers{self.num_encoder_layers}"
            f"_heads{self.heads}"
            f"_best.pth"
        )


@dataclass
class Config(BaseConfig):
    """Configuration for training and serving the single-digit classifier.

    Attributes:
        batch_size: Images per optimizer step.
        learning_rate: AdamW learning rate.
        classifier_arch: The architecture, which also names the checkpoint.
        classifier_epochs: Epoch cap; early stopping usually ends the run first.
        classifier_weight_decay: AdamW weight decay.
        classifier_patience: Epochs without improvement before stopping.
        classifier_gradient_clip: Gradient-norm clip.
        train_split: Fraction of MNIST used for training; the rest is validation.
        num_workers: DataLoader worker processes.
        api_host: Interface the FastAPI service binds to.
        api_port: Port the FastAPI service binds to.
    """

    batch_size: int = 64
    learning_rate: float = 1e-4
    num_epochs: int = 15

    classifier_arch: ClassifierArch = field(default_factory=ClassifierArch)

    classifier_epochs: int = 100
    classifier_weight_decay: float = 1e-4
    classifier_patience: int = 15
    classifier_gradient_clip: float = 1.0

    train_split: float = 0.8
    num_workers: int = 4

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    hf_repo_id: str | None = "roshbeed/ai-residency-week3-vision-transformers"
    wandb_project: str | None = "ai-residency-vit-classifier"

    @property
    def classifier_checkpoint_path(self) -> Path:
        """Full path to the best single-digit classifier checkpoint."""
        return self.checkpoint_dir / self.classifier_arch.checkpoint_name
