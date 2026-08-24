"""Run configuration for the multi-digit seq2seq model.

The architecture is a frozen dataclass rather than a dict so a wrong key is a type
error, not a `KeyError` at epoch 3. The checkpoint file name is derived from the
architecture itself, so the trainer and the API cannot disagree about which file a
given architecture writes.
"""

from dataclasses import dataclass, field
from pathlib import Path

from ai_residency_common import BaseConfig

__all__ = ["EOS_TOKEN", "START_TOKEN", "VOCAB_SIZE", "Config", "Seq2SeqArch"]

# The vocabulary is the ten digits plus two control tokens.
START_TOKEN = 10
EOS_TOKEN = 11
VOCAB_SIZE = 12


@dataclass(frozen=True)
class Seq2SeqArch:
    """Architecture of the multi-digit encoder-decoder model.

    Attributes:
        img_size: Side length of the square composite image.
        patch_size: Side length of each square patch.
        in_channels: Input channels; MNIST is grayscale.
        d_model: Model width.
        num_encoder_layers: Stacked encoder blocks.
        num_decoder_layers: Stacked decoder blocks.
        heads: Attention heads per block.
        head_dim: Width of each attention head.
        ff_hidden_dim: Hidden width of the feed-forward block.
        dropout: Dropout probability.
        vocab_size: Digits plus START and EOS.
        start_token: Token the decoder is primed with.
        eos_token: Token that ends a generated sequence.
        max_seq_length: Longest sequence the decoder handles, START and EOS included.
    """

    img_size: int = 84
    patch_size: int = 16
    in_channels: int = 1
    d_model: int = 256
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    heads: int = 8
    head_dim: int = 32
    ff_hidden_dim: int = 1024
    dropout: float = 0.1
    vocab_size: int = VOCAB_SIZE
    start_token: int = START_TOKEN
    eos_token: int = EOS_TOKEN
    max_seq_length: int = 5

    @property
    def checkpoint_name(self) -> str:
        """File name the checkpoint for this architecture is stored under."""
        return (
            f"vit_seq2seq_{self.img_size}x{self.img_size}"
            f"_patch{self.patch_size}"
            f"_d{self.d_model}"
            f"_enc{self.num_encoder_layers}"
            f"_dec{self.num_decoder_layers}"
            f"_heads{self.heads}"
            f"_best.pth"
        )


@dataclass
class Config(BaseConfig):
    """Configuration for training and serving the multi-digit seq2seq model.

    Attributes:
        batch_size: Composite images per optimizer step.
        learning_rate: AdamW learning rate.
        seq2seq_arch: The architecture, which also names the checkpoint.
        seq2seq_weight_decay: AdamW weight decay.
        seq2seq_gradient_clip: Gradient-norm clip.
        gradient_accumulation_steps: Steps accumulated before an optimizer step.
        warmup_steps: Linear warmup steps for the learning-rate schedule.
        label_smoothing: Label smoothing applied to the decoder's cross-entropy.
        train_split: Fraction of composites used for training.
        max_digits: Digits per composite image.
        samples_per_class: MNIST digits sampled per class when building composites.
        num_workers: DataLoader worker processes.
        api_host: Interface the FastAPI service binds to.
        api_port: Port the FastAPI service binds to.
    """

    batch_size: int = 64
    learning_rate: float = 1e-4
    num_epochs: int = 15

    seq2seq_arch: Seq2SeqArch = field(default_factory=Seq2SeqArch)

    seq2seq_weight_decay: float = 0.01
    seq2seq_gradient_clip: float = 1.0

    gradient_accumulation_steps: int = 2
    warmup_steps: int = 4000
    label_smoothing: float = 0.1

    train_split: float = 0.8
    max_digits: int = 3
    samples_per_class: int = 100
    num_workers: int = 4

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    hf_repo_id: str | None = "roshbeed/ai-residency-vit-seq2seq"
    wandb_project: str | None = "ai-residency-vit-seq2seq"

    @property
    def seq2seq_checkpoint_path(self) -> Path:
        """Full path to the best multi-digit seq2seq checkpoint."""
        return self.checkpoint_dir / self.seq2seq_arch.checkpoint_name

    @property
    def decoder_seq_length(self) -> int:
        """Sequence length the decoder works with: the digits plus START and EOS."""
        return self.max_digits + 2
