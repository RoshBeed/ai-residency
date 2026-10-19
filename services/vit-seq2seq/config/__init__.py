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

    # 112, not 84, because the patch embedding is a strided conv: at patch 16 an 84px
    # canvas yields a 5x5 grid covering only 80px, and the strided conv silently drops
    # columns and rows 80-83. The third digit sat at columns 56-83, so 14% of its width
    # was never seen by the model -- a permanent handicap on the last digit. 112 is 7x16,
    # so every pixel lands inside a patch.
    img_size: int = 112
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
        max_digits: Most digits per composite image.
        min_digits: Fewest digits per composite image.
        digit_size: Side length each MNIST digit is pasted at.
        augment_probability: Share of digits given a random affine warp.
        augment_rotation_degrees: Rotation range, plus or minus.
        augment_shear_degrees: Shear range, plus or minus.
        augment_scale_range: Scale range as (low, high).
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
    # Shortest composite. 3 reproduces the original fixed-length task, where every label
    # was `str(number).zfill(3)` and so EOS always landed at the same position -- the
    # decoder learned "emit three, then stop" and could never answer with one or two
    # digits, while the write-up advertised per-sequence EOS. 1 makes the stop a decision.
    min_digits: int = 1
    # Explicit rather than `img_size // max_digits`, which forced
    # `spacing = (img_size - max_digits * digit_size) // (max_digits + 1)` to 0 and so
    # made `max_jitter = spacing // 2` zero too -- the positional jitter the dataset
    # advertised had never moved anything. 28 is MNIST's native size, so digits are
    # pasted without resampling and the slack becomes real: at 112px, spacing is 7.
    digit_size: int = 28
    # Augmentation applied to every digit rather than 30% of them, and wide enough that
    # the three evenly-spaced cells of the original canvas stop being memorisable.
    augment_probability: float = 1.0
    augment_rotation_degrees: float = 15.0
    augment_shear_degrees: float = 12.0
    augment_scale_range: tuple[float, float] = (0.75, 1.15)
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
