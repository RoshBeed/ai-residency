"""Run configuration.

Two models live in this repo and they are not variants of each other: a single-digit
encoder-only classifier on 28x28 MNIST images, and an encoder-decoder seq2seq model
reading 84x84 composites of three digits. They have different input sizes, different
patch sizes and different widths, so their architectures are separate frozen
dataclasses rather than one shared block with conditionals.

Everything here used to be loose module-level dicts (`MODEL_CONFIG`, `TRAINING_CONFIG`,
`SEQ2SEQ_CONFIG`, `MODEL_PATHS`, ...) built at import time, including a device probe
that printed to stdout as a side effect of importing the module. Dataclasses replace
them, so a wrong key is a type error rather than a `KeyError` at epoch 3.

Checkpoint file names encode the architecture that produced them
(`vit_cls_28x28_patch4_d128_layers6_heads8_best.pth`). That convention is kept and is
now derived from the architecture config itself, so a trainer and the API cannot
disagree about which file a given architecture writes to or reads from.
"""

from dataclasses import dataclass, field
from pathlib import Path

from ai_residency_common import BaseConfig

__all__ = [
    "EOS_TOKEN",
    "START_TOKEN",
    "VOCAB_SIZE",
    "ClassifierArch",
    "Config",
    "Seq2SeqArch",
]

# The seq2seq vocabulary is the ten digits plus two control tokens.
START_TOKEN = 10
EOS_TOKEN = 11
VOCAB_SIZE = 12


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
    """Configuration for both training runs and for serving.

    The batch size and learning rate used to be chosen per detected device, so the
    same run on this Mac and in a container trained measurably different models. They
    are fixed here instead, at the values the committed checkpoints were trained with.

    Attributes:
        classifier_arch: Architecture of the single-digit classifier.
        seq2seq_arch: Architecture of the multi-digit seq2seq model.
        classifier_epochs: Epoch cap for the classifier; early stopping usually ends
            the run first.
        classifier_weight_decay: AdamW weight decay for the classifier.
        classifier_patience: Epochs without improvement before the classifier stops.
        classifier_gradient_clip: Gradient-norm clip for the classifier.
        seq2seq_weight_decay: AdamW weight decay for the seq2seq model.
        seq2seq_gradient_clip: Gradient-norm clip for the seq2seq model.
        gradient_accumulation_steps: An optimizer step is taken every N batches.
        warmup_steps: Linear learning-rate warmup before cosine annealing.
        label_smoothing: Label smoothing applied to the seq2seq loss.
        train_split: Fraction of the MNIST training set kept for training.
        max_digits: Digits per composite image.
        samples_per_class: Composites generated per distinct number.
        num_workers: DataLoader worker processes.
        api_host: Interface the FastAPI service binds to.
        api_port: Port the FastAPI service listens on.
    """

    batch_size: int = 64
    learning_rate: float = 1e-4
    num_epochs: int = 15

    classifier_arch: ClassifierArch = field(default_factory=ClassifierArch)
    seq2seq_arch: Seq2SeqArch = field(default_factory=Seq2SeqArch)

    classifier_epochs: int = 100
    classifier_weight_decay: float = 1e-4
    classifier_patience: int = 15
    classifier_gradient_clip: float = 1.0

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

    hf_repo_id: str | None = "roshbeed/ai-residency-week3-vision-transformers"
    wandb_project: str | None = "ai-residency-week3-vision-transformers"

    @property
    def classifier_checkpoint_path(self) -> Path:
        """Full path to the best single-digit classifier checkpoint."""
        return self.checkpoint_dir / self.classifier_arch.checkpoint_name

    @property
    def seq2seq_checkpoint_path(self) -> Path:
        """Full path to the best multi-digit seq2seq checkpoint."""
        return self.checkpoint_dir / self.seq2seq_arch.checkpoint_name

    @property
    def decoder_seq_length(self) -> int:
        """Sequence length the decoder works with: the digits plus START and EOS."""
        return self.max_digits + 2
