"""Run configuration.

This repo had no configuration layer at all before the standardization pass: model
names, checkpoint file names, the prompt, the sample size and the learning rate were
literals scattered across three flat modules and eleven notebook cells, and the same
value appeared with different numbers in different places. Everything lives here now,
so a training run and the service that loads its output cannot disagree.

`CheckpointInfo` is the part worth reading. Each shipped adapter checkpoint records
the architecture it was trained with, because *not* recording that is precisely what
produced this repo's longest-standing bug: `adapter_flickr_batched.pth` was saved from
an adapter with a trailing `LayerNorm` that was later deleted from the code, and every
later attempt to load it raised a `state_dict` key mismatch. `models.build_adapter`
reads the recorded architecture back, so a checkpoint names the shape it needs rather
than relying on the code not having moved since.
"""

from dataclasses import dataclass, field
from pathlib import Path

from ai_residency_common import BaseConfig

__all__ = ["CHECKPOINTS", "DEFAULT_CHECKPOINT", "CheckpointInfo", "Config"]


@dataclass(frozen=True)
class CheckpointInfo:
    """A shipped adapter checkpoint and the architecture that produced it.

    Attributes:
        description: How the checkpoint was trained, as shown in the UI and the
            README's results table.
        final_layer_norm: Whether the adapter that wrote it ended in a `LayerNorm`.
            Only `adapter_flickr_batched.pth` did.
    """

    description: str
    final_layer_norm: bool


# The three checkpoints, hosted on the Hub (gitignored locally). Listed rather than
# globbed so the UI can describe each one honestly instead of showing bare file names.
# The three `adapter*.pth` below all predate the patch-grid change: they were trained
# when the image reached the decoder as one pooled soft token, so the inference path now
# feeds them 49 tokens where they only ever saw one. They still load -- the adapter is
# per-token and its shape is unchanged -- but that is shape compatibility, not the input
# distribution they fit. They are kept because Section 6 reports their captions and
# because they are what the single-token result was measured on; they are not the
# default any more.
CHECKPOINTS: dict[str, CheckpointInfo] = {
    "adapter_patches.pth": CheckpointInfo(
        description="Flickr30k, 49 patch tokens per image (current architecture)",
        final_layer_norm=True,
    ),
    "adapter_flickr_batched.pth": CheckpointInfo(
        description="Flickr30k, 100 pairs / 3 epochs, batched (best-trained)",
        final_layer_norm=True,
    ),
    "adapter.pth": CheckpointInfo(
        # Labelled "untrained / freshly initialized" everywhere in this repo until the
        # standardization pass checked. Its LayerNorm weights sit 4.9e-05 off 1.0, so it
        # did train: 10 Flickr30k pairs, 3 epochs, unbatched, at 1e-5. That is 30 steps,
        # roughly a tenth of the movement in the other two, which is why it behaves like
        # an untrained adapter without being one.
        description="Flickr30k, 10 pairs / 3 epochs, unbatched at 1e-5 (barely moved off init)",
        final_layer_norm=False,
    ),
    "adapter_trained.pth": CheckpointInfo(
        description="10 local images / 20 epochs (overfit — degenerates into repeated tokens)",
        final_layer_norm=False,
    ),
}

DEFAULT_CHECKPOINT = "adapter_patches.pth"


@dataclass
class Config(BaseConfig):
    """Configuration for training the adapter, captioning, and zero-shot classification.

    The defaults are the values `adapter_flickr_batched.pth` was trained with
    (100 Flickr30k pairs, batch size 8, 3 epochs, AdamW at 1e-4), so re-running
    `train` reproduces that run rather than some other one.

    Attributes:
        clip_model_name: CLIP checkpoint used for the vision encoder and both
            projections.
        qwen_model_name: Decoder the adapter feeds a soft token into.
        dataset_name: Hugging Face dataset the captions come from.
        dataset_revision: The Hub's auto-converted parquet mirror. Current `datasets`
            no longer executes a repo's own loading script, so the original revision
            cannot be read.
        dataset_split: Split to sample from. Flickr30k on the Hub ships one `test`
            split holding all 31k images.
        num_samples: Images sampled from that split for a run.
        train_fraction: Share of those samples used for training; the rest is the
            held-out set the baseline and the best checkpoint are scored on.
        max_caption_length: Caption tokens kept before truncation.
        final_layer_norm: Whether a newly trained adapter ends in a `LayerNorm`.
        checkpoint_name: File the best adapter is written to, under `checkpoint_dir`.
        prompt: Text prefix the image's soft token is prepended to at inference.
        max_new_tokens: Generation cap for a caption.
        zero_shot_items: Labels the Part 1 zero-shot demo classifies against; one
            `data/images/<item>.jpg` exists per entry.
        api_host: Interface the FastAPI service binds to.
        api_port: Port the FastAPI service listens on.
    """

    batch_size: int = 8
    learning_rate: float = 1e-4
    num_epochs: int = 3
    early_stopping_patience: int = 3

    clip_model_name: str = "openai/clip-vit-base-patch32"
    qwen_model_name: str = "Qwen/Qwen3-0.6B-Base"

    dataset_name: str = "nlphuji/flickr30k"
    dataset_revision: str = "refs/convert/parquet"
    dataset_split: str = "test"
    num_samples: int = 100
    train_fraction: float = 0.8
    max_caption_length: int = 32

    final_layer_norm: bool = False
    checkpoint_name: str = DEFAULT_CHECKPOINT

    # Training runs on CPU on purpose, and this was measured rather than assumed. Ten
    # identical MPS runs of the default config produced the same result eight times, once
    # diverged upwards, and once reported a loss of exactly 0.0000 from epoch 2 onward —
    # then saved that as the best checkpoint. Three CPU runs were byte-identical. A
    # forward pass is fine on MPS (fixed batch, eight separate processes, identical to
    # eight decimal places), so inference keeps using `get_device()`; it is the backward
    # pass over a frozen 596M-parameter decoder that misbehaves. At this scale CPU costs
    # about 45 seconds against 15, which is not worth a silently wrong checkpoint.
    #
    # Re-tested 2026-10-17 and this still stands. Five steps on each backend agree to
    # three decimal places and MPS is 1.56x faster (1.11s against 1.73s a step), which
    # looks like a reason to switch and is not one: five steps cannot see a failure that
    # showed up in two runs out of ten. The expensive half of that rate is a checkpoint
    # reporting a perfect loss and being saved as the best one.
    train_on_cpu: bool = True

    prompt: str = "This is a photo of:"
    max_new_tokens: int = 30

    zero_shot_items: tuple[str, ...] = (
        "cat",
        "dog",
        "horse",
        "bird",
        "car",
        "person",
        "tree",
        "house",
        "book",
        "phone",
    )

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    hf_repo_id: str | None = "roshbeed/ai-residency-multimodal-captioning"
    # The ten demo images the zero-shot script and the UI's sample picker use. They
    # are fixtures rather than a dataset, but they are still data, and data does not
    # belong in git.
    hf_dataset_repo_id: str | None = "roshbeed/ai-residency-multimodal-captioning-data"
    wandb_project: str | None = "ai-residency-multimodal-captioning"

    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    data_dir: Path = field(default_factory=lambda: Path("data"))

    @property
    def image_dir(self) -> Path:
        """Directory holding the ten sample images the zero-shot demo classifies."""
        return self.data_dir / "images"
