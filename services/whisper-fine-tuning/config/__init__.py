"""Run configuration for the single-example Whisper fine-tuning demo.

Everything the demo needs in one dataclass: which Whisper checkpoint to fine-tune, the
clip to fine-tune on, and the ground truth for it. Whisper-tiny mishears the name in
that clip, which is the whole point -- the demo is a before-and-after on one example,
so the effect of a handful of gradient steps is visible rather than statistical.
"""

from dataclasses import dataclass, field
from pathlib import Path

from ai_residency_common import BaseConfig

__all__ = ["Config"]


@dataclass
class Config(BaseConfig):
    """Configuration for the fine-tuning demo and the service that serves it.

    Attributes:
        learning_rate: Adam learning rate for the fine-tuning steps.
        num_epochs: Fine-tuning steps taken on the single example.
        whisper_model_name: Which `openai-whisper` checkpoint to fine-tune.
        checkpoint_name: File name for the fine-tuned weights, under `checkpoint_dir`.
        audio_path: The demo clip to fine-tune on.
        target_text: Ground truth for `audio_path`.
    """

    learning_rate: float = 1e-4
    num_epochs: int = 1

    whisper_model_name: str = "tiny"
    checkpoint_name: str = "whisper_finetuned.pth"

    audio_path: Path = field(default_factory=lambda: Path("data/audio.mp3"))
    target_text: str = "My name is Rosh"

    hf_repo_id: str | None = "roshbeed/ai-residency-week5-audio-fine-tuning"
    wandb_project: str | None = "ai-residency-whisper-fine-tuning"
