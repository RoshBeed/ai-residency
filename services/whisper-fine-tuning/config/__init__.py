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
        num_epochs: Fine-tuning steps taken on the single example by `make demo`
            and `make train`.
        serve_num_steps: Steps `/transcribe` takes per request. Higher than
            `num_epochs` on purpose -- the endpoint streams a transcript between every
            step, so more steps show more of the trajectory -- and it was a literal
            written out twice in `serving/api.py` before it was a field. The cost is
            measured: one step corrects the demo clip and leaves word error rate on
            twelve held-out sentences at 0.1642, five steps takes it to 0.2687. The
            fine-tune is per-request and discarded, so that cost lands on the
            response, not on a stored checkpoint.
        model_name: Which Whisper checkpoint to fine-tune, as a Hugging Face id.
            Was a bare `"tiny"` while this service used `openai-whisper`; that
            package names its checkpoints differently from the Hub.
        checkpoint_name: File name for the fine-tuned weights, under `checkpoint_dir`.
        audio_path: Where the demo clip lives locally. It is not committed -- the
            runtime is ephemeral -- so it is pulled from `hf_dataset_repo_id` when
            absent, into this path.
        target_text: Ground truth for `audio_path`.
    """

    learning_rate: float = 1e-4
    num_epochs: int = 1
    serve_num_steps: int = 5

    model_name: str = "openai/whisper-tiny"
    checkpoint_name: str = "whisper_finetuned.pth"

    audio_path: Path = field(default_factory=lambda: Path("data/audio.mp3"))
    target_text: str = "My name is Rosh"

    hf_repo_id: str | None = "roshbeed/ai-residency-whisper-fine-tuning"
    hf_dataset_repo_id: str | None = "roshbeed/ai-residency-whisper-fine-tuning-data"
    wandb_project: str | None = "ai-residency-whisper-fine-tuning"
