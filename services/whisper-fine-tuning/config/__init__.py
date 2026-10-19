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
            written out twice in `serving/api.py` before it was a field.

            **Two, and the number moved when a shared library changed under it.** It
            was five, chosen from two data points; measuring every count in between put
            it at three. Then `speech.load_waveform` started normalising every clip to a
            fixed RMS -- for the emotion service, where a quiet upload was scoring near
            chance -- and that made this service's demo clip louder, its gradients
            larger, and its overfitting arrive sooner. The curve is not the same curve:

            | steps | demo clip reads | WER | sentences altered |
            |---|---|--:|--:|
            | 1 | "My name is Rosh" | 0.1493 | 2/12 |
            | **2** | **"My name is Rosh"** | **0.1493** | **2/12** |
            | 3 | "Mena Nis Rosh" | 0.1343 | 3/12 |
            | 4 | "My name is Rosh" | 0.2388 | 5/12 |
            | 5 | "My name is Rosh" | **1.1791** | 8/12 |

            Five now does more than four times the damage it did before the change, and
            three breaks the demo outright -- the one thing this service exists to show.
            Two corrects the clip, keeps a streamed trajectory, and sits below the base
            model's error rate.

            The lesson is not the number. **A constant tuned against one service's
            preprocessing is invalidated when a shared library changes that
            preprocessing**, silently and in another service's commit.

            The fine-tune is per-request and discarded, so that cost lands on the
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
    serve_num_steps: int = 2

    model_name: str = "openai/whisper-tiny"
    checkpoint_name: str = "whisper_finetuned.pth"

    audio_path: Path = field(default_factory=lambda: Path("data/audio.mp3"))
    target_text: str = "My name is Rosh"

    hf_repo_id: str | None = "roshbeed/ai-residency-whisper-fine-tuning"
    hf_dataset_repo_id: str | None = "roshbeed/ai-residency-whisper-fine-tuning-data"
    wandb_project: str | None = "ai-residency-whisper-fine-tuning"
