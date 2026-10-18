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

            **Three, because the whole curve was measured rather than the ends.** It was
            five, on the strength of two data points: one step and five. Running
            `make collateral` at every count in between says five is the only one that
            does real damage. Word error rate over twelve held-out CREMA-D sentences,
            against a 0.1642 base:

            | steps | WER | sentences altered |
            |---|--:|--:|
            | 1 | 0.1642 | 0/12 |
            | 2 | 0.1493 | 2/12 |
            | **3** | **0.1493** | **2/12** |
            | 4 | 0.1642 | 2/12 |
            | 5 | 0.2687 | 4/12 |

            The count of altered sentences matters more than the rate. Through four
            steps the two that change were *already wrong* at base -- whisper-tiny hears
            "it's eleven o'clock" as "Give us a lemon a clock" either way. At five,
            transcripts the base model got right start breaking: "The airplane is almost
            full" becomes "The airplay is almost full" and "Don't forget a jacket"
            becomes "Go for it". Three keeps a visible trajectory, corrects the demo
            clip, scores *below* the base rate, and damages nothing that worked.

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
    serve_num_steps: int = 3

    model_name: str = "openai/whisper-tiny"
    checkpoint_name: str = "whisper_finetuned.pth"

    audio_path: Path = field(default_factory=lambda: Path("data/audio.mp3"))
    target_text: str = "My name is Rosh"

    hf_repo_id: str | None = "roshbeed/ai-residency-whisper-fine-tuning"
    hf_dataset_repo_id: str | None = "roshbeed/ai-residency-whisper-fine-tuning-data"
    wandb_project: str | None = "ai-residency-whisper-fine-tuning"
