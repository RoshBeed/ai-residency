"""Run configuration.

Before this pass the repo had no configuration layer at all: the model size, the
learning rate, the demo clip's path and its ground-truth text, the emotion token list
and every dataset path were hardcoded at their point of use, in five different files.
`Config` collects all of it, inheriting the portfolio-wide fields (seed, batch_size,
learning_rate, num_epochs, checkpoint_dir, data_dir, hf_repo_id, wandb_project) from
`BaseConfig`.

`DATASET_SPECS` matters more here than in most repos. Two of the three corpora this
pipeline names are not actually obtainable without extra work — CREMA-D needs a
multi-gigabyte git-lfs clone and RAVDESS needs registration — so on a normal checkout
only EmoDB is on disk. Making the dataset set an explicit, inspectable config field
rather than a literal repeated across four scripts is what keeps "which datasets does
this run actually cover?" answerable.
"""

from dataclasses import dataclass, field
from pathlib import Path

from ai_residency_common import BaseConfig

__all__ = ["DATASET_SPECS", "Config", "DatasetSpec"]


@dataclass(frozen=True)
class DatasetSpec:
    """Where one speech-emotion corpus lives and what it is called.

    Attributes:
        key: Short identifier used in `Config.dataset_keys`.
        display_name: Human-readable name used in logs and plot titles.
        description: One-line description of the corpus.
        source_subdir: Raw audio directory, relative to `Config.datasets_dir`.
        organized_name: Directory name for the by-emotion copy, under
            `Config.datasets_dir`. The train/val split is written alongside it with a
            `_train_val` suffix.
        manifest_prefix: Prefix for this corpus's generated `{prefix}_{split}.csv`.
        emotion_codes: Maps the emotion code embedded in a filename to an emotion name.
        code_separator: Character the filename stem is split on to find the emotion
            code, or None if the code is a single character at a fixed position.
        code_index: Field index after splitting on `code_separator`, or the character
            offset into the stem when `code_separator` is None.
    """

    key: str
    display_name: str
    description: str
    source_subdir: str
    organized_name: str
    manifest_prefix: str
    emotion_codes: dict[str, str]
    code_separator: str | None
    code_index: int


# CREMA-D encodes the emotion as the third underscore-separated field of the stem
# (`1091_IEO_SAD_HI`); EmoDB encodes it as the sixth character (`08a01Fd`).
DATASET_SPECS: dict[str, DatasetSpec] = {
    "crema_d": DatasetSpec(
        key="crema_d",
        display_name="CREMA-D",
        description="Crowd-sourced Emotional Multimodal Actors Dataset",
        source_subdir="crema_d/AudioWAV",
        organized_name="crema_d_organized",
        manifest_prefix="cremad",
        emotion_codes={
            "ANG": "angry",
            "DIS": "disgust",
            "FEA": "fearful",
            "HAP": "happy",
            "NEU": "neutral",
            "SAD": "sad",
        },
        code_separator="_",
        code_index=2,
    ),
    "emodb": DatasetSpec(
        key="emodb",
        display_name="EmoDB",
        description="Berlin Database of Emotional Speech",
        source_subdir="emodb/wav",
        organized_name="emodb_organized",
        manifest_prefix="emodb",
        emotion_codes={
            "W": "happy",
            "L": "sad",
            "A": "angry",
            "F": "fearful",
            "E": "disgust",
            "T": "surprised",
            "N": "neutral",
        },
        code_separator=None,
        code_index=5,
    ),
}


@dataclass
class Config(BaseConfig):
    """Configuration for Week 5's fine-tuning run and data pipeline.

    Attributes:
        whisper_model_name: Which `openai-whisper` checkpoint to fine-tune.
        tokenizer_model_name: Hugging Face model whose tokenizer the emotion tokens
            are added to. Deliberately a different size from `whisper_model_name`:
            the tokenizer is shared across the Whisper family, and `whisper-small` is
            what the original team used.
        emotion_tokens: Special tokens added to the tokenizer, one per emotion label
            the datasets below use.
        tokenizer_dir_name: Directory under `checkpoint_dir` for the extended tokenizer.
        checkpoint_name: File name for the fine-tuned weights, under `checkpoint_dir`.
        audio_path: The demo clip Part 1 fine-tunes on.
        target_text: Ground truth for `audio_path`. Whisper-tiny mishears the name,
            which is the whole point of the single-example demo.
        datasets_dir: Root for third-party corpora. Separate from `data_dir`, which
            holds generated artifacts (manifests, plots) rather than raw audio.
        dataset_keys: Which corpora in `DATASET_SPECS` the pipeline covers.
        val_split: Fraction of each emotion's files held out for validation.
        samples_per_emotion: Files sampled per emotion when profiling audio duration.
    """

    learning_rate: float = 1e-4
    num_epochs: int = 1

    whisper_model_name: str = "tiny"
    tokenizer_model_name: str = "openai/whisper-small"
    emotion_tokens: tuple[str, ...] = (
        "<emotion_happy>",
        "<emotion_sad>",
        "<emotion_fearful>",
        "<emotion_neutral>",
        "<emotion_angry>",
        "<emotion_disgust>",
    )
    tokenizer_dir_name: str = "whisper_tokenizer_emotion"
    checkpoint_name: str = "whisper_finetuned.pth"

    audio_path: Path = field(default_factory=lambda: Path("data/audio.mp3"))
    target_text: str = "My name is Rosh"

    datasets_dir: Path = field(default_factory=lambda: Path("datasets"))
    dataset_keys: tuple[str, ...] = ("crema_d", "emodb")
    val_split: float = 0.2
    samples_per_emotion: int = 5

    hf_repo_id: str | None = "roshbeed/ai-residency-week5-audio-fine-tuning"
    wandb_project: str | None = "ai-residency-week5-audio-fine-tuning"

    @property
    def tokenizer_dir(self) -> Path:
        """Full path to the emotion-extended tokenizer directory."""
        return self.checkpoint_dir / self.tokenizer_dir_name

    @property
    def datasets(self) -> list[DatasetSpec]:
        """The dataset specs this run covers, in `dataset_keys` order."""
        return [DATASET_SPECS[key] for key in self.dataset_keys]

    def source_dir(self, spec: DatasetSpec) -> Path:
        """Raw audio directory for a corpus, as downloaded."""
        return self.datasets_dir / spec.source_subdir

    def organized_dir(self, spec: DatasetSpec) -> Path:
        """Directory holding one subdirectory of audio per emotion."""
        return self.datasets_dir / spec.organized_name

    def split_dir(self, spec: DatasetSpec) -> Path:
        """Directory holding the `train/` and `val/` subdirectories."""
        return self.datasets_dir / f"{spec.organized_name}_train_val"

    def manifest_path(self, spec: DatasetSpec, split: str) -> Path:
        """Generated CSV manifest for one corpus and split."""
        return self.data_dir / f"{spec.manifest_prefix}_{split}.csv"
