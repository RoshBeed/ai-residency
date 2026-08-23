"""Run configuration for emotion-aware Whisper.

`DATASET_SPECS` matters more here than in most repos. The two corpora encode the
emotion inside the audio file's own name, in different formats, at different
positions -- so where the label comes from is configuration, not something a reader
should have to reverse-engineer from a loop.

Emotion is predicted as a token, not by a separate classifier head. Whisper's decoder
already emits special tokens for language and task; adding one per emotion to the same
vocabulary lets the model emit the label through the mechanism it already uses.
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
    """Configuration for the data pipeline, the fine-tuning run and the service.

    Attributes:
        learning_rate: AdamW learning rate.
        num_epochs: Passes over the training manifest.
        batch_size: Clips per optimizer step.
        model_name: Hugging Face Whisper checkpoint to fine-tune. The tokenizer the
            emotion tokens are added to comes from the same model, so the extended
            vocabulary and the model's embedding table cannot drift apart.
        emotion_tokens: Special tokens added to the tokenizer, one per emotion label
            the corpora below use.
        tokenizer_dir_name: Directory under `checkpoint_dir` for the extended tokenizer.
        checkpoint_name: File name for the fine-tuned weights, under `checkpoint_dir`.
        sample_rate: Sample rate every clip is resampled to before feature extraction.
        max_duration_seconds: Clips are trimmed or padded to this length.
        datasets_dir: Root for third-party corpora. Separate from `data_dir`, which
            holds generated artifacts (manifests, plots) rather than raw audio.
        dataset_keys: Which corpora in `DATASET_SPECS` the pipeline covers.
        val_split: Fraction of each emotion's files held out for validation.
        samples_per_emotion: Files sampled per emotion when profiling audio duration.
    """

    learning_rate: float = 1e-5
    num_epochs: int = 3
    batch_size: int = 4

    model_name: str = "openai/whisper-small"
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
    checkpoint_name: str = "whisper_emotion.pth"

    sample_rate: int = 16_000
    max_duration_seconds: float = 30.0

    datasets_dir: Path = field(default_factory=lambda: Path("datasets"))
    dataset_keys: tuple[str, ...] = ("crema_d", "emodb")
    val_split: float = 0.2
    samples_per_emotion: int = 5

    hf_repo_id: str | None = "roshbeed/ai-residency-speech-emotion-recognition"
    hf_dataset_repo_id: str | None = "roshbeed/ai-residency-speech-emotion-manifests"
    wandb_project: str | None = "ai-residency-speech-emotion-recognition"

    @property
    def tokenizer_dir(self) -> Path:
        """Full path to the emotion-extended tokenizer directory."""
        return self.checkpoint_dir / self.tokenizer_dir_name

    @property
    def datasets(self) -> list[DatasetSpec]:
        """The dataset specs this run covers, in `dataset_keys` order."""
        return [DATASET_SPECS[key] for key in self.dataset_keys]

    @property
    def emotions(self) -> tuple[str, ...]:
        """The emotion labels, in the order their tokens were added."""
        return tuple(token[len("<emotion_") : -1] for token in self.emotion_tokens)

    def token_for(self, emotion: str) -> str:
        """The special token standing for one emotion label."""
        return f"<emotion_{emotion}>"

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
