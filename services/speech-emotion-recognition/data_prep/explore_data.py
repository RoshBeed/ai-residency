"""Exploratory analysis of whatever speech-emotion corpora are actually on disk.

Produces the three figures the README's data section uses (class balance, raw
spectrograms, example mel spectrograms) plus a printed summary of clip durations and
sample rates.

Every count here is recounted from disk or read from the `metadata.json` the prepare
stage wrote. Nothing is hardcoded. That is worth stating explicitly because the
previous version of the summary printed fixed totals (7,442 CREMA-D files and 535
EmoDB files) regardless of what was present, and CREMA-D has never actually been
downloaded in this environment — so it confidently reported 7,442 files for a corpus
with zero files on disk.
"""

import json
from pathlib import Path
from typing import Any

import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ai_residency_common import get_logger
from ai_residency_common.figures import save_figure
from config import Config, DatasetSpec

# macOS's default backend is interactive and blocks on show() when these run as a
# plain script rather than under Jupyter, which hung a verification run in an earlier
# pass. Agg renders straight to file and never opens a window.
matplotlib.use("Agg")

__all__ = [
    "analyze_audio_characteristics",
    "analyze_dataset_structure",
    "available_datasets",
    "create_visualizations",
    "explore",
    "main",
    "plot_example_spectrograms",
    "print_dataset_summary",
]

logger = get_logger(__name__)


def available_datasets(config: Config) -> list[DatasetSpec]:
    """Return the configured corpora that actually have training audio on disk.

    The test is the `train/` subdirectory rather than the split directory itself,
    because a split directory can exist as an empty shell holding nothing but a
    leftover `metadata.json` — which is exactly the state CREMA-D is in here, having
    never been downloaded. Testing the outer directory would report it as available
    and then silently produce no plots.

    Args:
        config: Run configuration.

    Returns:
        Specs in config order, filtered to what is genuinely present.
    """
    return [spec for spec in config.datasets if (config.split_dir(spec) / "train").is_dir()]


def _count_by_emotion(split_dir: Path) -> dict[str, int]:
    """Count `.wav` files per emotion subdirectory."""
    return {
        emotion_dir.name: len(list(emotion_dir.glob("*.wav")))
        for emotion_dir in sorted(split_dir.glob("*"))
        if emotion_dir.is_dir()
    }


def analyze_dataset_structure(config: Config) -> None:
    """Log the per-emotion file counts of every split that exists.

    Args:
        config: Run configuration.
    """
    logger.info("Dataset structure")

    for spec in config.datasets:
        split_root = config.split_dir(spec)
        if not split_root.exists():
            logger.warning("%s: not found at %s", spec.display_name, split_root)
            continue

        logger.info("%s:", spec.display_name)
        for split in ("train", "val"):
            split_dir = split_root / split
            if not split_dir.exists():
                continue
            counts = _count_by_emotion(split_dir)
            logger.info("  %s:", split.upper())
            for emotion, count in sorted(counts.items()):
                logger.info("    %s: %d files", emotion, count)
            logger.info("    Total: %d files", sum(counts.values()))


def analyze_audio_characteristics(config: Config) -> None:
    """Log duration and sample-rate statistics from a sample of each corpus.

    Args:
        config: Supplies `samples_per_emotion`.
    """
    logger.info("Audio characteristics")

    for spec in available_datasets(config):
        train_dir = config.split_dir(spec) / "train"
        if not train_dir.exists():
            continue

        logger.info("%s:", spec.display_name)
        durations: list[float] = []
        sample_rates: set[int] = set()

        for emotion_dir in sorted(train_dir.glob("*")):
            if not emotion_dir.is_dir():
                continue
            emotion_durations: list[float] = []
            for audio_file in sorted(emotion_dir.glob("*.wav"))[: config.samples_per_emotion]:
                waveform, sample_rate = librosa.load(audio_file, sr=None)
                duration = float(librosa.get_duration(y=waveform, sr=sample_rate))
                emotion_durations.append(duration)
                durations.append(duration)
                sample_rates.add(int(sample_rate))
            if emotion_durations:
                logger.info(
                    "  %s: avg duration = %.2fs", emotion_dir.name, np.mean(emotion_durations)
                )

        if durations:
            logger.info(
                "  Duration mean=%.2fs min=%.2fs max=%.2fs std=%.2fs",
                np.mean(durations),
                np.min(durations),
                np.max(durations),
                np.std(durations),
            )
            logger.info("  Sample rates: %s", sorted(sample_rates))


def create_visualizations(config: Config) -> Path | None:
    """Plot the train/val class balance and emotion distribution per corpus.

    Args:
        config: Run configuration.

    Returns:
        The figure that was written, or None if no corpus is on disk.
    """
    specs = available_datasets(config)
    if not specs:
        logger.warning("No corpora on disk; skipping class-balance plots")
        return None

    sns.set_palette("husl")
    figure, axes = plt.subplots(len(specs), 2, figsize=(15, 6 * len(specs)), squeeze=False)
    figure.suptitle("Speech Emotion Dataset Analysis", fontsize=16)

    for row, spec in enumerate(specs):
        split_root = config.split_dir(spec)
        train_counts = _count_by_emotion(split_root / "train")
        val_counts = _count_by_emotion(split_root / "val")
        emotions = sorted(train_counts)

        positions = np.arange(len(emotions))
        width = 0.35
        bar_axis = axes[row][0]
        bar_axis.bar(
            positions - width / 2,
            [train_counts.get(emotion, 0) for emotion in emotions],
            width,
            label="Train",
            alpha=0.8,
        )
        bar_axis.bar(
            positions + width / 2,
            [val_counts.get(emotion, 0) for emotion in emotions],
            width,
            label="Validation",
            alpha=0.8,
        )
        bar_axis.set_xlabel("Emotion")
        bar_axis.set_ylabel("Number of Files")
        bar_axis.set_title(f"{spec.display_name} - Train/Validation Split")
        bar_axis.set_xticks(positions)
        bar_axis.set_xticklabels(emotions, rotation=45)
        bar_axis.legend()
        bar_axis.grid(True, alpha=0.3)

        pie_axis = axes[row][1]
        sizes = [train_counts[emotion] for emotion in emotions]
        if sum(sizes) > 0:
            pie_axis.pie(
                sizes,
                labels=emotions,
                autopct="%1.1f%%",
                colors=plt.cm.Set3(np.linspace(0, 1, len(emotions))),
                startangle=90,
            )
        pie_axis.set_title(f"{spec.display_name} - Emotion Distribution (Train)")

    plt.tight_layout()
    output_path = config.data_dir / "dataset_analysis.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(plt.gcf(), output_path)
    plt.close(figure)
    logger.info("Saved class-balance plots to %s", output_path)
    return output_path


def _sample_per_emotion(train_dir: Path, per_emotion: int) -> list[tuple[Path, str]]:
    """Return up to `per_emotion` files from each emotion subdirectory."""
    samples: list[tuple[Path, str]] = []
    for emotion_dir in sorted(train_dir.glob("*")):
        if emotion_dir.is_dir():
            for audio_file in sorted(emotion_dir.glob("*.wav"))[:per_emotion]:
                samples.append((audio_file, emotion_dir.name))
    return samples


def plot_example_spectrograms(config: Config) -> list[Path]:
    """Plot one raw spectrogram per emotion and a grid of example mel spectrograms.

    Args:
        config: Run configuration.

    Returns:
        The figures that were written.
    """
    specs = available_datasets(config)
    if not specs:
        logger.warning("No corpora on disk; skipping spectrogram plots")
        return []

    spec = specs[0]
    train_dir = config.split_dir(spec) / "train"
    written: list[Path] = []

    one_per_emotion = _sample_per_emotion(train_dir, 1)
    if one_per_emotion:
        figure, axes = plt.subplots(2, 3, figsize=(18, 10))
        figure.suptitle(f"Audio Spectrograms by Emotion ({spec.display_name})", fontsize=16)
        for index, (audio_file, emotion) in enumerate(one_per_emotion[:6]):
            axis = axes[index // 3][index % 3]
            waveform, sample_rate = librosa.load(audio_file, sr=16000)
            decibels = librosa.amplitude_to_db(np.abs(librosa.stft(waveform)), ref=np.max)
            librosa.display.specshow(decibels, sr=sample_rate, x_axis="time", y_axis="log", ax=axis)
            axis.set_title(emotion.capitalize())
            axis.set_xlabel("Time (s)")
            axis.set_ylabel("Frequency (Hz)")
        plt.tight_layout()
        output_path = config.data_dir / "audio_spectrograms.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_figure(plt.gcf(), output_path)
        plt.close(figure)
        logger.info("Saved spectrograms to %s", output_path)
        written.append(output_path)

    examples = _sample_per_emotion(train_dir, 2)
    if examples:
        columns = 2
        rows = (len(examples) + columns - 1) // columns
        figure, axes = plt.subplots(rows, columns, figsize=(12, 3 * rows), squeeze=False)
        figure.suptitle(f"Example Mel Spectrograms ({spec.display_name})", fontsize=16)
        for index, (audio_file, emotion) in enumerate(examples):
            axis = axes[index // columns][index % columns]
            waveform, sample_rate = librosa.load(audio_file, sr=16000)
            mel = librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_mels=128, fmax=8000)
            librosa.display.specshow(
                librosa.power_to_db(mel, ref=np.max),
                sr=sample_rate,
                x_axis="time",
                y_axis="mel",
                fmax=8000,
                ax=axis,
            )
            axis.set_title(f"{emotion.capitalize()}\n{audio_file.name}")
            axis.set_xlabel("Time (s)")
            axis.set_ylabel("Mel Frequency")
            logger.info("Example: %s | label: %s", audio_file.name, emotion)
        plt.tight_layout(rect=(0, 0, 1, 0.96))
        output_path = config.data_dir / "example_melspectrograms.png"
        save_figure(plt.gcf(), output_path)
        plt.close(figure)
        logger.info("Saved example mel spectrograms to %s", output_path)
        written.append(output_path)

    return written


def print_dataset_summary(config: Config) -> None:
    """Log a summary of each corpus, read from its own `metadata.json`.

    Reading the metadata the prepare stage wrote means the totals reflect what is
    genuinely on disk, including zero files for a corpus that was never downloaded.

    Args:
        config: Run configuration.
    """
    logger.info("Dataset summary")

    for spec in config.datasets:
        logger.info("%s:", spec.display_name)
        logger.info("  Description: %s", spec.description)

        metadata_file = config.split_dir(spec) / "metadata.json"
        if not metadata_file.exists():
            logger.warning("  Not on disk (no metadata.json at %s)", metadata_file)
            continue

        with metadata_file.open() as handle:
            metadata: dict[str, Any] = json.load(handle)

        # Read the counts rather than the `emotions` list: a metadata file left behind
        # by an older version of the prepare stage lists "train" and "val" there, which
        # would print as if they were emotion labels.
        emotion_counts: dict[str, int] = metadata.get("emotion_counts", {})
        emotions = sorted(name for name, count in emotion_counts.items() if count > 0)

        logger.info("  Emotions: %s", ", ".join(emotions) if emotions else "none on disk")
        logger.info("  Total files: %s", f"{metadata.get('total_files', 0):,}")
        logger.info("  Format: %s", str(metadata.get("format", "unknown")).upper())
        logger.info("  Sample rate: %sHz", metadata.get("sample_rate", "unknown"))
        logger.info("  Path: %s", config.split_dir(spec))


def explore(config: Config) -> None:
    """Run every analysis stage in order.

    Args:
        config: Run configuration.
    """
    analyze_dataset_structure(config)
    analyze_audio_characteristics(config)
    create_visualizations(config)
    plot_example_spectrograms(config)
    print_dataset_summary(config)


def main() -> None:
    """Entry point for the `explore` pipeline stage."""
    explore(Config())


if __name__ == "__main__":
    main()
