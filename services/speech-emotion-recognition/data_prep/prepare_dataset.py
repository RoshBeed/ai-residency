"""Organize raw speech-emotion corpora by emotion and split them into train and val.

Each corpus ships as one flat directory of `.wav` files with the emotion encoded in
the filename, in a different scheme per corpus (CREMA-D uses the third
underscore-separated field, EmoDB the sixth character). `DatasetSpec` in `config.py`
carries both the scheme and the code-to-emotion mapping, so this module has one
`organize_dataset` rather than one function per corpus.

The split is seeded through `set_seed`, driven by `Config.seed`, so the same files
land in train and val on every run. That matters more than it looks: the manifests
this produces are what any downstream fine-tuning would read, and an unseeded split
would silently change the evaluation set every time the pipeline is re-run.
"""

import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from ai_residency_common import get_logger, set_seed
from config import Config, DatasetSpec

__all__ = [
    "create_metadata",
    "create_train_val_split",
    "emotion_for_file",
    "main",
    "organize_dataset",
    "prepare_datasets",
]

logger = get_logger(__name__)


def emotion_for_file(spec: DatasetSpec, audio_file: Path) -> str | None:
    """Read the emotion label encoded in a filename.

    Args:
        spec: The corpus this file belongs to.
        audio_file: File whose stem carries the emotion code.

    Returns:
        The emotion name, or None if the stem is too short or the code is unknown.
    """
    stem = audio_file.stem

    if spec.code_separator is None:
        if len(stem) <= spec.code_index:
            return None
        code = stem[spec.code_index]
    else:
        fields = stem.split(spec.code_separator)
        if len(fields) <= spec.code_index:
            return None
        code = fields[spec.code_index]

    return spec.emotion_codes.get(code)


def organize_dataset(config: Config, spec: DatasetSpec) -> Path | None:
    """Copy a corpus into one subdirectory per emotion.

    Args:
        config: Supplies the dataset root.
        spec: The corpus to organize.

    Returns:
        The organized directory, or None if the raw audio is not on disk.
    """
    source_dir = config.source_dir(spec)
    target_dir = config.organized_dir(spec)

    if not source_dir.exists():
        logger.warning("%s not found at %s; skipping", spec.display_name, source_dir)
        return None

    logger.info("Organizing %s from %s", spec.display_name, source_dir)
    for known_emotion in set(spec.emotion_codes.values()):
        (target_dir / known_emotion).mkdir(parents=True, exist_ok=True)

    emotion_counts: defaultdict[str, int] = defaultdict(int)
    for audio_file in sorted(source_dir.glob("*.wav")):
        emotion = emotion_for_file(spec, audio_file)
        if emotion is None:
            logger.warning("Unrecognised emotion code in %s", audio_file.name)
            continue
        shutil.copy2(audio_file, target_dir / emotion / audio_file.name)
        emotion_counts[emotion] += 1

    logger.info("Organized %d files from %s", sum(emotion_counts.values()), spec.display_name)
    for emotion, count in sorted(emotion_counts.items()):
        logger.info("  %s: %d files", emotion, count)

    return target_dir


def create_train_val_split(config: Config, spec: DatasetSpec) -> Path:
    """Split an organized corpus into `train/` and `val/` subdirectories.

    Args:
        config: Supplies `val_split` and the dataset root.
        spec: The corpus to split.

    Returns:
        The directory containing the `train` and `val` subdirectories.
    """
    organized_dir = config.organized_dir(spec)
    split_dir = config.split_dir(spec)
    logger.info("Creating train/val split for %s", spec.display_name)

    # Clear any previous split before repopulating. Without this, re-running the stage
    # copies a new assignment on top of the old one and leaves every file that moved
    # sides present in *both* train and val: a second run on EmoDB produced 705 files
    # from a 535-file corpus, 170 of them duplicated across the split. Silent train/val
    # leakage, and invisible unless the assignment happens to change between runs.
    if split_dir.exists():
        shutil.rmtree(split_dir)

    emotion_dirs = sorted(path for path in organized_dir.iterdir() if path.is_dir())
    for split in ("train", "val"):
        for emotion_dir in emotion_dirs:
            (split_dir / split / emotion_dir.name).mkdir(parents=True, exist_ok=True)

    total_files = 0
    split_counts: dict[str, dict[str, int]] = {"train": {}, "val": {}}

    for emotion_dir in emotion_dirs:
        emotion = emotion_dir.name
        audio_files = sorted(emotion_dir.glob("*.wav"))
        random.shuffle(audio_files)

        split_index = int(len(audio_files) * (1 - config.val_split))
        for split, files in (
            ("train", audio_files[:split_index]),
            ("val", audio_files[split_index:]),
        ):
            for audio_file in files:
                shutil.copy2(audio_file, split_dir / split / emotion / audio_file.name)
            split_counts[split][emotion] = len(files)
            total_files += len(files)

    logger.info("Split %d files for %s", total_files, spec.display_name)
    for split in ("train", "val"):
        logger.info("  %s:", split)
        for emotion, count in sorted(split_counts[split].items()):
            logger.info("    %s: %d files", emotion, count)

    return split_dir


def create_metadata(config: Config, spec: DatasetSpec) -> dict[str, Any]:
    """Write a `metadata.json` describing what actually landed on disk.

    The counts are recounted from the split directory rather than carried over from
    the organizing step, so the file always describes real files. The layout is
    `{split}/{emotion}/*.wav`, so this has to recurse two levels: the split
    directory's own children are only `train` and `val`, which contain no audio.

    Args:
        config: Supplies the dataset root.
        spec: The corpus to describe.

    Returns:
        The metadata that was written.
    """
    split_dir = config.split_dir(spec)
    emotion_totals: defaultdict[str, int] = defaultdict(int)
    split_counts: dict[str, dict[str, int]] = {}
    total_files = 0

    for split in ("train", "val"):
        counts: dict[str, int] = {}
        for emotion_dir in sorted((split_dir / split).glob("*")):
            if emotion_dir.is_dir():
                file_count = len(list(emotion_dir.glob("*.wav")))
                counts[emotion_dir.name] = file_count
                emotion_totals[emotion_dir.name] += file_count
                total_files += file_count
        split_counts[split] = counts

    metadata: dict[str, Any] = {
        "dataset": spec.display_name,
        "description": f"Organized {spec.display_name} dataset for emotion detection",
        "emotions": sorted(emotion_totals),
        "total_files": total_files,
        "format": "wav",
        "sample_rate": 16000,
        "train_val_split": True,
        "emotion_counts": dict(emotion_totals),
        "split_counts": split_counts,
    }

    with (split_dir / "metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)

    logger.info("Wrote metadata for %s (%d files)", spec.display_name, total_files)
    return metadata


def prepare_datasets(config: Config) -> list[DatasetSpec]:
    """Organize, split and describe every corpus named in the config.

    Args:
        config: Run configuration.

    Returns:
        The specs that were actually present on disk and got prepared.
    """
    set_seed(config.seed)

    prepared: list[DatasetSpec] = []
    for spec in config.datasets:
        if organize_dataset(config, spec) is None:
            continue
        create_train_val_split(config, spec)
        create_metadata(config, spec)
        prepared.append(spec)
        logger.info("%s ready at %s", spec.display_name, config.split_dir(spec))

    if not prepared:
        logger.warning(
            "No corpora found under %s. Run the download stage first, or narrow "
            "Config.dataset_keys to what is actually available.",
            config.datasets_dir,
        )
    return prepared


def main() -> None:
    """Entry point for the `prepare` pipeline stage."""
    prepare_datasets(Config())


if __name__ == "__main__":
    main()
