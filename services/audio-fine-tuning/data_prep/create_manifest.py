"""Write `(audio_path, label)` CSV manifests for each prepared train/val split.

A manifest is the handoff point between the data pipeline and any training loop: one
row per clip, an absolute path and an emotion label, nothing else. Manifests are
written under `data/` alongside the other generated artifacts rather than into
`datasets/`, which holds only third-party corpora.

Every corpus named in `Config.dataset_keys` is checked for on disk and skipped if
absent, which is the normal case for CREMA-D — it needs a multi-gigabyte git-lfs clone
that a fresh checkout will not have.
"""

import csv
from pathlib import Path

from ai_residency_common import get_logger

from config import Config

__all__ = ["create_manifest", "create_manifests", "main"]

logger = get_logger(__name__)


def create_manifest(split_dir: Path, output_csv: Path) -> int:
    """Write one manifest for one split directory.

    Args:
        split_dir: Directory holding one subdirectory of audio per emotion.
        output_csv: File to write.

    Returns:
        How many rows were written.
    """
    rows: list[dict[str, str]] = []
    for emotion_dir in sorted(split_dir.iterdir()):
        if not emotion_dir.is_dir():
            continue
        for audio_file in sorted(emotion_dir.glob("*.wav")):
            rows.append({"audio_path": str(audio_file.resolve()), "label": emotion_dir.name})

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["audio_path", "label"])
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Wrote %d rows to %s", len(rows), output_csv)
    return len(rows)


def create_manifests(config: Config) -> list[Path]:
    """Write manifests for every split of every corpus present on disk.

    Args:
        config: Run configuration.

    Returns:
        The manifest files that were written.
    """
    written: list[Path] = []
    for spec in config.datasets:
        split_root = config.split_dir(spec)
        if not split_root.exists():
            logger.warning("Skipping %s: nothing at %s", spec.display_name, split_root)
            continue
        for split in ("train", "val"):
            split_dir = split_root / split
            if not split_dir.exists():
                logger.warning("Skipping %s %s: nothing at %s", spec.display_name, split, split_dir)
                continue
            output_csv = config.manifest_path(spec, split)
            create_manifest(split_dir, output_csv)
            written.append(output_csv)
    return written


def main() -> None:
    """Entry point for the `manifest` pipeline stage."""
    create_manifests(Config())


if __name__ == "__main__":
    main()
