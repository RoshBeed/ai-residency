"""Fetch the raw speech-emotion corpora.

The three corpora this week names are not equally obtainable, and that asymmetry is
the reason this stage exists as its own step:

- **EmoDB** downloads directly as a ~40MB zip with no authentication. It is the only
  one that reliably lands on a fresh machine, and it is what every artifact in this
  repo was actually produced from.
- **CREMA-D** needs a multi-gigabyte `git lfs clone`. It works, but it is slow enough
  that nothing downstream should assume it is present.
- **RAVDESS** needs registration, so this only creates the directory shape and points
  at the download page. It cannot fetch the audio.

Everything downstream treats a missing corpus as a skip rather than an error, so
running this stage is optional.
"""

import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import requests
from ai_residency_common import get_logger

from config import Config

__all__ = [
    "download_file",
    "main",
    "setup_crema_d",
    "setup_emodb",
    "setup_ravdess",
]

logger = get_logger(__name__)

EMODB_URL = "http://emodb.bilderbar.info/download/download.zip"
CREMA_D_REPO = "https://github.com/CheyneyComputerScience/CREMA-D.git"
RAVDESS_URL = "https://zenodo.org/record/1188976"

RAVDESS_EMOTIONS = (
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised",
)


def download_file(url: str, destination: Path) -> None:
    """Stream a URL to a file.

    Args:
        url: What to download.
        destination: Where to write it.
    """
    logger.info("Downloading %s", url)
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    with destination.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=8192):
            handle.write(chunk)
    logger.info("Downloaded %s", destination)


def setup_ravdess(config: Config) -> Path:
    """Create the RAVDESS directory shape and a metadata stub.

    The audio itself needs registration, so this cannot download anything.

    Args:
        config: Supplies the dataset root.

    Returns:
        The directory that was created.
    """
    dataset_dir = config.datasets_dir / "ravdess"
    for emotion in RAVDESS_EMOTIONS:
        (dataset_dir / emotion).mkdir(parents=True, exist_ok=True)

    metadata = {
        "dataset": "RAVDESS",
        "description": "Ryerson Audio-Visual Database of Emotional Speech and Song",
        "emotions": list(RAVDESS_EMOTIONS),
        "total_files": 0,
        "format": "wav",
        "sample_rate": 16000,
        "download_url": RAVDESS_URL,
    }
    with (dataset_dir / "metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)

    logger.info("Created RAVDESS directory shape at %s", dataset_dir)
    logger.info("RAVDESS audio needs registration: %s", RAVDESS_URL)
    return dataset_dir


def setup_crema_d(config: Config) -> Path | None:
    """Clone CREMA-D via git-lfs.

    Args:
        config: Supplies the dataset root.

    Returns:
        The cloned directory, or None if git-lfs is unavailable or the clone failed.
    """
    dataset_dir = config.datasets_dir / "crema_d"

    try:
        subprocess.run(["git", "lfs", "version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("git-lfs is not installed. Install it with: brew install git-lfs")
        return None

    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    logger.info("Cloning CREMA-D (multi-gigabyte git-lfs clone, this is slow)")
    try:
        subprocess.run(["git", "lfs", "clone", CREMA_D_REPO, str(dataset_dir)], check=True)
    except subprocess.CalledProcessError as error:
        logger.error("Failed to clone CREMA-D: %s", error)
        return None

    # The repo nests everything one level deeper than the paths downstream expect.
    nested = dataset_dir / "CREMA-D"
    if nested.exists():
        for item in nested.iterdir():
            shutil.move(str(item), str(dataset_dir))
        nested.rmdir()

    logger.info("CREMA-D ready at %s", dataset_dir)
    return dataset_dir


def setup_emodb(config: Config) -> Path:
    """Download and extract EmoDB.

    Args:
        config: Supplies the dataset root.

    Returns:
        The extracted directory.
    """
    dataset_dir = config.datasets_dir / "emodb"
    zip_path = dataset_dir / "emodb.zip"

    download_file(EMODB_URL, zip_path)
    logger.info("Extracting EmoDB")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(dataset_dir)
    zip_path.unlink()

    logger.info("EmoDB ready at %s", dataset_dir)
    return dataset_dir


def main() -> None:
    """Entry point for the `download` pipeline stage."""
    config = Config()
    setup_ravdess(config)
    setup_emodb(config)
    setup_crema_d(config)
    logger.info("Dataset setup complete. Run the prepare stage next.")


if __name__ == "__main__":
    main()
