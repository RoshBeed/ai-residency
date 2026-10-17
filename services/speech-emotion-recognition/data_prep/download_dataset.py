"""Fetch the raw speech-emotion corpora.

The three corpora named here are not equally obtainable, and that asymmetry is
the reason this stage exists as its own step:

- **EmoDB** downloads directly as a ~40MB zip with no authentication. It is the only
  one that reliably lands on a fresh machine, and it is what every artifact in this
  repo was actually produced from.
- **CREMA-D** is what the shipped checkpoint was trained on. Its own repository holds
  video, audio and perceptual ratings behind a multi-gigabyte `git lfs clone`, so this
  fetches the audio from the project's own mirror instead: one 473MB tarball, no git-lfs,
  no account. Mirroring it is the same call made for text8 — a published corpus the
  pipeline must not depend on someone else's infrastructure to reach.
- **RAVDESS** needs registration, so this only creates the directory shape and points
  at the download page. It cannot fetch the audio.

Everything downstream treats a missing corpus as a skip rather than an error, so
running this stage is optional.
"""

import json
import re
import shutil
import tarfile
import zipfile
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download

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
# CREMA-D's own repository carries video, audio and perceptual ratings, and needs a
# multi-gigabyte `git lfs clone` to get at the WAVs. This mirror is the audio only,
# as one tarball, under the same ODbL licence. See the dataset card for attribution.
CREMA_D_MIRROR = "roshbeed/ai-residency-speech-emotion-recognition-data"
CREMA_D_ARCHIVE = "crema_d.tar.gz"
CREMA_D_SOURCE = "https://github.com/CheyneyComputerScience/CREMA-D"
CREMA_D_CLIP = re.compile(r"^\d{4}_[A-Z]{3}_[A-Z]{3}_[A-Z]{2}\.wav$")
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
    """Fetch CREMA-D's audio from the project's own mirror.

    Args:
        config: Supplies the dataset root and the mirror's repo id.

    Returns:
        The directory holding the WAVs, or None if the download failed.
    """
    dataset_dir = config.datasets_dir / "crema_d"
    audio_dir = dataset_dir / "AudioWAV"

    if audio_dir.exists() and any(audio_dir.glob("*.wav")):
        logger.info("CREMA-D already present at %s", audio_dir)
        return dataset_dir

    logger.info("Downloading CREMA-D audio from %s (473MB)", CREMA_D_MIRROR)
    try:
        tar_path = hf_hub_download(
            CREMA_D_MIRROR, CREMA_D_ARCHIVE, repo_type="dataset", local_dir=str(dataset_dir)
        )
    except Exception as error:
        # Downstream stages treat a missing corpus as a skip, so a failure here must not
        # take the whole download stage down with it.
        logger.error("Failed to download CREMA-D: %s", error)
        return None

    audio_dir.mkdir(parents=True, exist_ok=True)
    kept = skipped = 0
    with tarfile.open(tar_path) as archive:
        for member in archive:
            # Extract by basename against CREMA-D's own naming, and only plain files.
            # A tarball is attacker-controlled input in general: this drops directory
            # entries, links, and any path that would escape `audio_dir`.
            name = Path(member.name).name
            if not member.isfile() or not CREMA_D_CLIP.match(name):
                skipped += 1
                continue
            source = archive.extractfile(member)
            if source is None:
                skipped += 1
                continue
            with open(audio_dir / name, "wb") as target:
                shutil.copyfileobj(source, target)
            kept += 1

    Path(tar_path).unlink(missing_ok=True)
    logger.info("CREMA-D ready at %s: %d clips (%d members skipped)", audio_dir, kept, skipped)
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
