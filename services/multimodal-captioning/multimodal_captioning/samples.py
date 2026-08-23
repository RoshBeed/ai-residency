"""Demo images for the zero-shot script and the UI's sample picker.

Each is named after the label it should match, which is what makes the zero-shot
demo self-checking. They used to be committed; they are fetched from a dataset repo
now, on first use, because a repo is for code.
"""

from pathlib import Path

from ai_residency_common import get_logger, resolve_checkpoint

from config import Config

__all__ = ["available_samples", "sample_image"]

logger = get_logger(__name__)


def sample_image(config: Config, name: str) -> Path | None:
    """Return a demo image, downloading it if this checkout does not have it.

    Args:
        config: Supplies `image_dir` and `hf_dataset_repo_id`.
        name: Image stem, e.g. `"cat"`.

    Returns:
        Path to the image, or None if it is neither local nor on the Hub.
    """
    path, _ = resolve_checkpoint(
        config.image_dir / f"{name}.jpg", config.hf_dataset_repo_id, repo_type="dataset"
    )
    return path


def available_samples(config: Config) -> list[Path]:
    """Return every demo image that can be shown, fetching any that are missing.

    Args:
        config: Supplies `image_dir`, `zero_shot_items` and `hf_dataset_repo_id`.

    Returns:
        The images that resolved, sorted by name. Empty when offline with an empty
        checkout, which the caller should render as "no samples" rather than an error.
    """
    resolved = [sample_image(config, name) for name in config.zero_shot_items]
    return sorted(path for path in resolved if path is not None)
