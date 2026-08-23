"""Pull trained artifacts back from the Hugging Face Hub.

The inverse of `upload_to_hf`, and the reason a clone of this repo can serve queries
without first spending hours on the training stages.
"""

from pathlib import Path

from ai_residency_common import get_logger, pull_checkpoint

from config import Config

__all__ = ["download_artifacts", "main"]

logger = get_logger(__name__)


def download_artifacts(config: Config) -> list[Path]:
    """Fetch both trained towers into `checkpoint_dir`.

    The CBOW checkpoint is not pulled by name because its file name carries the run
    timestamp, so there is nothing stable to ask for.

    Args:
        config: Supplies `hf_repo_id` and `checkpoint_dir`.

    Returns:
        The files that were downloaded. Empty if no token or repo is configured.
    """
    if config.hf_repo_id is None:
        logger.warning("No hf_repo_id configured; nothing to download")
        return []

    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    downloaded = [
        pull_checkpoint(config.hf_repo_id, name, destination=config.checkpoint_dir)
        for name in (config.query_tower_path.name, config.document_tower_path.name)
    ]
    return [path for path in downloaded if path is not None]


def main() -> None:
    """Entry point for downloading trained artifacts."""
    paths = download_artifacts(Config())
    logger.info("Downloaded %d artifacts", len(paths))


if __name__ == "__main__":
    main()
