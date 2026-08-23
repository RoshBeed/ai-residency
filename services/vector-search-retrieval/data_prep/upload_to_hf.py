"""Push trained artifacts to the Hugging Face Hub.

Everything this repo trains is gitignored, so a fresh clone has no usable model in it.
Pushing the artifacts to the Hub is what makes the repo runnable somewhere other than
the machine it was trained on.

This used to be a hundred lines of hand-rolled `HfApi` calls uploading source files,
`requirements.txt` and paths like `src/cbow/checkpoints/*.pth` that no longer exist. It
delegates to `ai_residency_common.push_checkpoint` now, which every repo in the portfolio shares,
and uploads only the artifacts — the code is already in git.
"""

from ai_residency_common import get_logger, push_checkpoint
from cbow.vocabulary import CANONICAL_CBOW_CHECKPOINT, latest_cbow_checkpoint
from config import Config

__all__ = ["main", "upload_artifacts"]

logger = get_logger(__name__)


def upload_artifacts(config: Config) -> int:
    """Upload the CBOW embeddings and both trained towers.

    Args:
        config: Supplies `hf_repo_id` and the artifact paths.

    Returns:
        How many files were uploaded. Zero means no token was configured, which
        `push_checkpoint` warns about rather than raising.
    """
    if config.hf_repo_id is None:
        logger.warning("No hf_repo_id configured; nothing to upload")
        return 0

    uploaded = 0

    # Under its canonical name, not the run-timestamped one the trainer writes
    # locally: a downstream host has to be able to ask for it by name. Missing is
    # not fatal -- only tower training reads it, and the towers are the artifacts
    # that matter here, so a host that has those should still be able to push them.
    try:
        uploaded += push_checkpoint(
            config.hf_repo_id,
            latest_cbow_checkpoint(config),
            path_in_repo=CANONICAL_CBOW_CHECKPOINT,
        )
    except FileNotFoundError:
        logger.warning("No CBOW checkpoint to upload; skipping")

    for path in (config.query_tower_path, config.document_tower_path):
        if path.exists():
            uploaded += push_checkpoint(config.hf_repo_id, path)

    return uploaded


def main() -> None:
    """Entry point for uploading trained artifacts."""
    uploaded = upload_artifacts(Config())
    logger.info("Uploaded %d artifacts", uploaded)


if __name__ == "__main__":
    main()
