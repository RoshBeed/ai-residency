"""Checkpoint hosting on the Hugging Face Hub.

Trained weights are gitignored in every repo here, which means a fresh clone or a
submodule checkout has no usable model in it. The portfolio hub worked around that
by bind-mounting checkpoint paths from sibling repo directories on one specific
machine, which is why it could not be deployed anywhere else. Pushing weights to the
Hub removes that workaround: any checkout can pull what it needs.

Every function degrades to a no-op with a warning when no token is configured, so
`make train` still works for someone who has cloned the repo without these accounts.
"""

import os
from pathlib import Path
from typing import Literal, NamedTuple

from huggingface_hub import HfApi, get_token, hf_hub_download

from ai_residency_common.logging import get_logger

__all__ = [
    "RepoType",
    "ResolvedCheckpoint",
    "get_hf_token",
    "pull_checkpoint",
    "push_checkpoint",
    "resolve_checkpoint",
]

logger = get_logger(__name__)

_TOKEN_VARS = ("HF_TOKEN", "HUGGINGFACE_TOKEN")

# The Hub splits weights from the data that produced them. Both matter here for the
# same reason: a machine that did not run the training still needs them.
RepoType = Literal["model", "dataset"]


class ResolvedCheckpoint(NamedTuple):
    """Where a service's weights came from.

    Attributes:
        path: The usable checkpoint file, or None if there was no way to get one.
        source: `"local"`, `"hub"`, or `"none"`. Worth reporting from a health check:
            a service that silently fell back to an untrained model still answers
            requests, just with nonsense, and that is hard to spot from the outside.
    """

    path: Path | None
    source: Literal["local", "hub", "none"]


def get_hf_token() -> str | None:
    """Return the Hugging Face token, from the environment or the local Hub login.

    Checks `HF_TOKEN` then `HUGGINGFACE_TOKEN`, since repos in this portfolio have
    historically used both names, and finally the credential `huggingface-cli login`
    writes to `~/.cache/huggingface/token`.

    That last fallback is not a convenience. Without it, `push_checkpoint` returns
    False and warns on a machine whose owner is demonstrably logged in -- and the
    warning scrolls past in the middle of a training run, so what you notice is that
    hours of compute produced nothing on the Hub. Every other Hub call in this
    package already picks the cached token up automatically; only ours did not.

    Returns:
        The token, or None when there is no environment variable and no local login.
    """
    for var in _TOKEN_VARS:
        token = os.environ.get(var)
        if token:
            return token
    try:
        return get_token()
    except Exception:  # pragma: no cover - a corrupt token file must not stop training
        return None


def push_checkpoint(
    repo_id: str,
    path: str | Path,
    *,
    path_in_repo: str | None = None,
    private: bool = False,
    repo_type: RepoType = "model",
) -> bool:
    """Upload a checkpoint file to a model repo on the Hub.

    Intended to be called only when a new best checkpoint is written, not on every
    epoch: these artifacts are large (one is ~1.2GB) and re-uploading each
    improvement during a sweep would dominate training time.

    Args:
        repo_id: Target model repo, e.g. `"roshbeed/ai-residency-word-embeddings"`.
        path: Local checkpoint file to upload.
        path_in_repo: Name to store it under. Defaults to the local file name.
        private: Whether to create the repo private if it does not yet exist. Defaults
            to False, because serving pulls these with no credentials at all: a repo
            created private is one every container reports as `Repository Not Found`,
            and the push that created it looks like a success. Only the first push for
            a repo is affected -- this does not change the visibility of an existing
            one.
        repo_type: `"model"` for weights, `"dataset"` for the inputs that produced
            them. Both are files this run needs; the Hub just files them differently.

    Returns:
        True if the upload happened, False if it was skipped for lack of a token.
    """
    token = get_hf_token()
    if token is None:
        logger.warning(
            "No Hugging Face token (set %s); keeping %s local only",
            " or ".join(_TOKEN_VARS),
            path,
        )
        return False

    path = Path(path)
    api = HfApi(token=token)
    try:
        api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path_in_repo or path.name,
            repo_id=repo_id,
            repo_type=repo_type,
        )
    except Exception as error:
        # The checkpoint is already on disk by the time this runs, so a failed upload
        # costs nothing that is not recoverable by pushing again. Raising here would
        # throw away a finished training run over a credentials problem, which on a
        # rented GPU box means paying for the run twice. Loud, but not fatal.
        logger.error("Could not push %s to %s: %s", path.name, repo_id, error)
        return False

    logger.info("Pushed %s to https://huggingface.co/%s", path.name, repo_id)
    return True


def pull_checkpoint(
    repo_id: str,
    filename: str,
    *,
    destination: str | Path | None = None,
    repo_type: RepoType = "model",
) -> Path | None:
    """Download a checkpoint from the Hub.

    Args:
        repo_id: Source model repo.
        filename: File to fetch from that repo.
        destination: Directory to place it in. Defaults to the Hub cache.
        repo_type: `"model"` or `"dataset"`.

    Returns:
        Path to the downloaded file, or None if the download was not possible.
    """
    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            token=get_hf_token(),
            local_dir=str(destination) if destination is not None else None,
        )
    except Exception as error:
        logger.warning("Could not pull %s from %s: %s", filename, repo_id, error)
        return None

    logger.info("Pulled %s from %s", filename, repo_id)
    return Path(local_path)


def resolve_checkpoint(
    path: str | Path,
    repo_id: str | None,
    *,
    filename: str | None = None,
    repo_type: RepoType = "model",
) -> ResolvedCheckpoint:
    """Return a usable checkpoint, downloading it from the Hub if it is missing.

    This is what lets a service run somewhere other than the machine that trained
    it. Weights are gitignored in every repo here, so a fresh clone, a submodule
    checkout, or an inference host separate from the training host all start with no
    local file; before this, each of those could only serve an untrained model.

    Args:
        path: Where the checkpoint lives locally, and where a downloaded one is put.
        repo_id: Model repo to fall back to. None means local-only.
        filename: Name of the file within that repo. Defaults to `path`'s own name.
        repo_type: `"model"` or `"dataset"`. Training inputs that cannot simply be
            downloaded again belong in a dataset repo, for the same reason weights
            belong in a model one: a host that did not produce them can still get them.

    Returns:
        The resolved path and where it came from.
    """
    path = Path(path)
    if path.exists():
        return ResolvedCheckpoint(path, "local")

    if repo_id is None:
        logger.warning("No checkpoint at %s and no hf_repo_id configured", path)
        return ResolvedCheckpoint(None, "none")

    logger.info("No checkpoint at %s; pulling %s from %s", path, filename or path.name, repo_id)
    pulled = pull_checkpoint(
        repo_id, filename or path.name, destination=path.parent, repo_type=repo_type
    )
    if pulled is None:
        return ResolvedCheckpoint(None, "none")
    return ResolvedCheckpoint(pulled, "hub")
