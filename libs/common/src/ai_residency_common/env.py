"""Environment loading from a `.env` file.

Credentials in this portfolio (`HF_TOKEN`, `WANDB_API_KEY`) live in a `.env` file
rather than a shell profile, so that a training box and an inference box each carry
their own and nothing has to be exported by hand before `make train`.

Every `.env` from the working directory up to the filesystem root is loaded, nearest
first. That layering is the point: on a machine where these repos sit side by side
under one parent, a single `.env` beside them covers all of them, while a repo that
needs something of its own can add a `.env` holding just that.

Stopping at the first file found would be the obvious implementation and is wrong.
One repo here has a `.env` with only its Redis settings in it, and that file silently
shadowed the shared credentials one directory up — training ran, reported no Hugging
Face token, and kept its checkpoint local, in the one repo out of six where that was
hard to notice.

Nothing here overrides a variable that is already set, so nearest-first means nearest
wins, and real environment beats every file. That last part is what makes this safe in
containers, where the values arrive from compose's `environment:` block or a
Kubernetes Secret and no `.env` file exists at all.
"""

from pathlib import Path

from dotenv import load_dotenv

from ai_residency_common.logging import get_logger

__all__ = ["load_env"]

logger = get_logger(__name__)


def load_env() -> list[Path]:
    """Load every `.env` file from the working directory upwards.

    Called once on package import. Nearer files take precedence, and any variable
    already set in the environment is left alone.

    Returns:
        The files that were loaded, nearest first. Empty if there were none.
    """
    loaded: list[Path] = []
    directory = Path.cwd().resolve()

    for candidate in (directory, *directory.parents):
        path = candidate / ".env"
        if path.is_file():
            load_dotenv(path)
            loaded.append(path)

    if loaded:
        logger.debug("Loaded environment from %s", ", ".join(str(path) for path in loaded))
    return loaded
