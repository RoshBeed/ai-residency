"""Base configuration and seeding.

Each repo declares its own `Config` dataclass inheriting from `BaseConfig`, so the
fields every training run needs (seed, device, where checkpoints and data live) are
named the same everywhere and typed once.

Any field can be overridden from the environment as `AI_RESIDENCY_<FIELD_NAME>` in upper case
(`AI_RESIDENCY_NUM_EPOCHS=1`). Defaults in these dataclasses describe a real training run, and
hardcoding them meant the only way to shorten a run for a smoke test, or lengthen one
on a rented GPU box, was to edit the source. The override walks the dataclass fields
rather than naming a fixed set, so a repo's own hyperparameters are covered too
without it having to opt in.

`set_seed` exists because the absence of one was a real, repeated bug in this
portfolio rather than a hypothetical: four of the six repos reported numbers from
unseeded runs that could not be reproduced, and in two of them seeding was what
exposed an actual correctness bug underneath.
"""

import os
import random
import types
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints

import torch

from ai_residency_common.device import get_device
from ai_residency_common.logging import get_logger

__all__ = ["BaseConfig", "set_seed"]

logger = get_logger(__name__)

_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})


def _coerce(raw: str, annotation: Any) -> Any:
    """Convert an environment string to the type a config field declares.

    Args:
        raw: The raw environment value.
        annotation: The field's resolved type annotation.

    Returns:
        The converted value.

    Raises:
        ValueError: If the value cannot be converted to the declared type.
    """
    origin = get_origin(annotation)

    # `str | None` and friends: an empty value clears the field, anything else is
    # converted to whichever member of the union is not None.
    if origin in (Union, types.UnionType):
        members = [arg for arg in get_args(annotation) if arg is not type(None)]
        if raw == "" and len(members) < len(get_args(annotation)):
            return None
        return _coerce(raw, members[0])

    # Sequences are comma-separated, which is how a shell can express one.
    if origin in (list, tuple):
        parts = [part.strip() for part in raw.split(",") if part.strip()]
        (item_type,) = (get_args(annotation) or (str,))[:1]
        items = [_coerce(part, item_type) for part in parts]
        return items if origin is list else tuple(items)

    if annotation is bool:
        lowered = raw.strip().lower()
        if lowered in _TRUE:
            return True
        if lowered in _FALSE:
            return False
        raise ValueError(f"{raw!r} is not a boolean")

    if annotation is Path:
        return Path(raw)

    if annotation in (int, float, str):
        return annotation(raw)

    raise ValueError(f"no conversion to {annotation!r}")


def set_seed(seed: int) -> None:
    """Seed Python and PyTorch RNGs so a run is reproducible.

    Seeds `torch`'s global generator, which covers weight initialisation, dropout and
    `random_split`, plus the `random` module. Call this once, before building models
    or splitting data.

    Args:
        seed: The seed to apply.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Seeded RNGs with %d", seed)


@dataclass
class BaseConfig:
    """Fields common to every training run in the portfolio.

    Subclass this per repo and add the model-specific hyperparameters. Follow the
    portfolio naming spec for those: `learning_rate`, `batch_size`, `num_epochs`,
    spelled out rather than abbreviated.

    Attributes:
        seed: RNG seed, applied via `set_seed`.
        batch_size: Examples per batch.
        learning_rate: Optimizer learning rate.
        num_epochs: Maximum training epochs.
        checkpoint_dir: Where trained artifacts are written locally.
        data_dir: Where datasets and generated corpora live.
        hf_repo_id: Model repo to push best checkpoints to, e.g.
            `"roshbeed/ai-residency-week1-word-embeddings"`. None keeps checkpoints local.
        hf_dataset_repo_id: Dataset repo holding training inputs that cannot simply
            be downloaded again — our own extracts and derived corpora, not
            third-party datasets, which have their own homes and licences.
        wandb_project: W&B project to log metrics to. None disables tracking.
    """

    seed: int = 42
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_epochs: int = 10
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    checkpoint_name: str = "best_model.pth"
    data_dir: Path = field(default_factory=lambda: Path("data"))
    hf_repo_id: str | None = None
    hf_dataset_repo_id: str | None = None
    wandb_project: str | None = None

    def __post_init__(self) -> None:
        """Apply any `AI_RESIDENCY_<FIELD_NAME>` overrides present in the environment."""
        hints = get_type_hints(type(self))
        for spec in fields(self):
            raw = os.environ.get(f"AI_RESIDENCY_{spec.name.upper()}")
            if raw is None:
                continue
            try:
                value = _coerce(raw, hints[spec.name])
            except ValueError as error:
                raise ValueError(
                    f"AI_RESIDENCY_{spec.name.upper()}={raw!r} is not valid for "
                    f"{type(self).__name__}.{spec.name}: {error}"
                ) from error
            setattr(self, spec.name, value)
            logger.info("Config override %s=%r from the environment", spec.name, value)

    @property
    def checkpoint_path(self) -> Path:
        """Full path to the best-checkpoint file on local disk."""
        return self.checkpoint_dir / self.checkpoint_name

    @property
    def device(self) -> torch.device:
        """The device this run should use, resolved at access time."""
        return get_device()
