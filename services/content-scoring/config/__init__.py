"""Run configuration.

The scoring stage depends on word embeddings trained in the sibling
`word-embeddings` service: the frozen CBOW vectors (and the
vocabulary that indexes them) are pulled from that repo's Hugging Face repo rather
than reproduced here. Only the regression head's own hyperparameters live in this
repo.

Paths are relative to the repo root, which is where every `make` target runs from.
"""

from dataclasses import dataclass, field
from pathlib import Path

from ai_residency_common import BaseConfig

__all__ = ["Config"]


@dataclass
class Config(BaseConfig):
    """Configuration for the content-score regression head.

    Attributes:
        num_epochs: Maximum epochs; early stopping usually ends it first.
        patience: Epochs without improvement before stopping.
        weight_decay: Adam weight decay.
        cbow_repo_id: Hugging Face repo holding the frozen CBOW checkpoint, owned by
            the word-embeddings repo. None keeps the local copy at `cbow_checkpoint`
            as the only source.
        cbow_checkpoint: Where the CBOW checkpoint lives (or is downloaded to) locally.
        posts_file: Cleaned posts with their scores, from the shared dataset repo.
        checkpoint_name: File name for the best scorer, under `checkpoint_dir`.
        info_name: File name for the JSON sidecar describing that checkpoint.
    """

    num_epochs: int = 50
    patience: int = 10
    weight_decay: float = 1e-5

    cbow_repo_id: str | None = "roshbeed/ai-residency-word-embeddings"
    cbow_checkpoint: Path = field(default_factory=lambda: Path("checkpoints/cbow/cbow_model.pt"))
    posts_file: Path = field(default_factory=lambda: Path("data/hn_data_cleaned.json"))

    checkpoint_name: str = "best_predictor.pth"
    info_name: str = "predictor_info.json"

    hf_repo_id: str | None = "roshbeed/ai-residency-content-scoring"
    # The labelled Hacker News posts live here rather than in git. They came from a
    # Postgres mirror that is not reachable from every network, so they are the one
    # input this repo cannot simply fetch or regenerate.
    hf_dataset_repo_id: str | None = "roshbeed/ai-residency-word-embeddings-data"
    wandb_project: str | None = "ai-residency-content-scoring"

    @property
    def info_path(self) -> Path:
        """Full path to the checkpoint's JSON sidecar."""
        return self.checkpoint_dir / self.info_name
