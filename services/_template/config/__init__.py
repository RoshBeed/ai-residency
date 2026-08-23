"""Run configuration.

Every repo declares its own `Config` here, inheriting the fields shared across the
portfolio from `BaseConfig` (seed, batch_size, learning_rate, num_epochs,
checkpoint_dir, data_dir, hf_repo_id, wandb_project) and adding only what is
specific to its own model.

Follow the naming spec for new fields: spell hyperparameters out (`hidden_dim`, not
`hdim`) so the same concept reads the same way in every repo.
"""

from dataclasses import dataclass

from ai_residency_common import BaseConfig

__all__ = ["Config"]


@dataclass
class Config(BaseConfig):
    """Configuration for this repo's model and training run.

    Attributes:
        input_dim: Width of a single input example.
        hidden_dim: Width of the hidden layer.
        num_examples: Size of the generated demo dataset.
        checkpoint_name: File name for the best checkpoint, under `checkpoint_dir`.
    """

    input_dim: int = 16
    hidden_dim: int = 32
    num_examples: int = 512
    checkpoint_name: str = "best_model.pth"

    # Set these per repo to enable artifact hosting and metric tracking. Left None
    # here so the template runs offline out of the box.
    hf_repo_id: str | None = None
    wandb_project: str | None = None
