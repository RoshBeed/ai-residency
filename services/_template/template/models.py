"""Model definitions.

One model class per file section, named for what it is rather than for the exercise it
came from. A real repo replaces this with its own architecture.
"""

import torch
from torch import nn

from config import Config

__all__ = ["Predictor"]


class Predictor(nn.Module):
    """A small feed-forward regressor.

    Attributes:
        network: The layer stack applied in `forward`.
    """

    def __init__(self, config: Config) -> None:
        """Build the network from the run configuration.

        Args:
            config: Supplies `input_dim` and `hidden_dim`.
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict a value for each example.

        Args:
            features: Batch of shape `(batch_size, input_dim)`.

        Returns:
            Predictions of shape `(batch_size, 1)`.
        """
        output: torch.Tensor = self.network(features)
        return output
