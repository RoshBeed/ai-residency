"""The value model's regression loss."""

import torch
import torch.nn.functional as F

__all__ = ["ValueLoss"]


class ValueLoss:
    """Computes the value function loss."""

    @staticmethod
    def mse_loss(predicted_values: torch.Tensor, target_values: torch.Tensor) -> torch.Tensor:
        """Return the mean squared error between predicted values and returns.

        Args:
            predicted_values: The value model's estimate of V(s) per state.
            target_values: Reward-to-go for those states.

        Returns:
            A scalar loss to minimise.
        """
        return F.mse_loss(predicted_values, target_values)
