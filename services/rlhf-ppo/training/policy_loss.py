"""PPO's clipped surrogate objective."""

import torch

__all__ = ["PolicyLoss"]


class PolicyLoss:
    """Computes the PPO-clip policy loss."""

    @staticmethod
    def ppo_clip_loss(
        new_logprobs: torch.Tensor,
        old_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        clip_epsilon: float = 0.2,
    ) -> torch.Tensor:
        """Return the clipped surrogate loss for one batch of (s, a) pairs.

        The ratio `pi_new / pi_old` says how much more likely the current policy is to
        take the action it already took. Multiplying it by the advantage gives the
        surrogate objective; clipping it removes the gradient once the ratio moves
        outside `[1 - eps, 1 + eps]`, which is what keeps a single update from
        destroying the policy. Taking the minimum of the clipped and unclipped terms
        makes the bound pessimistic, so clipping can only ever reduce the objective.

        Args:
            new_logprobs: Log-probabilities under the current policy.
            old_logprobs: Log-probabilities recorded when the actions were sampled.
            advantages: Advantage estimate per pair.
            clip_epsilon: Half-width of the trust region on the ratio.

        Returns:
            A scalar loss to minimise. With mixed-sign advantages this can legitimately
            be negative; there is no positivity invariant here.
        """
        ratio = torch.exp(new_logprobs - old_logprobs)
        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
        return -torch.mean(torch.min(unclipped, clipped))
