"""A flat buffer of the fields a PPO update actually reads."""

import torch

from sa_pair import SAPair

__all__ = ["ExperienceBuffer"]


class ExperienceBuffer:
    """Holds the subset of each (s, a) pair a policy update needs.

    Note that `PPOTrainer.ppo_step` does not currently use this: it trains directly
    on the `sa_pairs` list it just built, in one pass, so there is nothing to buffer
    between steps. The class exists because a real multi-epoch PPO loop would reuse
    the same batch of experience for several optimisation epochs, and it is exercised
    by the tests. See TODO.md.
    """

    def __init__(self) -> None:
        """Start with an empty buffer."""
        self.buffer: list[SAPair] = []

    def add_experience(self, sa_pairs: list[SAPair]) -> None:
        """Append the training-relevant fields of each pair.

        `logprob` is stored as `old_logprob` because that is what it becomes the
        moment the policy is updated: the fixed reference point the PPO ratio is
        measured against.

        Args:
            sa_pairs: Fully populated pairs from a completed PPO step.
        """
        for pair in sa_pairs:
            self.buffer.append(
                {
                    "state_text": pair["state_text"],
                    # The ids, not just the text: a replayed experience is scored by
                    # PolicyRatio, which needs the exact tokens the generator scored.
                    "state_token_ids": list(pair["state_token_ids"]),
                    "action_token_id": pair["action_token_id"],
                    "old_logprob": pair["logprob"],
                    "advantage": pair["advantage"],
                    "reward_to_go": pair["reward_to_go"],
                }
            )

    def get_batch(self, batch_size: int) -> list[SAPair]:
        """Return a random batch, or everything if the buffer is smaller than that.

        Args:
            batch_size: How many experiences to draw.

        Returns:
            The sampled experiences.
        """
        if len(self.buffer) < batch_size:
            return self.buffer
        indices = torch.randperm(len(self.buffer))[:batch_size].tolist()
        return [self.buffer[index] for index in indices]

    def clear(self) -> None:
        """Drop everything currently buffered."""
        self.buffer = []
