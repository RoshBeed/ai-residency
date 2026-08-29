"""A flat buffer of the fields a PPO update actually reads."""

import torch

from rlhf_ppo.types import SAPair

__all__ = ["ExperienceBuffer"]


class ExperienceBuffer:
    """Holds the subset of each (s, a) pair a policy update needs.

    `PPOTrainer.ppo_step` fills this, reads one batch back out and clears it, all
    within a single step. That looks like a pass-through, and today it is: the buffer
    holds one step's experience rather than accumulating across steps.

    It is a separate class because the shape it imposes is what a multi-epoch update
    needs. PPO's whole premise is reusing one batch of experience for several
    optimisation epochs against a changing policy, and the buffer is where that reuse
    will happen -- keeping the tensor extraction out of the trainer means the loop can
    grow without the trainer learning about storage.
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
