"""Recomputing log-probabilities under the current policy."""

from typing import Any

import torch
import torch.nn.functional as F

from models import PolicyModel

__all__ = ["PolicyRatio"]


class PolicyRatio:
    """Scores already-taken actions under the current policy."""

    @staticmethod
    def get_new_logprobs(
        policy_model: PolicyModel,
        tokenizer: Any,
        state_texts: list[str],
        action_token_ids: list[int],
        device: torch.device,
        batch_size: int = 32,
    ) -> torch.Tensor:
        """Return `log pi_new(a | s)` for each (state, action) in the batch.

        These are the numerator of the PPO ratio. They come from a plain forward pass
        over the state, with no sampling warpers applied, which is deliberately the
        same quantity `SequenceGenerator` records as `old_logprob` — the ratio is only
        meaningful if both sides measure the same distribution.

        The result keeps its gradient: this is what the policy loss backpropagates
        through.

        Args:
            policy_model: The policy being trained.
            tokenizer: Tokenizer for the state text.
            state_texts: The state each action was taken from.
            action_token_ids: The action taken at each state.
            device: Device to run on.
            batch_size: How many states to score at once.

        Returns:
            One log-probability per (state, action), shaped (len(state_texts),).
        """
        new_logprobs: list[torch.Tensor] = []

        for start in range(0, len(state_texts), batch_size):
            batch_states = state_texts[start : start + batch_size]
            batch_action_ids = action_token_ids[start : start + batch_size]

            inputs = tokenizer(batch_states, return_tensors="pt", padding=True, truncation=True).to(
                device
            )
            outputs = policy_model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
            log_probs = F.log_softmax(outputs.logits[:, -1, :], dim=-1)

            action_ids = torch.tensor(batch_action_ids, dtype=torch.long, device=device)
            new_logprobs.append(log_probs[torch.arange(len(batch_states)), action_ids])

        return torch.cat(new_logprobs, dim=0)
