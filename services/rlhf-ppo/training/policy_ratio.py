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
        state_token_ids: list[list[int]],
        action_token_ids: list[int],
        device: torch.device,
        batch_size: int = 32,
    ) -> torch.Tensor:
        """Return `log pi_new(a | s)` for each (state, action) in the batch.

        These are the numerator of the PPO ratio. They come from a plain forward pass
        over the state, with no sampling warpers applied, which is deliberately the
        same quantity `SequenceGenerator` records as `old_logprob` — the ratio is only
        meaningful if both sides measure the same distribution.

        The state arrives as token ids rather than text, and that is the whole point.
        This used to take `state_text` and re-tokenize it, which quietly assumed that
        decode-then-encode is the identity. It is not: for a low-probability token the
        round trip can come back as a different sequence, so the two sides of the ratio
        scored different states and the ratio was not 1 before any gradient step had
        been taken. Carrying the ids the generator actually scored removes the
        assumption instead of narrowing it.

        The result keeps its gradient: this is what the policy loss backpropagates
        through.

        Args:
            policy_model: The policy being trained.
            tokenizer: Tokenizer, used only for its pad token id.
            state_token_ids: Token ids of the state each action was taken from, with
                no padding — this pads them itself.
            action_token_ids: The action taken at each state.
            device: Device to run on.
            batch_size: How many states to score at once.

        Returns:
            One log-probability per (state, action), shaped (len(state_token_ids),).
        """
        new_logprobs: list[torch.Tensor] = []
        pad_id = tokenizer.pad_token_id

        for start in range(0, len(state_token_ids), batch_size):
            batch_states = state_token_ids[start : start + batch_size]
            batch_action_ids = action_token_ids[start : start + batch_size]

            # Left padding, matching the tokenizer's own `padding_side`. The score is
            # read from `logits[:, -1, :]`, so the state's real last token has to be
            # the last column; right padding would score a pad token instead.
            width = max(len(ids) for ids in batch_states)
            input_ids = torch.full(
                (len(batch_states), width), pad_id, dtype=torch.long, device=device
            )
            attention_mask = torch.zeros(
                (len(batch_states), width), dtype=torch.long, device=device
            )
            for row, ids in enumerate(batch_states):
                input_ids[row, width - len(ids) :] = torch.tensor(
                    ids, dtype=torch.long, device=device
                )
                attention_mask[row, width - len(ids) :] = 1

            outputs = policy_model(input_ids, attention_mask=attention_mask)
            log_probs = F.log_softmax(outputs.logits[:, -1, :], dim=-1)

            action_ids = torch.tensor(batch_action_ids, dtype=torch.long, device=device)
            new_logprobs.append(log_probs[torch.arange(len(batch_states)), action_ids])

        return torch.cat(new_logprobs, dim=0)
