"""Building (state, action) pairs from a tokenised generation."""

from typing import Any

from sa_pair import SAPair

__all__ = ["SAPairBuilder"]


class SAPairBuilder:
    """Turns one prompt and its generated tokens into per-step (s, a) pairs."""

    def __init__(self, tokenizer: Any) -> None:
        """Store the tokenizer to use.

        Args:
            tokenizer: A Hugging Face tokenizer.
        """
        self.tokenizer = tokenizer

    def build_sa_pairs(
        self,
        prompt: str,
        gen_tokens: list[str],
        gen_token_ids: list[int],
        policy_logprobs: list[float],
        ref_logprobs: list[float],
        prompt_idx: int,
    ) -> list[SAPair]:
        """Return one pair per generated token.

        The state grows as generation proceeds: step `t`'s state is the prompt plus
        every token generated before `t`, and the action is the token generated at
        `t`. That is what makes token-by-token generation an RL trajectory rather
        than a single decision.

        Args:
            prompt: The prompt the sequence started from.
            gen_tokens: Generated token strings, in order.
            gen_token_ids: Vocabulary ids of those tokens.
            policy_logprobs: Policy log-probability of each generated token.
            ref_logprobs: Reference model's log-probability of each generated token.
            prompt_idx: Index of this prompt within the batch.

        Returns:
            One (s, a) pair per generated token.
        """
        sa_pairs: list[SAPair] = []
        state = prompt

        for step, (token, token_id, policy_logprob, ref_logprob) in enumerate(
            zip(gen_tokens, gen_token_ids, policy_logprobs, ref_logprobs, strict=True)
        ):
            sa_pairs.append(
                {
                    "state_text": state,
                    "action_token": token,
                    "action_token_id": token_id,
                    "logprob": policy_logprob,
                    "ref_logprob": ref_logprob,
                    "prompt_idx": prompt_idx,
                    "step": step,
                }
            )
            state += token

        return sa_pairs
