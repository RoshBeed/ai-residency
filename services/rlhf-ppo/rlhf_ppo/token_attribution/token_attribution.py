"""Batch-level (state, action) pair extraction."""

from typing import Any

from rlhf_ppo.token_attribution.extractor import TokenExtractor
from rlhf_ppo.token_attribution.sa_pair_builder import SAPairBuilder
from rlhf_ppo.types import SAPair

__all__ = ["TokenAttribution"]


class TokenAttribution:
    """Divides a batch of generated sequences into (state, action) pairs.

    `SequenceGenerator` builds its pairs inline while it walks the generated tokens,
    since it already has the ids and the log-probabilities in hand there. This class
    is the standalone equivalent for pairs recovered from generated *text* after the
    fact -- scoring a completion that arrived from somewhere else, or rebuilding the
    trace for an existing sample.

    Recovering pairs from text is lossy in a way the inline path is not: re-encoding
    a decoded string does not always return the tokens that produced it. The PPO ratio
    reads `state_token_ids` for that reason, so anything derived here is for
    inspection rather than for a gradient.
    """

    def __init__(self, tokenizer: Any) -> None:
        """Build the extractor and pair builder around a tokenizer.

        Args:
            tokenizer: A Hugging Face tokenizer.
        """
        self.tokenizer = tokenizer
        self.extractor = TokenExtractor(tokenizer)
        self.builder = SAPairBuilder(tokenizer)

    def extract_sa_pairs(
        self,
        prompts: list[str],
        generated_texts: list[str],
        policy_logprobs: list[list[float]],
        ref_logprobs: list[list[float]],
    ) -> list[SAPair]:
        """Return the (s, a) pairs for every prompt in a batch.

        Args:
            prompts: The prompts that were generated from.
            generated_texts: The completion for each prompt.
            policy_logprobs: Per-token policy log-probabilities, per prompt.
            ref_logprobs: Per-token reference log-probabilities, per prompt.

        Returns:
            Every prompt's pairs, concatenated.
        """
        all_sa_pairs: list[SAPair] = []

        for prompt_idx, (prompt, generated_text, policy, reference) in enumerate(
            zip(prompts, generated_texts, policy_logprobs, ref_logprobs, strict=True)
        ):
            tokens, token_ids = self.extractor.extract_tokens(generated_text)
            all_sa_pairs.extend(
                self.builder.build_sa_pairs(
                    prompt, tokens, token_ids, policy, reference, prompt_idx
                )
            )

        return all_sa_pairs
