"""Generation, and the (state, action) pairs it produces."""

from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F

from ai_residency_common import get_logger
from config import PPOConfig
from models import PolicyModel, SFTModel
from sa_pair import SAPair

__all__ = ["SequenceGenerator"]

logger = get_logger(__name__)


class SequenceGenerator:
    """Samples completions and records the trajectory PPO needs from them."""

    def __init__(
        self,
        policy_model: PolicyModel,
        sft_model: SFTModel,
        tokenizer: Any,
        config: PPOConfig,
        device: torch.device,
    ) -> None:
        """Wire up the two models generation reads from.

        Args:
            policy_model: The model being trained, which does the sampling.
            sft_model: The frozen reference model, for the KL term.
            tokenizer: Shared tokenizer.
            config: Run configuration.
            device: Device to run on.
        """
        self.policy_model = policy_model
        self.sft_model = sft_model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device

    def _token_logprob(
        self,
        model: PolicyModel | SFTModel,
        state_ids: torch.Tensor,
        state_mask: torch.Tensor,
        action_id: int,
    ) -> float:
        """Return `log p(action | state)` from an unwarped forward pass.

        Args:
            model: The model to score under.
            state_ids: Token ids of the state, shaped (state_len,).
            state_mask: Attention mask for those ids, shaped (state_len,). Required,
                not optional: prompts are left-padded to a common length, so for every
                prompt except the longest in the batch the state begins with padding
                that must not be attended to.
            action_id: Vocabulary id of the action actually taken.

        Returns:
            The log-probability of that action under that model.
        """
        with torch.no_grad():
            outputs = model(state_ids.unsqueeze(0), attention_mask=state_mask.unsqueeze(0))
            log_probs = F.log_softmax(outputs.logits[0, -1, :], dim=-1)
            return float(log_probs[action_id].item())

    def generate_sequences(
        self,
        prompts: list[str],
        on_token: Callable[[int, str, float], None] | None = None,
    ) -> tuple[list[SAPair], list[str]]:
        """Generate a completion per prompt and extract one (s, a) pair per token.

        Args:
            prompts: The prompts to generate from.
            on_token: Optional callback invoked after each token is generated, as
                ``on_token(step, action_token, logprob)``. Lets a caller stream
                progress to a client while generation runs rather than only
                reporting the finished batch. ``logprob`` is the policy model's
                score for the token actually sampled, so the caller can shade the
                chip immediately.

        Returns:
            Every prompt's (s, a) pairs concatenated, and the generated text per prompt.
        """
        logger.info(
            "Generating %d completions of up to %d tokens",
            len(prompts),
            self.config.max_new_tokens,
        )

        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(
            self.device
        )

        with torch.no_grad():
            outputs = self.policy_model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
                temperature=self.config.temperature,
                return_dict_in_generate=True,
                output_scores=True,
            )

        sequences = outputs.sequences
        num_generated = len(outputs.scores)
        prompt_len = inputs["input_ids"].shape[1]

        all_sa_pairs: list[SAPair] = []
        generated_texts: list[str] = []

        # Every generated token is real, so the mask over a state is the prompt's own
        # mask (which marks left padding) followed by ones.
        generated_mask = torch.ones(
            (sequences.shape[0], num_generated),
            dtype=inputs["attention_mask"].dtype,
            device=self.device,
        )
        full_mask = torch.cat([inputs["attention_mask"], generated_mask], dim=1)

        for prompt_idx in range(sequences.shape[0]):
            generated_tokens: list[str] = []

            for step in range(num_generated):
                # State: the prompt plus everything generated so far.
                state_ids = sequences[prompt_idx, : prompt_len + step]
                state_mask = full_mask[prompt_idx, : prompt_len + step]
                state_text: str = self.tokenizer.decode(state_ids, skip_special_tokens=True)
                # The ids the forward pass below actually scores, with the prompt's
                # left padding dropped. `state_text` stays for display and for the
                # value model; the PPO ratio uses these, so that both sides of it
                # score the same tokens rather than the same string.
                real_state_ids: list[int] = state_ids[state_mask.bool()].tolist()

                # Action: the token that was actually sampled at this step.
                action_id = int(sequences[prompt_idx, prompt_len + step].item())
                action_token: str = self.tokenizer.decode([action_id])
                generated_tokens.append(action_token)

                # Both log-probabilities come from a plain forward pass over the state,
                # never from `generate()`'s own `scores`. Those scores are the
                # *post-warping* distribution, after temperature, top-k and top-p have
                # zeroed out excluded tokens, and using them was a real bug rather than
                # a stylistic choice: `scores[step][prompt_idx][action_id]` came back as
                # -inf for a token `generate()` had just sampled, because the top-p
                # mask and the categorical draw disagreed at the nucleus boundary. That
                # -inf then cascaded through the KL term, GAE and both losses as
                # inf/-inf for the rest of the sequence. An unwarped forward pass is
                # also the same quantity `PolicyRatio.get_new_logprobs` recomputes
                # during the update, so the two sides of the PPO ratio are comparable,
                # and `log_softmax` of a real forward pass can never be -inf for a real
                # token.
                #
                # The attention mask matters for that comparability, and its absence
                # here used to break it. Without it, the state's leading pad tokens were
                # attended to, while `get_new_logprobs` re-tokenizes the decoded state
                # text, which has no padding in it. The two sides then measured
                # different distributions for every prompt except the longest in the
                # batch: on this repo's own two example prompts the PPO ratio reached
                # 2.38 before a single gradient step had been taken, when it should be
                # exactly 1.
                logprob = self._token_logprob(self.policy_model, state_ids, state_mask, action_id)
                ref_logprob = self._token_logprob(self.sft_model, state_ids, state_mask, action_id)

                all_sa_pairs.append(
                    {
                        "state_text": state_text,
                        "state_token_ids": real_state_ids,
                        "action_token": action_token,
                        "action_token_id": action_id,
                        "logprob": logprob,
                        "ref_logprob": ref_logprob,
                        "prompt_idx": prompt_idx,
                        "step": step,
                    }
                )
                if on_token is not None:
                    on_token(step, action_token, logprob)

            generated_texts.append("".join(generated_tokens))

        return all_sa_pairs, generated_texts
