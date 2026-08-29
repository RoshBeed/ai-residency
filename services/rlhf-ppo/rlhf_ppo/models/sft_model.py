"""The reference model the policy is kept close to."""

from typing import Any

import torch
import torch.nn.functional as F

from rlhf_ppo.models.base_model import BaseModel

__all__ = ["SFTModel"]


class SFTModel(BaseModel):
    """The frozen reference model in the KL term.

    In a full RLHF pipeline this would be the supervised fine-tuned checkpoint the
    policy is initialised from. No supervised fine-tuning stage exists in this repo,
    so it is a second copy of the base model instead: the KL penalty still works
    exactly as written, it just measures drift from the base model rather than from
    an SFT model. That is a real gap, documented in the README rather than papered
    over by the class name.
    """

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run a forward pass through the wrapped causal LM.

        Args:
            input_ids: Token ids.
            attention_mask: Mask marking real tokens against padding.
            labels: Optional labels, if a language-modelling loss is wanted.
            **kwargs: Passed straight through to the wrapped model.

        Returns:
            The wrapped model's output object, including `logits`.
        """
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs
        )

    def get_logprobs(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return the reference distribution over the next token.

        Gradients are never needed here: this model is frozen, it only supplies the
        reference term in the KL penalty.

        Args:
            input_ids: Token ids of the state.
            attention_mask: Mask marking real tokens against padding.

        Returns:
            Log-probabilities over the vocabulary, shaped (batch, vocab).
        """
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            return F.log_softmax(outputs.logits[:, -1, :], dim=-1)
