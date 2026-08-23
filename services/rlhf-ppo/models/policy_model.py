"""The policy model: the one PPO actually trains."""

from typing import Any

import torch
import torch.nn.functional as F

from models.base_model import BaseModel

__all__ = ["PolicyModel"]


class PolicyModel(BaseModel):
    """The model being optimised by PPO."""

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
        """Return the log-probability distribution over the next token.

        Args:
            input_ids: Token ids of the state.
            attention_mask: Mask marking real tokens against padding.

        Returns:
            Log-probabilities over the vocabulary, shaped (batch, vocab).
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return F.log_softmax(outputs.logits[:, -1, :], dim=-1)

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate sampling to the wrapped model's `generate`."""
        return self.model.generate(*args, **kwargs)
