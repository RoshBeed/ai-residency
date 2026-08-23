"""The value model: predicts the return from a state."""

import torch
from torch import nn

from base_setup import BaseModelSetup
from models.base_model import BaseModel

__all__ = ["ValueModel"]


class ValueModel(BaseModel):
    """Estimates V(s), the expected return from a partially generated sequence.

    A scalar head on top of the last real token's hidden state. Its predictions are
    what the TD errors and therefore the advantages are computed against, so it is
    trained on reward-to-go targets in the same PPO step that uses it.
    """

    def __init__(self, base_model_setup: BaseModelSetup) -> None:
        """Build the wrapped model and its scalar head.

        Args:
            base_model_setup: The shared factory to build the base model from.
        """
        super().__init__(base_model_setup)
        self.value_head = nn.Linear(self.model.config.hidden_size, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return one value estimate per sequence in the batch.

        Args:
            input_ids: Token ids, left-padded.
            attention_mask: Mask marking real tokens against padding.

        Returns:
            A value estimate per sequence, shaped (batch,).
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = outputs.hidden_states[-1]
        # Read the hidden state of the last *real* token, which is not the last
        # position once anything in the batch is padded.
        last_index = attention_mask.sum(dim=1) - 1
        gather_index = last_index.unsqueeze(1).unsqueeze(2).expand(-1, 1, last_hidden.size(-1))
        last_token_hidden = last_hidden.gather(1, gather_index).squeeze(1)
        value: torch.Tensor = self.value_head(last_token_hidden).squeeze(-1)
        return value
