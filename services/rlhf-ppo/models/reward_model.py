"""The reward model: scores a completed sequence."""

import torch
from torch import nn

from base_setup import BaseModelSetup
from models.base_model import BaseModel

__all__ = ["RewardModel"]


class RewardModel(BaseModel):
    """Predicts a scalar reward for a prompt plus its completion.

    Structurally this is the standard RLHF reward model: a scalar head over the last
    real token's hidden state. It is not what the live PPO pipeline scores with,
    though: `PPOTrainer` uses `ppo_trainer.RewardComputer`, which wraps a pretrained
    OpenAssistant reward model (`OpenAssistant/reward-model-deberta-v3-large-v2`). This
    class is kept for the component tests and as the reference implementation of the
    from-scratch scalar-head reward model.
    """

    def __init__(self, base_model_setup: BaseModelSetup) -> None:
        """Build the wrapped model and its scalar head.

        Args:
            base_model_setup: The shared factory to build the base model from.
        """
        super().__init__(base_model_setup)
        self.value_head = nn.Linear(self.model.config.hidden_size, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return one reward per sequence in the batch.

        Args:
            input_ids: Token ids of prompt plus completion, left-padded.
            attention_mask: Mask marking real tokens against padding.

        Returns:
            A scalar reward per sequence, shaped (batch,).
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
        reward: torch.Tensor = self.value_head(last_token_hidden).squeeze(-1)
        return reward
