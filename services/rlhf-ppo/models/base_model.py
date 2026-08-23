"""The shared base every PPO model role is built on."""

from torch import nn

from base_setup import BaseModelSetup

__all__ = ["BaseModel"]


class BaseModel(nn.Module):
    """A LoRA-wrapped copy of the base causal LM, plus the shared tokenizer.

    PPO needs four models that start life identical and then diverge: the frozen
    reference, the policy being trained, the reward scorer and the value estimator.
    Each subclass adds only what its role needs on top of this.
    """

    def __init__(self, base_model_setup: BaseModelSetup) -> None:
        """Build a fresh LoRA-wrapped model for one PPO role.

        Args:
            base_model_setup: The shared factory to build it from.
        """
        super().__init__()
        self.model = base_model_setup.create_base_model()
        self.tokenizer = base_model_setup.tokenizer
