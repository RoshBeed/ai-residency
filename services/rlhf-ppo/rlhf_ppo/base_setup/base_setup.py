"""One place to build the tokenizer and the LoRA-wrapped base model."""

from typing import Any

from config import PPOConfig
from rlhf_ppo.base_setup.lora_setup import LoRASetup
from rlhf_ppo.base_setup.model_setup import ModelSetup
from rlhf_ppo.base_setup.tokenizer_setup import TokenizerSetup

__all__ = ["BaseModelSetup"]


class BaseModelSetup:
    """Builds the shared tokenizer and hands out LoRA-wrapped base models.

    All four PPO roles are constructed from one of these, so they agree on the
    tokenizer, the padding side and the adapter configuration by construction rather
    than by four separate call sites happening to pass the same arguments.
    """

    def __init__(self, config: PPOConfig) -> None:
        """Build the tokenizer, adapter config and model factory from a run config.

        Args:
            config: The run configuration.
        """
        self.config = config
        self.tokenizer_setup = TokenizerSetup(config.model_name)
        self.lora_setup = LoRASetup(
            r=config.lora_r,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout,
        )
        self.model_setup = ModelSetup(config.model_name, self.lora_setup)

    @property
    def tokenizer(self) -> Any:
        """The shared tokenizer."""
        return self.tokenizer_setup.tokenizer

    def create_base_model(self) -> Any:
        """Return a fresh LoRA-wrapped copy of the base model."""
        return self.model_setup.create_base_model()
