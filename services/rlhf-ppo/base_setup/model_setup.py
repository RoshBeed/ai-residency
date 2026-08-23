"""Loading the base model and wrapping it in LoRA adapters."""

from typing import Any

import torch
from peft import get_peft_model
from transformers import AutoModelForCausalLM

from base_setup.lora_setup import LoRASetup

__all__ = ["ModelSetup"]


class ModelSetup:
    """Creates LoRA-wrapped copies of the base causal LM."""

    def __init__(self, model_name: str, lora_setup: LoRASetup) -> None:
        """Record what to load and how to adapt it.

        Args:
            model_name: Hugging Face model id of the base model.
            lora_setup: The adapter configuration to wrap it with.
        """
        self.model_name = model_name
        self.lora_setup = lora_setup

    def create_base_model(self) -> Any:
        """Load the base model in float32 and return it wrapped in LoRA adapters.

        Returns:
            A PEFT model ready to be used as one of the three PPO base-model roles.
        """
        # Some `transformers` checkpoints load in bfloat16 by default, driven by the
        # checkpoint's own dtype metadata (Qwen3 was the original culprit). The reward
        # and value heads added on top are plain float32 `nn.Linear` layers, so an
        # implicit bf16 load crashes the first time either head is called:
        # `RuntimeError: mat1 and mat2 must have the same dtype`. Force float32
        # explicitly rather than relying on the loader's default.
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            dtype=torch.float32,
        )
        return get_peft_model(base_model, self.lora_setup.lora_config)
