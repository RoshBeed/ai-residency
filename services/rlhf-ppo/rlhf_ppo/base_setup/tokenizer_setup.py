"""Tokenizer setup for batched decoder-only generation."""

from typing import Any

from transformers import AutoTokenizer

__all__ = ["TokenizerSetup"]


class TokenizerSetup:
    """Loads the tokenizer and configures it for left-padded batched generation."""

    def __init__(self, model_name: str) -> None:
        """Load and configure the tokenizer.

        Args:
            model_name: Hugging Face model id to load the tokenizer from.
        """
        self.tokenizer: Any = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # This is a decoder-only model batched through `generate()` (see
        # ppo_trainer/generator.py). Right-padding, the default, leaves a shorter
        # prompt's real tokens sitting before a block of padding that a decoder-only
        # model was never trained to attend past, which garbles that prompt's
        # generation while the longest prompt in the batch looks fine. Left-padding
        # keeps every prompt's real content immediately before the first generated
        # token regardless of its length.
        self.tokenizer.padding_side = "left"
