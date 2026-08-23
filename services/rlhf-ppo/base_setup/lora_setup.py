"""LoRA adapter configuration."""

from typing import Any

from peft import LoraConfig

__all__ = ["LoRASetup"]


class LoRASetup:
    """Holds the LoRA configuration the base model is wrapped with.

    LoRA is what makes running four copies of a 0.6B model on one machine practical:
    the frozen base weights are shared and only the small adapter matrices are
    trained, so a PPO update touches a few million parameters rather than 600
    million.
    """

    def __init__(self, r: int = 16, alpha: int = 32, dropout: float = 0.1) -> None:
        """Build the adapter configuration.

        Args:
            r: Rank of the low-rank update. Higher means more adapter capacity.
            alpha: Scaling applied to the adapter output.
            dropout: Dropout inside the adapters.
        """
        self.lora_config: Any = LoraConfig(
            r=r,
            lora_alpha=alpha,
            # All four attention projections, so the adapter can reshape attention
            # itself rather than only rescale its output.
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
