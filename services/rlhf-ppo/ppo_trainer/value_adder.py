"""Attaching value estimates to (state, action) pairs."""

from typing import Any

import torch

from models import ValueModel
from sa_pair import SAPair

__all__ = ["ValueAdder"]


class ValueAdder:
    """Runs the value model over every state in a batch of (s, a) pairs."""

    def __init__(
        self,
        value_model: ValueModel,
        tokenizer: Any,
        batch_size: int = 32,
        device: torch.device | None = None,
    ) -> None:
        """Wire up the value model.

        Args:
            value_model: The model predicting V(s).
            tokenizer: Shared tokenizer.
            batch_size: How many states to score at once.
            device: Device to run on. Defaults to wherever the model already lives.
        """
        self.value_model = value_model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.device = device if device is not None else next(value_model.parameters()).device

    def add_values_to_sa_pairs(self, sa_pairs: list[SAPair]) -> list[SAPair]:
        """Add a `value` key to every pair, in place.

        Args:
            sa_pairs: Pairs carrying `state_text`.

        Returns:
            The same list, with `value` set on every pair.
        """
        state_texts = [pair["state_text"] for pair in sa_pairs]
        values: list[float] = []

        for start in range(0, len(state_texts), self.batch_size):
            batch_texts = state_texts[start : start + self.batch_size]
            inputs = self.tokenizer(
                batch_texts, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)

            with torch.no_grad():
                batch_values = self.value_model(inputs["input_ids"], inputs["attention_mask"])

            values.extend(float(value) for value in batch_values.reshape(-1).tolist())

        for pair, value in zip(sa_pairs, values, strict=True):
            pair["value"] = value

        return sa_pairs
