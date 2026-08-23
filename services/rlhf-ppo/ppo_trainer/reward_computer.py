"""Scoring generated sequences with the reward model."""

import torch
from ai_residency_common import get_logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import PPOConfig

__all__ = ["RewardComputer"]

logger = get_logger(__name__)


class RewardComputer:
    """Runs the OpenAssistant DeBERTa-v3 reward model over prompt-plus-completion strings.

    This is an actual trained reward model -- not the untrained in-repo scalar head
    this repo previously used. It was fine-tuned on human preference comparisons from
    four datasets including ``summarize_from_feedback`` (Reddit TL;DR), so its scores
    are a real (if imperfect) quality signal for the PPO pipeline.
    """

    def __init__(self, config: PPOConfig, device: torch.device) -> None:
        """Load the reward model and its own tokenizer.

        Args:
            config: Run configuration, supplying ``reward_model_name``.
            device: Device to run on.
        """
        logger.info("Loading reward model %s on %s", config.reward_model_name, device)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.reward_model_name,
        ).to(device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(config.reward_model_name)
        self.device = device

    def compute_rewards(self, prompts: list[str], generated_texts: list[str]) -> list[float]:
        """Return one scalar reward per prompt.

        The reward is a property of the whole sequence, not of any single token, which
        is exactly why the credit-assignment machinery downstream (KL penalty, TD
        errors, GAE) exists at all.

        Args:
            prompts: The prompts that were generated from.
            generated_texts: The completion for each prompt.

        Returns:
            One reward per prompt, in the same order.
        """
        full_texts = [
            prompt + completion for prompt, completion in zip(prompts, generated_texts, strict=True)
        ]
        # The OpenAssistant reward model takes a single string per sample (no
        # separate question/answer pair here -- we concatenate prompt+completion).
        # For multi-sample batches, tokenize together with padding.
        inputs = self.tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True).to(
            self.device
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            rewards = outputs.logits

        return [float(reward) for reward in rewards.reshape(-1).tolist()]
