"""The gradient updates a PPO step performs."""

from typing import Any

import torch
from ai_residency_common import get_logger

from config import PPOConfig
from models import PolicyModel, ValueModel
from sa_pair import SAPair
from training import PolicyLoss, PolicyRatio, ValueLoss

__all__ = ["ModelTrainer"]

logger = get_logger(__name__)


class ModelTrainer:
    """Owns the two optimizers and runs the policy and value updates."""

    def __init__(
        self,
        policy_model: PolicyModel,
        value_model: ValueModel,
        tokenizer: Any,
        config: PPOConfig,
        device: torch.device,
    ) -> None:
        """Build the optimizers for the two trainable models.

        Args:
            policy_model: The model PPO optimises.
            value_model: The critic.
            tokenizer: Shared tokenizer.
            config: Run configuration, supplying both learning rates.
            device: Device to run on.
        """
        self.policy_model = policy_model
        self.value_model = value_model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device

        self.policy_optimizer = torch.optim.AdamW(
            self.policy_model.parameters(), lr=config.learning_rate
        )
        self.value_optimizer = torch.optim.AdamW(
            self.value_model.parameters(), lr=config.value_learning_rate
        )

    def train_value_model(self, sa_pairs: list[SAPair]) -> float:
        """Regress the value model onto reward-to-go targets.

        Args:
            sa_pairs: Pairs carrying `state_text` and `reward_to_go`.

        Returns:
            Mean value loss across the batch.
        """
        self.value_model.train()

        # Shuffle so consecutive tokens from one sequence, which are highly
        # correlated, do not all land in the same minibatch.
        order = torch.randperm(len(sa_pairs)).tolist()
        state_texts = [sa_pairs[index]["state_text"] for index in order]
        targets = [sa_pairs[index]["reward_to_go"] for index in order]

        total_loss = 0.0
        for start in range(0, len(state_texts), self.config.batch_size):
            batch_states = state_texts[start : start + self.config.batch_size]
            batch_targets = torch.tensor(
                targets[start : start + self.config.batch_size], dtype=torch.float32
            ).to(self.device)

            inputs = self.tokenizer(
                batch_states, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)

            predictions = self.value_model(inputs["input_ids"], inputs["attention_mask"]).squeeze(
                -1
            )
            loss = ValueLoss.mse_loss(predictions, batch_targets)

            self.value_optimizer.zero_grad()
            # `Tensor.backward` carries no annotations in torch's own source, so
            # calling it from typed code trips strict mypy's untyped-call check.
            # `torch.autograd.backward` is what it delegates to, and is typed.
            torch.autograd.backward(loss)
            self.value_optimizer.step()

            total_loss += loss.item() * len(batch_states)

        self.value_model.eval()
        return total_loss / len(state_texts)

    def train_policy_model(self, sa_pairs: list[SAPair]) -> float:
        """Take one PPO-clip gradient step on the policy.

        Args:
            sa_pairs: Pairs carrying `state_text`, `action_token_id`, `logprob` and
                `advantage`.

        Returns:
            The policy loss for this step.
        """
        self.policy_model.train()

        state_texts = [pair["state_text"] for pair in sa_pairs]
        action_token_ids = [int(pair["action_token_id"]) for pair in sa_pairs]
        old_logprobs = torch.tensor(
            [
                pair["old_logprob"] if "old_logprob" in pair else pair["logprob"]
                for pair in sa_pairs
            ],
            dtype=torch.float32,
        ).to(self.device)
        advantages = torch.tensor([pair["advantage"] for pair in sa_pairs], dtype=torch.float32).to(
            self.device
        )

        new_logprobs = PolicyRatio.get_new_logprobs(
            self.policy_model,
            self.tokenizer,
            state_texts,
            action_token_ids,
            self.device,
            self.config.batch_size,
        )

        loss = PolicyLoss.ppo_clip_loss(
            new_logprobs.to(self.device), old_logprobs, advantages, self.config.clip_epsilon
        )

        self.policy_optimizer.zero_grad()
        torch.autograd.backward(loss)
        self.policy_optimizer.step()

        self.policy_model.eval()
        return float(loss.item())

    def compute_policy_ratios(self, sa_pairs: list[SAPair]) -> list[SAPair]:
        """Recompute logprobs under the (now updated) policy and add the ratio to each pair.

        ``train_policy_model`` already computed logprobs before the gradient step for
        the loss, but those are stale the moment the optimizer fires -- the model
        weights have moved. This recomputes them post-update so ``policy_ratio =
        exp(logprob_new - logprob_old)`` reflects the actual parameter change. A ratio
        near 1 means the policy barely moved on that token; a ratio far from 1 means
        it moved, and the direction of the move is what the clipping in the loss
        constrained.

        Args:
            sa_pairs: Pairs carrying ``state_text``, ``action_token_id`` and ``logprob``
                (the pre-update logprob).

        Returns:
            The same pairs, now with ``policy_ratio`` populated.
        """
        state_texts = [pair["state_text"] for pair in sa_pairs]
        action_token_ids = [int(pair["action_token_id"]) for pair in sa_pairs]
        old_logprobs = torch.tensor([pair["logprob"] for pair in sa_pairs], dtype=torch.float32).to(
            self.device
        )

        new_logprobs = PolicyRatio.get_new_logprobs(
            self.policy_model,
            self.tokenizer,
            state_texts,
            action_token_ids,
            self.device,
            self.config.batch_size,
        )

        ratios = torch.exp(new_logprobs - old_logprobs)
        for pair, ratio in zip(sa_pairs, ratios.tolist(), strict=True):
            pair["policy_ratio"] = float(ratio)

        return sa_pairs

    def compute_clipped_ratios(self, sa_pairs: list[SAPair]) -> list[SAPair]:
        """Compute ratio = π_θ / π_old per token, then clamp to [1-ε, 1+ε].

        This is the raw clip operation that ``train_policy_model`` applies inside
        the loss. Exposing it as its own stage separates the clip from the loss
        computation so the per-token effect of the clip is visible before the
        scalar loss renders it opaque.

        The ratio is computed from a *pre-update* forward pass (the policy weights
        have not been touched yet at this point), so for the first-ever call the
        ratio is exactly 1.0 everywhere. After repeated calls (cached pod), the
        ratio drifts as earlier gradient steps accumulate.

        Args:
            sa_pairs: Pairs carrying ``state_text``, ``action_token_id`` and ``logprob``.

        Returns:
            The same pairs, now with ``clipped_ratio`` populated.
        """
        state_texts = [pair["state_text"] for pair in sa_pairs]
        action_token_ids = [int(pair["action_token_id"]) for pair in sa_pairs]
        old_logprobs = torch.tensor([pair["logprob"] for pair in sa_pairs], dtype=torch.float32).to(
            self.device
        )

        new_logprobs = PolicyRatio.get_new_logprobs(
            self.policy_model,
            self.tokenizer,
            state_texts,
            action_token_ids,
            self.device,
            self.config.batch_size,
        )

        ratios = torch.exp(new_logprobs - old_logprobs)
        clipped = torch.clamp(ratios, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon)
        for pair, c in zip(sa_pairs, clipped.tolist(), strict=True):
            pair["clipped_ratio"] = float(c)

        return sa_pairs
