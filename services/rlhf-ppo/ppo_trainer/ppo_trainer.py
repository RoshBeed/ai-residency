"""The ten stages of one PPO step, in order."""

from collections.abc import Callable
from typing import Any, TypedDict

import torch
from ai_residency_common import (
    get_device,
    get_logger,
    load_checkpoint,
    log_metrics,
    save_checkpoint,
    set_seed,
)

from advantage import GAEAdvantage, ReturnCalculator, TDError
from base_setup import BaseModelSetup
from config import PPOConfig
from experience_buffer import ExperienceBuffer
from models import PolicyModel, SFTModel, ValueModel
from ppo_trainer.generator import SequenceGenerator
from ppo_trainer.model_trainer import ModelTrainer
from ppo_trainer.reward_computer import RewardComputer
from ppo_trainer.value_adder import ValueAdder
from reward_kl import KLDivergence, RewardAdjuster
from sa_pair import SAPair
from token_attribution import TokenAttribution

__all__ = ["PPOStepResult", "PPOTrainer"]

logger = get_logger(__name__)


class PPOStepResult(TypedDict):
    """Everything one PPO step produces.

    Attributes:
        sa_pairs: Fully populated (s, a) pairs, one per generated token.
        generated_texts: The completion generated for each prompt.
        rewards: The reward model's score for each prompt.
        value_loss: Mean value-model loss for this step.
        policy_loss: PPO-clip policy loss for this step.
    """

    sa_pairs: list[SAPair]
    generated_texts: list[str]
    rewards: list[float]
    value_loss: float
    policy_loss: float


class PPOTrainer:
    """Runs a full PPO step over a batch of prompts.

    Four copies of the base model live here at once: the policy being trained, a
    frozen reference for the KL term, a reward scorer and a value estimator. On a
    0.6B base model that is the practical reason LoRA is used rather than full
    fine-tuning.
    """

    def __init__(
        self,
        config: PPOConfig,
        on_progress: Callable[[str], None] | None = None,
    ) -> None:
        """Build the three base-model copies, the reward scorer, and every pipeline stage.

        Args:
            config: Run configuration.
            on_progress: Optional callback fired during initialisation (after each
                model load, etc.) so a caller watching a long first call -- the
                model loads take a while -- can show what is happening rather than
                leaving the client in silence.
        """
        self.config = config

        def progress(msg: str) -> None:
            logger.info(msg)
            if on_progress is not None:
                on_progress(msg)

        # Seeding has to happen here rather than in the entry point, because it must
        # precede both the three models' random LoRA initialisation and the sampled
        # `generate()` calls in `ppo_step`. Without it neither the generated text nor
        # any loss this repo reports would be reproducible.
        set_seed(config.seed)

        self.device = get_device()
        logger.info("Running PPO on %s", self.device)

        self.base_setup = BaseModelSetup(config)

        progress("Loading SFT (reference) model...")
        self.sft_model = SFTModel(self.base_setup).to(self.device)
        # Perturb the reference model's LoRA weights so it is not bit-identical to the
        # policy. Without this logprob == ref_logprob everywhere, KL is zero, and every
        # downstream stage is uniform. A small random nudge (std=0.05) produces visible
        # per-token KL without degrading the model.
        for name, param in self.sft_model.named_parameters():
            if "lora" in name:
                param.data += torch.randn_like(param) * 0.05
        progress("Loading policy model...")
        self.policy_model = PolicyModel(self.base_setup).to(self.device)
        progress("Loading value model...")
        self.value_model = ValueModel(self.base_setup).to(self.device)

        self._load_ckpt()

        progress("Building pipeline stages...")
        self.token_attribution = TokenAttribution(self.base_setup.tokenizer)
        self.kl_divergence = KLDivergence()
        self.reward_adjuster = RewardAdjuster()
        self.td_error = TDError()
        self.gae_advantage = GAEAdvantage()
        self.return_calculator = ReturnCalculator()
        self.experience_buffer = ExperienceBuffer()

        self.generator = SequenceGenerator(
            self.policy_model, self.sft_model, self.base_setup.tokenizer, config, self.device
        )
        self.reward_computer = RewardComputer(config, self.device)
        self.value_adder = ValueAdder(
            self.value_model, self.base_setup.tokenizer, config.batch_size, self.device
        )
        self.model_trainer = ModelTrainer(
            self.policy_model, self.value_model, self.base_setup.tokenizer, config, self.device
        )

        # Only the two trainable models are switched to train mode, and only for the
        # duration of their own update.
        self.sft_model.eval()
        self.policy_model.eval()
        self.value_model.eval()

        self.step_count = 0

    def _load_ckpt(self) -> None:
        """Load persisted LoRA adapters if they exist from a previous pod lifetime."""
        for label, model, path in [
            ("policy LoRA", self.policy_model, self.config.checkpoint_dir / "policy_lora.pth"),
            ("value model", self.value_model, self.config.checkpoint_dir / "value_model.pth"),
        ]:
            if not path.exists():
                continue
            try:
                state = load_checkpoint(path, map_location="cpu")
            except FileNotFoundError:
                continue
            if state is not None:
                missing, unexpected = model.load_state_dict(state, strict=False)
                logger.info(
                    "Loaded %s checkpoint (%d missing, %d unexpected)",
                    label,
                    len(missing),
                    len(unexpected),
                )

    def _save_ckpt(self) -> None:
        """Persist LoRA adapter weights so repeated calls compound across pod restarts."""
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        policy_state = {k: v for k, v in self.policy_model.state_dict().items() if "lora" in k}
        save_checkpoint(
            self.config.checkpoint_dir / "policy_lora.pth",
            model=self.policy_model,
            state_dict=policy_state,
        )
        value_state = {
            k: v
            for k, v in self.value_model.state_dict().items()
            if "lora" in k or "value_head" in k
        }
        save_checkpoint(
            self.config.checkpoint_dir / "value_model.pth",
            model=self.value_model,
            state_dict=value_state,
        )
        logger.info("Saved policy + value checkpoints to %s", self.config.checkpoint_dir)

    def ppo_step(
        self,
        prompts: list[str],
        on_stage: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> PPOStepResult:
        """Run one full PPO step: generate, score, credit-assign, update.

        Args:
            prompts: The prompts to generate from.
            on_stage: Optional callback invoked after each of the ten stages below,
                as `on_stage(stage_name, payload)`. Purely additive -- lets a caller
                (the streaming API) observe the step as it happens without this
                method knowing anything about HTTP or SSE.

        Returns:
            The generated text, rewards, per-token trace and both losses.
        """
        self.step_count += 1

        def emit(stage: str, payload: dict[str, Any]) -> None:
            if on_stage is not None:
                on_stage(stage, payload)

        # 1. Generate, recording one (s, a) pair per generated token. Per-token
        # progress is relayed so a streaming client sees tokens appear live rather
        # than waiting on the full batch.
        def on_token(step: int, token: str, logprob: float) -> None:
            emit("generating", {"step": step, "token": token, "logprob": logprob})

        sa_pairs, generated_texts = self.generator.generate_sequences(prompts, on_token=on_token)
        emit("generate", {"sa_pairs": sa_pairs, "generated_texts": generated_texts})

        # 2-4. Score: per-token KL, sequence reward, KL-penalised reward.
        self.kl_divergence.add_kl_to_sa_pairs(sa_pairs)
        emit("kl_divergence", {"sa_pairs": sa_pairs})
        rewards = self.reward_computer.compute_rewards(prompts, generated_texts)
        emit("rewards", {"rewards": rewards})
        self.reward_adjuster.adjust_rewards_with_kl(sa_pairs, rewards, self.config.kl_coef)
        emit("adjusted_reward", {"sa_pairs": sa_pairs})

        # 5-8. Credit assignment: values, TD errors, GAE advantages, returns.
        self.value_adder.add_values_to_sa_pairs(sa_pairs)
        emit("value", {"sa_pairs": sa_pairs})
        self.td_error.compute_td_errors(sa_pairs, self.config.gamma)
        emit("td_error", {"sa_pairs": sa_pairs})
        self.gae_advantage.compute_gae_advantages(sa_pairs, self.config.gamma, self.config.lam)
        emit("advantage", {"sa_pairs": sa_pairs})
        self.return_calculator.add_reward_to_go(sa_pairs)
        emit("reward_to_go", {"sa_pairs": sa_pairs})

        # Store the fields a multi-epoch loop would replay before the first epoch's
        # gradient update overwrites them.
        self.experience_buffer.add_experience(sa_pairs)
        batch = self.experience_buffer.get_batch(len(sa_pairs))

        # 9. Train the critic against reward-to-go targets.
        value_loss = self.model_trainer.train_value_model(batch)
        emit("value_loss", {"value_loss": value_loss})

        # 10. Clip: compute the raw ratio and clamp it, before it is fed into
        # the loss. This exposes the per-token effect of the clip so the caller
        # can see which tokens the clip bound actually constrained.
        self.model_trainer.compute_clipped_ratios(sa_pairs)
        emit("clip", {"sa_pairs": sa_pairs})

        # 11. PPO-clip loss: the clipped surrogate objective, gradient step.
        policy_loss = self.model_trainer.train_policy_model(batch)
        emit("policy_loss", {"policy_loss": policy_loss})

        # 12. Post-update policy ratio -- recompute logprobs under the now-updated
        # policy weights so the ratio reflects the actual parameter change, not the
        # pre-update logprobs already computed inside the loss.
        self.model_trainer.compute_policy_ratios(sa_pairs)
        emit("policy_ratio", {"sa_pairs": sa_pairs})
        self.experience_buffer.clear()

        self._save_ckpt()

        mean_reward = sum(rewards) / len(rewards)
        logger.info(
            "PPO step %d: value_loss=%.4f policy_loss=%.4f mean_reward=%.4f",
            self.step_count,
            value_loss,
            policy_loss,
            mean_reward,
        )
        log_metrics(
            {
                "value_loss": value_loss,
                "policy_loss": policy_loss,
                "mean_reward": mean_reward,
            },
            step=self.step_count,
        )

        return {
            "sa_pairs": sa_pairs,
            "generated_texts": generated_texts,
            "rewards": rewards,
            "value_loss": value_loss,
            "policy_loss": policy_loss,
        }
