"""Run configuration for a PPO step.

`PPOConfig` inherits the fields every repo in this portfolio shares from
`ai_residency_common.BaseConfig` — seed, batch_size, learning_rate, num_epochs,
checkpoint_dir, data_dir, hf_repo_id, wandb_project, plus a `device` property — and
adds only what is specific to PPO on top.
"""

from dataclasses import dataclass
from pathlib import Path

from ai_residency_common import BaseConfig

__all__ = ["PPOConfig"]


@dataclass
class PPOConfig(BaseConfig):
    """Configuration for PPO training.

    `seed` is inherited from `BaseConfig` and applied by `PPOTrainer` via
    `set_seed` before any model is constructed or any token sampled, so a fixed
    config produces identical generated text, rewards and losses run to run.

    Attributes:
        model_name: Hugging Face model id of the base model all four model copies
            (SFT, policy, reward, value) are built from.
        lora_r: LoRA rank.
        lora_alpha: LoRA scaling factor.
        lora_dropout: Dropout applied inside the LoRA adapters.
        value_learning_rate: Learning rate for the value model, kept separate from
            the inherited `learning_rate`, which drives the policy. The value head
            is randomly initialised and needs a faster rate than the LoRA-adapted
            policy.
        max_new_tokens: Tokens generated per prompt in a PPO step. One (s, a) pair
            is produced per generated token, so this also sets how much experience
            a single step yields.
        gamma: Discount factor for TD errors and GAE.
        lam: GAE lambda, trading bias against variance in the advantage estimate.
        kl_coef: Weight on the per-token KL penalty against the reference model.
        clip_epsilon: PPO's clipping range on the probability ratio.
        temperature: Sampling temperature during generation.
        top_k: Top-k filter during generation.
        top_p: Nucleus sampling threshold during generation.
    """

    # Model
    model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    reward_model_name: str = "OpenAssistant/reward-model-deberta-v3-large-v2"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1

    # Training. `learning_rate` is inherited; PPO on a LoRA-adapted policy wants a
    # much smaller rate than BaseConfig's default of 1e-3.
    learning_rate: float = 1e-5
    value_learning_rate: float = 1e-4
    max_new_tokens: int = 10
    gamma: float = 0.99
    lam: float = 0.95
    kl_coef: float = 0.1
    clip_epsilon: float = 0.2

    # Generation
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95

    # The step does update the policy's LoRA weights, so there is an artifact worth
    # persisting -- but be clear about what it is. One step against a reward model
    # nothing here trains is a demonstration that the algorithm runs end to end, not
    # a model whose behaviour has meaningfully improved. The file name says so.
    checkpoint_name: str = "policy_after_one_ppo_step.pth"
    hf_repo_id: str | None = "roshbeed/ai-residency-rlhf-ppo"
    wandb_project: str | None = "ai-residency-rlhf-ppo"

    @property
    def checkpoint_path(self) -> Path:
        """Full path to the policy checkpoint a step writes."""
        return self.checkpoint_dir / self.checkpoint_name
