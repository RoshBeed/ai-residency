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
        prompt_dataset_repo_id: Where real TL;DR prompts come from. The two prompts
            `run_ppo` ships are illustrative and have a literal "POST: ..." where the
            post should be, which is fine for showing the algorithm executes and
            useless for showing reward moving.
        prompt_dataset_split: Split to read prompts from.
        loop_checkpoint_name: Filename `make train-loop` writes, kept distinct from
            the single-step one so neither can be mistaken for the other.
        num_iterations: PPO steps in a loop run. Each one generates, scores and
            updates on `prompt_batch_size` prompts.
        prompt_batch_size: Prompts generated from per PPO step.
        eval_prompts: Held-out prompts the before/after reward is measured on. Fixed
            across the run, and never trained on, so the two numbers are comparable.
        eval_every: PPO steps between reward measurements.
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

    # A real loop. `make train` still runs the single documented step; `make train-loop`
    # runs this.
    prompt_dataset_repo_id: str = "CarperAI/openai_summarize_tldr"
    prompt_dataset_split: str = "train"
    num_iterations: int = 60
    # Inherited from BaseConfig as 32, lowered to 16 after measuring the value
    # update's forward+backward over (batch x ~450-token state) on MPS: 0.26s at
    # batch 4, 0.23s at 8, 0.26s at 16 and 55.16s at 32, a 200x cliff with no
    # gradual climb before it. The cliff turned out to be a symptom rather than the
    # disease -- see the `make train-loop` note on why this service runs on CPU --
    # but 16 is measured safe on both devices and 32 is not, so it stays.
    batch_size: int = 16
    prompt_batch_size: int = 4
    eval_prompts: int = 16
    eval_every: int = 10
    # Completions per held-out prompt for the sampled reward. Greedy decode only moves
    # when a change flips an argmax, which a 60-step run at lr=1e-5 never did -- it
    # reported the identical reward four times while the policy weights moved 8.2e-5.
    # Averaging sampled completions from a fixed seed responds to a shift in the
    # distribution instead of only to a change of its mode.
    eval_samples: int = 4

    # The step does update the policy's LoRA weights, so there is an artifact worth
    # persisting -- but be clear about what it is. One step against a reward model
    # nothing here trains is a demonstration that the algorithm runs end to end, not
    # a model whose behaviour has meaningfully improved. The file name says so.
    checkpoint_name: str = "policy_after_one_ppo_step.pth"
    # A loop run must not overwrite the one-step artifact or inherit its name. The
    # filename is the only thing a downstream puller sees, and "after_one_ppo_step"
    # on a 60-step adapter is a lie that nothing else would catch.
    loop_checkpoint_name: str = "policy_after_ppo_loop.pth"
    hf_repo_id: str | None = "roshbeed/ai-residency-rlhf-ppo"
    wandb_project: str | None = "ai-residency-rlhf-ppo"

    @property
    def checkpoint_path(self) -> Path:
        """Full path to the policy checkpoint a single step writes."""
        return self.checkpoint_dir / self.checkpoint_name

    @property
    def loop_checkpoint_path(self) -> Path:
        """Full path to the policy checkpoint a loop run writes."""
        return self.checkpoint_dir / self.loop_checkpoint_name
