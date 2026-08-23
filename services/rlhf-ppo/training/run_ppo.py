"""Entry point for `make train`: one PPO step on two example prompts.

This is deliberately a single step rather than a training loop. The reward model is a
pretrained OpenAssistant scorer rather than something trained here, and the policy and
value models are only LoRA-initialised, so running thousands of steps would still not
demonstrate learning. One step that provably executes every stage of the algorithm is
the honest artefact.
"""

from peft import get_peft_model_state_dict

from ai_residency_common import (
    finish_tracking,
    get_logger,
    init_tracking,
    log_metrics,
    push_checkpoint,
    save_checkpoint,
)
from config import PPOConfig
from ppo_trainer import PPOStepResult, PPOTrainer

__all__ = ["EXAMPLE_PROMPTS", "main", "run"]

logger = get_logger(__name__)

# The Reddit TL;DR summarization format the PPO task brief targets.
EXAMPLE_PROMPTS = [
    "SUBREDDIT: r/relationships\nTITLE: Should I admit to snooping?\nPOST: ...\nTL;DR:",
    "SUBREDDIT: r/advice\nTITLE: Need help with decision\nPOST: ...\nTL;DR:",
]


def run(config: PPOConfig, prompts: list[str] | None = None) -> PPOStepResult:
    """Run one PPO step and report what it produced.

    Args:
        config: Run configuration.
        prompts: Prompts to generate from. Defaults to `EXAMPLE_PROMPTS`.

    Returns:
        The step's results.
    """
    init_tracking(config.wandb_project, config=vars(config))

    # `set_seed` is not called here: `PPOTrainer.__init__` does it, because it has to
    # run before the four models' LoRA weights are initialised.
    trainer = PPOTrainer(config)
    results = trainer.ppo_step(prompts if prompts is not None else EXAMPLE_PROMPTS)

    for index, text in enumerate(results["generated_texts"]):
        logger.info("Completion %d: %s", index + 1, text)
    logger.info("Rewards: %s", [round(reward, 4) for reward in results["rewards"]])
    logger.info("Value loss: %.4f", results["value_loss"])
    logger.info("Policy loss: %.4f", results["policy_loss"])

    first_pair = results["sa_pairs"][0]
    logger.info("First (s, a) pair, state: %s...", first_pair["state_text"][:100])
    logger.info("First (s, a) pair, action: %s", first_pair["action_token"])
    logger.info("First (s, a) pair, KL divergence: %.4f", first_pair["kl_div"])
    logger.info("First (s, a) pair, advantage: %.4f", first_pair["advantage"])
    logger.info("First (s, a) pair, reward-to-go: %.4f", first_pair["reward_to_go"])

    log_metrics(
        {
            "value_loss": results["value_loss"],
            "policy_loss": results["policy_loss"],
            "mean_reward": sum(results["rewards"]) / len(results["rewards"]),
        }
    )

    # The step moved the policy's LoRA weights, so persist them like every other repo
    # here persists its trained artifact. This is not a model that got better -- it is
    # the output of a run that provably executed, which is what makes it reproducible
    # from somewhere else.
    #
    # Only the adapter: the policy wraps a frozen SmolLM2-135M, so its full state_dict
    # is 2.4GB of weights PPO never touched, re-uploaded on every run.
    adapter_weights = get_peft_model_state_dict(trainer.policy_model.model)
    save_checkpoint(
        config.checkpoint_path,
        trainer.policy_model,
        epoch=1,
        metrics={"value_loss": results["value_loss"], "policy_loss": results["policy_loss"]},
        extra={
            "note": "policy LoRA adapter after exactly one PPO step",
            "base_model": config.model_name,
        },
        state_dict=adapter_weights,
    )
    if config.hf_repo_id is not None:
        push_checkpoint(config.hf_repo_id, config.checkpoint_path)

    finish_tracking()
    return results


def main() -> None:
    """Entry point for `make train`."""
    run(PPOConfig())


if __name__ == "__main__":
    main()
