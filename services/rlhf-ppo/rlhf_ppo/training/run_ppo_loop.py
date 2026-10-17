"""A real PPO loop, and the before/after reward it exists to produce.

`run_ppo.py` runs one step on two illustrative prompts. That shows the algorithm
executes; it cannot show it *works*, because a single update to a LoRA-adapted policy
moves nothing measurable and the prompts have a literal "POST: ..." where the post
should be. This service's own write-up said as much: what it needed next was a
multi-epoch loop with reward measured before and after.

The measurement is the point, so it is built to be hard to fool:

* **A fixed held-out prompt set.** Reward is measured on the same prompts every time,
  and those prompts are never trained on. Measuring on the training batch would report
  the policy's progress at memorising four prompts.
* **The pre-PPO reward is the baseline**, taken before a single gradient step, with the
  same sampler and the same prompts. A reward number alone says nothing -- the scale is
  whatever the OpenAssistant scorer happens to output for this kind of text.
* **Greedy decoding at evaluation.** Training samples at temperature 1.0, which is what
  PPO needs for exploration, but a sampled evaluation would mix policy improvement with
  sampling noise across runs.

What this is not: PPO against a reward model nobody here trained, on a 135M policy, for
tens of steps. Reward going up means the policy moved toward what that scorer likes. It
is not evidence of a better summarizer, and the KL penalty against the frozen reference
is what stops "what the scorer likes" from collapsing into gibberish that games it.
"""

import argparse
import statistics

import torch
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
from datasets import load_dataset
from rlhf_ppo.ppo import PPOTrainer

__all__ = ["load_prompts", "mean_reward", "run_loop"]

logger = get_logger(__name__)


def load_prompts(config: PPOConfig, count: int) -> list[str]:
    """Load real TL;DR prompts.

    Args:
        config: Supplies the dataset and split.
        count: How many prompts to take.

    Returns:
        Prompt strings, already ending in the `TL;DR:` cue the policy completes.
    """
    dataset = load_dataset(
        config.prompt_dataset_repo_id, split=f"{config.prompt_dataset_split}[:{count}]"
    )
    prompts = [str(row["prompt"]).strip() for row in dataset]
    logger.info("Loaded %d prompts from %s", len(prompts), config.prompt_dataset_repo_id)
    return prompts


def mean_reward(trainer: PPOTrainer, prompts: list[str]) -> float:
    """Score the current policy on fixed prompts without updating it.

    Generation here is greedy rather than sampled. The trainer's own generator samples
    at `temperature`, which is right for collecting experience and wrong for a
    before/after comparison: two measurements of an unchanged policy would differ.

    Args:
        trainer: Supplies the policy, tokenizer and reward model.
        prompts: The held-out prompts.

    Returns:
        Mean reward over the prompts.
    """
    model = trainer.policy_model.model
    tokenizer = trainer.generator.tokenizer
    texts: list[str] = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {key: value.to(trainer.device) for key, value in inputs.items()}
            generated = model.generate(
                **inputs,
                max_new_tokens=trainer.config.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
            completion = generated[0][inputs["input_ids"].shape[1] :]
            texts.append(tokenizer.decode(completion, skip_special_tokens=True))
    if was_training:
        model.train()
    return statistics.fmean(trainer.reward_computer.compute_rewards(prompts, texts))


def run_loop(config: PPOConfig) -> dict[str, float]:
    """Run `num_iterations` PPO steps and report reward before and after.

    Args:
        config: Run configuration.

    Returns:
        The baseline and final mean reward on the held-out prompts.
    """
    init_tracking(config.wandb_project, config=vars(config))
    trainer = PPOTrainer(config)

    # The evaluation prompts come first in the split and the training prompts after it,
    # so the two sets cannot overlap.
    everything = load_prompts(
        config, config.eval_prompts + config.num_iterations * config.prompt_batch_size
    )
    eval_prompts = everything[: config.eval_prompts]
    train_prompts = everything[config.eval_prompts :]

    baseline = mean_reward(trainer, eval_prompts)
    logger.info("Pre-PPO mean reward on %d held-out prompts: %.4f", len(eval_prompts), baseline)
    log_metrics({"baseline_mean_reward": baseline}, step=0)

    latest = baseline
    for iteration in range(1, config.num_iterations + 1):
        start = (iteration - 1) * config.prompt_batch_size
        batch = train_prompts[start : start + config.prompt_batch_size]
        if not batch:
            logger.info("Ran out of prompts at iteration %d", iteration)
            break

        results = trainer.ppo_step(batch)
        train_reward = statistics.fmean(results["rewards"])
        log_metrics(
            {
                "train_mean_reward": train_reward,
                "value_loss": results["value_loss"],
                "policy_loss": results["policy_loss"],
            },
            step=iteration,
        )
        logger.info(
            "Iteration %d/%d  train_reward=%.4f  value_loss=%.4f  policy_loss=%.4f",
            iteration,
            config.num_iterations,
            train_reward,
            results["value_loss"],
            results["policy_loss"],
        )

        if iteration % config.eval_every == 0 or iteration == config.num_iterations:
            latest = mean_reward(trainer, eval_prompts)
            log_metrics({"held_out_mean_reward": latest}, step=iteration)
            logger.info(
                "Iteration %d: held-out mean reward %.4f against baseline %.4f",
                iteration,
                latest,
                baseline,
            )

    adapter_weights = get_peft_model_state_dict(trainer.policy_model.model)
    save_checkpoint(
        config.checkpoint_path,
        trainer.policy_model,
        epoch=config.num_iterations,
        metrics={"baseline_mean_reward": baseline, "final_mean_reward": latest},
        extra={
            "note": f"policy LoRA adapter after {config.num_iterations} PPO steps",
            "base_model": config.model_name,
            "reward_model": config.reward_model_name,
        },
        state_dict=adapter_weights,
    )
    if config.hf_repo_id is not None:
        push_checkpoint(config.hf_repo_id, config.checkpoint_path)

    logger.info(
        "Mean reward %.4f -> %.4f on held-out prompts (%s)",
        baseline,
        latest,
        "improved" if latest > baseline else "did NOT improve",
    )
    finish_tracking()
    return {"baseline_mean_reward": baseline, "final_mean_reward": latest}


def main() -> None:
    """Entry point for `make train-loop`."""
    parser = argparse.ArgumentParser(description="Run a PPO loop and measure reward")
    parser.add_argument("--iterations", type=int, default=None, help="PPO steps to run")
    args = parser.parse_args()
    config = PPOConfig()
    if args.iterations is not None:
        config.num_iterations = args.iterations
    run_loop(config)


if __name__ == "__main__":
    main()
