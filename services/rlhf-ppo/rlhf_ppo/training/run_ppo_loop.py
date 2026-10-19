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

__all__ = ["load_prompts", "mean_reward", "policy_drift", "run_loop"]

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


def mean_reward(
    trainer: PPOTrainer, prompts: list[str], *, samples: int = 0, seed: int = 0
) -> float:
    """Score the current policy on fixed prompts without updating it.

    Two modes, because one is not enough. **Greedy** (`samples=0`) is exactly
    reproducible: two measurements of an unchanged policy agree to the last digit, which
    is what makes a before/after difference trustworthy. Its weakness is the mirror
    image -- it only moves when a change flips an argmax, so a policy that has genuinely
    shifted can read as untouched. A 60-step run at `learning_rate=1e-5` perturbed the
    policy weights by a relative Frobenius norm of 8.2e-5 and reported the *identical*
    greedy reward at iterations 10, 20, 30 and 40, to four decimal places.

    **Seeded sampling** (`samples=N`) is the sensitive one: it averages N completions per
    prompt at the training temperature, from a fixed seed, so it responds to a shift in
    the output distribution rather than only to a change of its mode. The seed is what
    keeps it comparable between calls.

    Args:
        trainer: Supplies the policy, tokenizer and reward model.
        prompts: The held-out prompts.
        samples: Completions per prompt. 0 means one greedy completion.
        seed: Seed for the sampler, so repeated calls draw the same noise.

    Returns:
        Mean reward over the prompts (and over samples, when sampling).
    """
    model = trainer.policy_model.model
    tokenizer = trainer.generator.tokenizer
    config = trainer.config
    scored_prompts: list[str] = []
    texts: list[str] = []
    was_training = model.training
    model.eval()

    if samples:
        # Same noise every call, so a difference between calls is the policy moving
        # rather than the sampler. Restored afterwards: the training loop's own
        # exploration must not inherit this seed.
        state = torch.get_rng_state()
        torch.manual_seed(seed)

    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {key: value.to(trainer.device) for key, value in inputs.items()}
            generated = model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                do_sample=bool(samples),
                num_return_sequences=max(samples, 1),
                temperature=config.temperature if samples else None,
                top_k=config.top_k if samples else None,
                top_p=config.top_p if samples else None,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
            prompt_length = inputs["input_ids"].shape[1]
            for sequence in generated:
                texts.append(tokenizer.decode(sequence[prompt_length:], skip_special_tokens=True))
                scored_prompts.append(prompt)

    if samples:
        torch.set_rng_state(state)
    if was_training:
        model.train()
    return statistics.fmean(trainer.reward_computer.compute_rewards(scored_prompts, texts))


def policy_drift(trainer: PPOTrainer) -> float:
    """How far the LoRA adapter has moved the policy, relative to the base weights.

    This exists because a reward that does not move is ambiguous on its own: either the
    policy did not change, or it changed by less than the metric can see. One number
    separates those. LoRA's B matrices initialise to zero, so the adapter contributes
    nothing at step 0 and this starts at exactly 0.0.

    The quantity is `||(alpha/r) * B @ A|| / ||W||` on the attention query projections,
    averaged over layers -- the size of the update the adapter adds, as a fraction of the
    weight it is added to. A 60-step run at `learning_rate=1e-5` reads 8.2e-5 here, which
    is 0.008% of the base weights and far too small to change a greedy argmax.

    Args:
        trainer: Supplies the LoRA-adapted policy.

    Returns:
        Mean relative Frobenius norm of the adapter's effective weight delta.
    """
    model = trainer.policy_model.model
    scale = trainer.config.lora_alpha / trainer.config.lora_r
    ratios: list[float] = []
    with torch.no_grad():
        for module in model.modules():
            lora_a = getattr(module, "lora_A", None)
            lora_b = getattr(module, "lora_B", None)
            base = getattr(module, "base_layer", None)
            if lora_a is None or lora_b is None or base is None:
                continue
            for name, a_layer in lora_a.items():
                b_layer = lora_b[name]
                delta = scale * (b_layer.weight @ a_layer.weight)
                denominator = base.weight.norm().item()
                if denominator:
                    ratios.append(delta.norm().item() / denominator)
    return statistics.fmean(ratios) if ratios else 0.0


def run_loop(config: PPOConfig) -> dict[str, float]:
    """Run `num_iterations` PPO steps and report reward before and after.

    Args:
        config: Run configuration.

    Returns:
        The baseline and final mean reward on the held-out prompts.
    """
    init_tracking(config.wandb_project, config=vars(config))
    # Deliberately not resuming: the number this loop exists to produce is reward before
    # and after PPO, which only means anything if "before" is the untrained policy.
    trainer = PPOTrainer(config, load_adapters=False)

    # The evaluation prompts come first in the split and the training prompts after it,
    # so the two sets cannot overlap.
    everything = load_prompts(
        config, config.eval_prompts + config.num_iterations * config.prompt_batch_size
    )
    eval_prompts = everything[: config.eval_prompts]
    train_prompts = everything[config.eval_prompts :]

    # Zero unless something resumed, which `load_adapters=False` prevents -- asserted
    # rather than assumed, because a non-zero baseline drift would silently invalidate
    # every improvement this loop reports.
    baseline_drift = policy_drift(trainer)
    if baseline_drift:
        logger.error(
            "Baseline policy is already %.3e from the base weights; 'pre-PPO' is a "
            "misnomer and the before/after below understates the starting point",
            baseline_drift,
        )

    baseline = mean_reward(trainer, eval_prompts)
    baseline_sampled = mean_reward(
        trainer, eval_prompts, samples=config.eval_samples, seed=config.seed
    )
    logger.info(
        "Pre-PPO mean reward on %d held-out prompts: greedy %.4f, sampled (%dx) %.4f",
        len(eval_prompts),
        baseline,
        config.eval_samples,
        baseline_sampled,
    )
    log_metrics(
        {"baseline_mean_reward": baseline, "baseline_sampled_mean_reward": baseline_sampled},
        step=0,
    )

    latest = baseline
    latest_sampled = baseline_sampled
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
            latest_sampled = mean_reward(
                trainer, eval_prompts, samples=config.eval_samples, seed=config.seed
            )
            log_metrics(
                {
                    "held_out_mean_reward": latest,
                    "held_out_sampled_mean_reward": latest_sampled,
                    "policy_drift": policy_drift(trainer),
                },
                step=iteration,
            )
            logger.info(
                "Iteration %d: greedy %.4f (baseline %.4f), sampled %.4f (baseline %.4f), "
                "policy moved %.3e",
                iteration,
                latest,
                baseline,
                latest_sampled,
                baseline_sampled,
                policy_drift(trainer),
            )

    adapter_weights = get_peft_model_state_dict(trainer.policy_model.model)
    save_checkpoint(
        config.loop_checkpoint_path,
        trainer.policy_model,
        epoch=config.num_iterations,
        metrics={
            "baseline_mean_reward": baseline,
            "final_mean_reward": latest,
            "baseline_sampled_mean_reward": baseline_sampled,
            "final_sampled_mean_reward": latest_sampled,
            "policy_drift": policy_drift(trainer),
        },
        extra={
            "note": f"policy LoRA adapter after {config.num_iterations} PPO steps",
            "base_model": config.model_name,
            "reward_model": config.reward_model_name,
        },
        state_dict=adapter_weights,
    )
    if config.hf_repo_id is not None:
        push_checkpoint(config.hf_repo_id, config.loop_checkpoint_path)

    logger.info(
        "Greedy mean reward %.4f -> %.4f (%s); sampled %.4f -> %.4f (%s); "
        "policy weights moved %.3e relative",
        baseline,
        latest,
        "improved" if latest > baseline else "did NOT improve",
        baseline_sampled,
        latest_sampled,
        "improved" if latest_sampled > baseline_sampled else "did NOT improve",
        policy_drift(trainer),
    )
    finish_tracking()
    return {
        "baseline_mean_reward": baseline,
        "final_mean_reward": latest,
        "baseline_sampled_mean_reward": baseline_sampled,
        "final_sampled_mean_reward": latest_sampled,
        "policy_drift": policy_drift(trainer),
    }


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
