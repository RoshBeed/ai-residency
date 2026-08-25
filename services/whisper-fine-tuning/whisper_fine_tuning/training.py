"""Fine-tuning loop.

Part 1 is deliberately a single-example demo: take released Whisper-tiny
weights, take one clip whose transcript the model gets wrong, and take one gradient
step toward the correct text. It is a smoke test that the loss, the teacher-forced
shift and the optimizer are all wired up correctly, not a real training run.

That still fits the portfolio's canonical training shape:

    Config -> set_seed -> baseline -> init_tracking
           -> step loop -> log_metrics
           -> on new best: save_checkpoint + push_checkpoint
           -> finish_tracking

The baseline here is the loss the released weights already achieve on this pair,
measured before any gradient is applied. That is the number the fine-tuned model has
to beat for the demo to mean anything, and it is cheap and honest to compute because
training and evaluation are the same single example — this is a memorisation demo by
construction, and reporting it against its own starting point says so plainly.
"""

from collections.abc import Callable

import torch

from ai_residency_common import (
    finish_tracking,
    get_logger,
    init_tracking,
    log_metrics,
    push_checkpoint,
    save_checkpoint,
    set_seed,
)
from config import Config
from speech import Whisper
from whisper_fine_tuning.models import load_whisper_model, prepare_training_data

__all__ = [
    "compute_loss",
    "evaluate",
    "fine_tune",
    "main",
    "train",
]

logger = get_logger(__name__)


def compute_loss(whisper: Whisper, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return the teacher-forced cross-entropy loss for one batch.

    The shift the decoder needs -- predicting position *n* from positions before it --
    is done by the model, which builds its decoder input by shifting `labels` right and
    prepending `decoder_start_token_id`. This used to be written out here as
    `logits[:, :-1]` against `tokens[:, 1:]` plus a separate cross-entropy, which is
    the same operation done by hand against a library that did not offer it.

    Args:
        whisper: A loaded model.
        features: Encoder input, shape `(batch, 80, 3000)`.
        labels: Target token ids, shape `(batch, sequence)`.

    Returns:
        Scalar loss.
    """
    loss: torch.Tensor = whisper.model(input_features=features, labels=labels).loss
    return loss


def evaluate(whisper: Whisper, features: torch.Tensor, labels: torch.Tensor) -> float:
    """Return the loss for one batch without updating any weights.

    Args:
        whisper: A loaded model.
        features: Encoder input.
        labels: Target token ids.

    Returns:
        The loss as a plain float.
    """
    whisper.model.eval()
    with torch.no_grad():
        return float(compute_loss(whisper, features, labels).item())


def fine_tune(
    whisper: Whisper,
    audio_path: str,
    target_text: str,
    *,
    learning_rate: float = 1e-4,
    num_steps: int = 1,
    on_step: Callable[[int, float], None] | None = None,
) -> Whisper:
    """Fine-tune a model in place on a single (audio, text) pair.

    This is the entry point the API and the Streamlit UI call per request, where the
    caller already holds a live model and wants it nudged toward one correction. A
    fresh optimizer is built each call, matching how those callers use it: each
    correction is independent rather than a continuation of one long run.

    Args:
        whisper: Model to update, modified in place and also returned.
        audio_path: Audio to fine-tune against.
        target_text: The transcript the model should produce for that audio.
        learning_rate: Adam learning rate.
        num_steps: Gradient steps to take.
        on_step: Optional callback invoked after each optimizer step with the
            1-indexed step number and that step's loss, so a caller can trace the
            loop instead of only seeing the final result.

    Returns:
        The same bundle, its model updated.
    """
    features, labels = prepare_training_data(whisper, audio_path, target_text)
    optimizer = torch.optim.Adam(whisper.model.parameters(), lr=learning_rate)

    whisper.model.train()
    for step in range(1, num_steps + 1):
        loss = compute_loss(whisper, features, labels)
        optimizer.zero_grad()
        torch.autograd.backward(loss)
        optimizer.step()
        if on_step is not None:
            on_step(step, float(loss.item()))

    return whisper


def train(config: Config) -> float:
    """Run the Part 1 fine-tuning demo end to end.

    Args:
        config: Run configuration.

    Returns:
        The best loss reached on the demo pair.
    """
    set_seed(config.seed)

    whisper = load_whisper_model(config)
    audio_path = str(config.audio_path)
    features, labels = prepare_training_data(whisper, audio_path, config.target_text)

    baseline_loss = evaluate(whisper, features, labels)
    logger.info(
        "Baseline (released weights, before any update) loss on %r: %.4f",
        config.target_text,
        baseline_loss,
    )

    optimizer = torch.optim.Adam(whisper.model.parameters(), lr=config.learning_rate)
    init_tracking(
        config.wandb_project,
        config={"baseline_loss": baseline_loss, **vars(config)},
    )

    best_loss = float("inf")
    for step in range(1, config.num_epochs + 1):
        whisper.model.train()
        loss = compute_loss(whisper, features, labels)
        optimizer.zero_grad()
        # `Tensor.backward` carries no annotations in torch, so strict mypy rejects
        # calling it on a genuinely-typed Tensor. This is the same code path.
        torch.autograd.backward(loss)
        optimizer.step()

        train_loss = float(loss.item())
        val_loss = evaluate(whisper, features, labels)
        log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=step)
        logger.info(
            "Step %d/%d  train_loss=%.4f  val_loss=%.4f",
            step,
            config.num_epochs,
            train_loss,
            val_loss,
        )

        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(
                config.checkpoint_path,
                whisper.model,
                optimizer=optimizer,
                epoch=step,
                metrics={"val_loss": val_loss, "baseline_loss": baseline_loss},
                extra={
                    "model_name": config.model_name,
                    "target_text": config.target_text,
                },
            )
            # Only on a new best: a Whisper checkpoint is large and pushing every
            # improvement would dominate the run.
            if config.hf_repo_id is not None:
                push_checkpoint(config.hf_repo_id, config.checkpoint_path)

    finish_tracking()
    logger.info(
        "Best loss %.4f against baseline %.4f (%s)",
        best_loss,
        baseline_loss,
        "beats baseline" if best_loss < baseline_loss else "does NOT beat baseline",
    )
    return best_loss


def main() -> None:
    """Entry point for `make train`."""
    train(Config())


if __name__ == "__main__":
    main()
