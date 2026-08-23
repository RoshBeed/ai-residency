"""Training loop for the multi-digit seq2seq model.

Same canonical shape as the classifier's trainer: seed, loaders, baseline, tracking,
epoch loop, publish on a new best.

Two details here are worth knowing before reading the loop. Targets are shifted by
one position, so the decoder sees positions `[0 .. n-2]` and predicts `[1 .. n-1]`;
without the shift the causal mask still lets position `i` see itself and the model
learns the identity function. And the scheduler steps once per batch, not once per
epoch, because `total_steps` is counted in batches — stepping it per epoch left a
15-epoch run permanently inside its 4000-step warmup.
"""

import argparse
import math
from collections import Counter
from dataclasses import asdict

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ai_residency_common import (
    finish_tracking,
    get_device,
    get_logger,
    init_tracking,
    log_metrics,
    push_checkpoint,
    save_checkpoint,
    set_seed,
)
from config import Config
from datasets.multi_digit_dataset import Example, MultiDigitMNISTDataset
from models.seq2seq.seq2seq import Seq2SeqVisionTransformer

__all__ = [
    "WarmupCosineScheduler",
    "compute_baseline",
    "evaluate",
    "main",
    "sequence_to_number",
    "train",
]

logger = get_logger(__name__)


class WarmupCosineScheduler:
    """Linear warmup into cosine annealing, stepped once per batch.

    Attributes:
        base_lr: Learning rate warmup ramps up to and cosine decays from.
        step_count: Batches stepped so far.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-6,
    ) -> None:
        """Build the schedule.

        Args:
            optimizer: Optimizer whose learning rate is driven.
            warmup_steps: Batches spent ramping linearly up to the base rate.
            total_steps: Total batches in the run, warmup included.
            min_lr: Floor the cosine decays to.
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = float(optimizer.param_groups[0]["lr"])
        self.step_count = 0

    def step(self) -> None:
        """Advance one batch and apply the resulting learning rate."""
        self.step_count += 1

        if self.step_count <= self.warmup_steps:
            learning_rate = self.base_lr * (self.step_count / self.warmup_steps)
        else:
            progress = (self.step_count - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            learning_rate = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (
                1 + math.cos(math.pi * progress)
            )

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = learning_rate

    def get_last_lr(self) -> float:
        """Return the learning rate currently in effect."""
        return float(self.optimizer.param_groups[0]["lr"])


def sequence_to_number(sequence: torch.Tensor, eos_token: int, start_token: int) -> int:
    """Read a token sequence back as the number it spells.

    Args:
        sequence: Token ids for one example, shape `(seq_len,)`.
        eos_token: Token that ends the sequence.
        start_token: Token the sequence was primed with.

    Returns:
        The number, with leading zeros collapsed the same way the targets are built.
    """
    number = 0
    for token in sequence.tolist():
        if token == eos_token:
            break
        if token in (0, start_token):
            continue
        number = number * 10 + int(token)
    return number


def compute_baseline(train_loader: DataLoader[Example], val_loader: DataLoader[Example]) -> float:
    """Return the validation exact-match rate of always guessing the commonest number.

    With a thousand numbers represented equally this lands near 0.1%, which is the
    point: it makes clear how far above chance a real result actually is.

    Args:
        train_loader: Split the commonest number is taken from.
        val_loader: Split the baseline is scored on.

    Returns:
        Exact-match accuracy as a percentage.
    """
    config = Config()
    arch = config.seq2seq_arch

    train_counts: Counter[int] = Counter()
    for _, targets in train_loader:
        train_counts.update(
            sequence_to_number(row, arch.eos_token, arch.start_token) for row in targets
        )
    majority_number = train_counts.most_common(1)[0][0]

    correct = 0
    total = 0
    for _, targets in val_loader:
        for row in targets:
            correct += int(
                sequence_to_number(row, arch.eos_token, arch.start_token) == majority_number
            )
            total += 1

    return 100.0 * correct / total


def evaluate(
    model: Seq2SeqVisionTransformer,
    loader: DataLoader[Example],
    device: torch.device,
    config: Config,
    description: str = "Evaluating",
) -> float:
    """Return the autoregressive exact-match accuracy over a loader.

    Args:
        model: Model to evaluate.
        loader: Data to evaluate on.
        device: Device to run on.
        config: Run configuration, for the token ids and sequence length.
        description: Progress-bar label.

    Returns:
        Exact-match accuracy as a percentage.
    """
    arch = config.seq2seq_arch
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in tqdm(loader, desc=description, leave=False):
            images, targets = images.to(device), targets.to(device)
            generated = model.generate(
                images,
                start_token=arch.start_token,
                max_length=config.decoder_seq_length,
            )
            for predicted_row, target_row in zip(generated, targets, strict=True):
                predicted = sequence_to_number(predicted_row, arch.eos_token, arch.start_token)
                expected = sequence_to_number(target_row, arch.eos_token, arch.start_token)
                correct += int(predicted == expected)
                total += 1

    return 100.0 * correct / total


def train(config: Config) -> float:
    """Train the multi-digit seq2seq model and publish the best checkpoint.

    Args:
        config: Run configuration.

    Returns:
        The best validation exact-match accuracy reached, as a percentage.
    """
    set_seed(config.seed)
    device = get_device()
    arch = config.seq2seq_arch
    logger.info("Training the multi-digit seq2seq model on %s", device)

    dataset = MultiDigitMNISTDataset(
        data_dir=config.data_dir,
        image_size=arch.img_size,
        max_digits=config.max_digits,
        train_split=config.train_split,
        samples_per_class=config.samples_per_class,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
        return_digit_sequence=True,
        add_start_eos=True,
        start_token=arch.start_token,
        eos_token=arch.eos_token,
    )
    train_loader, val_loader, test_loader = dataset.get_data_loaders()

    baseline_accuracy = compute_baseline(train_loader, val_loader)
    logger.info(
        "Baseline (predict commonest number) validation accuracy: %.2f%%", baseline_accuracy
    )

    model = Seq2SeqVisionTransformer(arch).to(device)
    logger.info("Model has %d parameters", sum(p.numel() for p in model.parameters()))

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.seq2seq_weight_decay,
    )
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=config.warmup_steps,
        total_steps=len(train_loader) * config.num_epochs,
    )

    init_tracking(
        config.wandb_project,
        config={"baseline_accuracy": baseline_accuracy, **asdict(config)},
        run_name="seq2seq",
    )

    best_accuracy = 0.0
    for epoch in range(1, config.num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        batches = 0

        for batch_index, (images, targets) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch}/{config.num_epochs}", leave=False)
        ):
            images, targets = images.to(device), targets.to(device)

            decoder_input = targets[:, :-1]
            decoder_target = targets[:, 1:]

            logits = model(images, decoder_input)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                decoder_target.reshape(-1),
                label_smoothing=config.label_smoothing,
            )
            # torch.autograd.backward rather than Tensor.backward: identical
            # behaviour, but Tensor.backward ships no annotations in torch 2.13, so
            # calling it from typed code is an unchecked call.
            torch.autograd.backward(loss / config.gradient_accumulation_steps)

            if (batch_index + 1) % config.gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), config.seq2seq_gradient_clip)
                optimizer.step()
                optimizer.zero_grad()

            scheduler.step()
            epoch_loss += loss.item()
            batches += 1

        # Flush whatever the last accumulation window did not complete.
        if batches % config.gradient_accumulation_steps != 0:
            nn.utils.clip_grad_norm_(model.parameters(), config.seq2seq_gradient_clip)
            optimizer.step()
            optimizer.zero_grad()

        train_loss = epoch_loss / batches
        val_accuracy = evaluate(model, val_loader, device, config, description="Validating")

        log_metrics(
            {
                "train_loss": train_loss,
                "val_accuracy": val_accuracy,
                "learning_rate": scheduler.get_last_lr(),
            },
            step=epoch,
        )
        logger.info(
            "Epoch %d/%d  train_loss=%.4f  val_exact_match=%.2f%%  lr=%.6f",
            epoch,
            config.num_epochs,
            train_loss,
            val_accuracy,
            scheduler.get_last_lr(),
        )

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            save_checkpoint(
                config.seq2seq_checkpoint_path,
                model,
                optimizer=optimizer,
                epoch=epoch,
                metrics={
                    "val_accuracy": val_accuracy,
                    "train_loss": train_loss,
                    "baseline_accuracy": baseline_accuracy,
                },
                extra={"arch": asdict(arch)},
            )
            if config.hf_repo_id is not None:
                push_checkpoint(config.hf_repo_id, config.seq2seq_checkpoint_path)

    test_accuracy = evaluate(model, test_loader, device, config, description="Testing")
    log_metrics({"test_accuracy": test_accuracy})
    finish_tracking()

    logger.info(
        "Best validation exact-match %.2f%% against baseline %.2f%% (%s); test exact-match %.2f%%",
        best_accuracy,
        baseline_accuracy,
        "beats baseline" if best_accuracy > baseline_accuracy else "does NOT beat baseline",
        test_accuracy,
    )
    return best_accuracy


def main() -> None:
    """Entry point for `make train-seq2seq`."""
    parser = argparse.ArgumentParser(description="Train the multi-digit seq2seq ViT")
    parser.add_argument("--epochs", type=int, help="Override the number of epochs")
    parser.add_argument("--batch-size", type=int, help="Override the batch size")
    parser.add_argument("--learning-rate", type=float, help="Override the learning rate")
    parser.add_argument(
        "--samples-per-class",
        type=int,
        help="Override how many composites are generated per distinct number",
    )
    parser.add_argument("--seed", type=int, help="Override the random seed")
    args = parser.parse_args()

    config = Config()
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.samples_per_class is not None:
        config.samples_per_class = args.samples_per_class
    if args.seed is not None:
        config.seed = args.seed

    train(config)


if __name__ == "__main__":
    main()
