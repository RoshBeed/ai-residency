"""Training loop for the single-digit classifier.

Follows the portfolio's canonical shape: seed, build loaders, compute a baseline,
start tracking, then an epoch loop that logs metrics and publishes a checkpoint
whenever validation improves.

The baseline matters more than it looks. Ten-class MNIST accuracy sounds
self-evidently good until you notice that always guessing the most common class
already scores around 11%, so a number only means something next to it.
"""

import argparse
import json
from collections import Counter
from dataclasses import asdict

import matplotlib
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from ai_residency_common import (
    finish_tracking,
    get_device,
    get_logger,
    init_tracking,
    load_checkpoint,
    log_metrics,
    push_checkpoint,
    resume_from,
    resume_path_for,
    save_checkpoint,
    save_resume,
    set_seed,
)
from ai_residency_common.figures import save_figure
from config import Config
from datasets.mnist_dataset import Example, MNISTDataset
from vit_classifier.classifier import VisionTransformer

# Select a non-interactive backend before anything imports pyplot. This module runs
# headlessly (Makefile target, Docker train stage), and the interactive macOS backend
# blocks forever inside show() waiting for a window nothing will ever close. pyplot is
# imported inside plot_history rather than here so this call is guaranteed to come
# first without an out-of-order import.
matplotlib.use("Agg")

__all__ = ["compute_baseline", "evaluate", "main", "plot_history", "train"]

logger = get_logger(__name__)


def compute_baseline(train_loader: DataLoader[Example], val_loader: DataLoader[Example]) -> float:
    """Return the validation accuracy of always predicting the commonest train class.

    Args:
        train_loader: Split the majority class is taken from.
        val_loader: Split the baseline is scored on.

    Returns:
        Accuracy as a percentage.
    """
    train_counts: Counter[int] = Counter()
    for _, targets in train_loader:
        train_counts.update(int(label) for label in targets.tolist())
    majority_class = train_counts.most_common(1)[0][0]

    correct = 0
    total = 0
    for _, targets in val_loader:
        correct += int((targets == majority_class).sum().item())
        total += targets.size(0)

    return 100.0 * correct / total


def evaluate(
    model: nn.Module,
    loader: DataLoader[Example],
    device: torch.device,
) -> tuple[float, float]:
    """Return mean loss and accuracy over a loader.

    Args:
        model: Model to evaluate.
        loader: Data to evaluate on.
        device: Device to run on.

    Returns:
        A `(loss, accuracy_percent)` pair.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            total_loss += criterion(logits, targets).item() * targets.size(0)
            correct += int((logits.argmax(dim=1) == targets).sum().item())
            total += targets.size(0)

    return total_loss / total, 100.0 * correct / total


def plot_history(history: dict[str, list[float]], path: str) -> None:
    """Write the loss and accuracy curves to a PNG.

    Args:
        history: Per-epoch series, keyed by metric name.
        path: Destination image file.
    """
    import matplotlib.pyplot as plt

    figure, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(15, 5))

    loss_axis.plot(history["train_loss"], label="Train")
    loss_axis.plot(history["val_loss"], label="Validation")
    loss_axis.set_title("Loss")
    loss_axis.set_xlabel("Epoch")
    loss_axis.legend()
    loss_axis.grid(visible=True)

    accuracy_axis.plot(history["train_accuracy"], label="Train")
    accuracy_axis.plot(history["val_accuracy"], label="Validation")
    accuracy_axis.set_title("Accuracy (%)")
    accuracy_axis.set_xlabel("Epoch")
    accuracy_axis.legend()
    accuracy_axis.grid(visible=True)

    figure.tight_layout()
    save_figure(figure, path)
    plt.close(figure)


def train(config: Config) -> float:
    """Train the single-digit classifier and publish the best checkpoint.

    Args:
        config: Run configuration.

    Returns:
        The best validation accuracy reached, as a percentage.
    """
    set_seed(config.seed)
    device = get_device()
    logger.info("Training the single-digit classifier on %s", device)

    dataset = MNISTDataset(
        data_dir=config.data_dir,
        image_size=config.classifier_arch.img_size,
        train_split=config.train_split,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
    )
    train_loader, val_loader, test_loader = dataset.get_data_loaders()

    baseline_accuracy = compute_baseline(train_loader, val_loader)
    logger.info("Baseline (predict commonest class) validation accuracy: %.2f%%", baseline_accuracy)

    model = VisionTransformer(config.classifier_arch).to(device)
    logger.info("Model has %d parameters", sum(p.numel() for p in model.parameters()))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.classifier_weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.classifier_epochs)

    # What makes two runs the same run. The epoch cap is in it because it is the
    # cosine schedule's `T_max`: the same weights under a different cap are on a
    # different learning-rate trajectory, so resuming across that is not a resume.
    fingerprint = {
        "arch": asdict(config.classifier_arch),
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "weight_decay": config.classifier_weight_decay,
        "epochs": config.classifier_epochs,
        "train_split": config.train_split,
        "gradient_clip": config.classifier_gradient_clip,
    }
    resume_path = resume_path_for(config.classifier_checkpoint_path)
    resumed = resume_from(
        resume_path,
        model,
        optimizer,
        fingerprint=fingerprint,
        default_best=0.0,
        map_location=str(device),
    )
    # The scheduler steps once per epoch, so a resumed run has to fast-forward it or it
    # restarts the cosine decay from the top and trains the rest of the run at the wrong
    # learning rate.
    for _ in range(resumed.start_epoch - 1):
        scheduler.step()

    init_tracking(
        config.wandb_project,
        config={"baseline_accuracy": baseline_accuracy, **asdict(config)},
        run_name="classifier",
    )

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }
    best_accuracy = resumed.best_metric
    # The patience counter deliberately restarts at zero on a resume. Carrying it would
    # mean persisting it, and getting it wrong in the other direction stops a run early
    # on a count it inherited; restarting it can only make a resumed run train longer,
    # and the shipped checkpoint is the ranked best either way.
    epochs_without_improvement = 0

    for epoch in range(resumed.start_epoch, config.classifier_epochs + 1):
        model.train()
        epoch_loss = 0.0
        correct = 0
        seen = 0

        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.classifier_gradient_clip)
            optimizer.step()

            epoch_loss += loss.item() * targets.size(0)
            correct += int((logits.argmax(dim=1) == targets).sum().item())
            seen += targets.size(0)

        scheduler.step()

        train_loss = epoch_loss / seen
        train_accuracy = 100.0 * correct / seen
        val_loss, val_accuracy = evaluate(model, val_loader, device)

        for name, value in (
            ("train_loss", train_loss),
            ("val_loss", val_loss),
            ("train_accuracy", train_accuracy),
            ("val_accuracy", val_accuracy),
        ):
            history[name].append(value)

        log_metrics(
            {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_accuracy": train_accuracy,
                "val_accuracy": val_accuracy,
                "learning_rate": optimizer.param_groups[0]["lr"],
            },
            step=epoch,
        )
        logger.info(
            "Epoch %d/%d  train_loss=%.4f train_acc=%.2f%%  val_loss=%.4f val_acc=%.2f%%",
            epoch,
            config.classifier_epochs,
            train_loss,
            train_accuracy,
            val_loss,
            val_accuracy,
        )

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            epochs_without_improvement = 0
            save_checkpoint(
                config.classifier_checkpoint_path,
                model,
                optimizer=optimizer,
                epoch=epoch,
                metrics={
                    "val_accuracy": val_accuracy,
                    "val_loss": val_loss,
                    "baseline_accuracy": baseline_accuracy,
                },
                extra={"arch": asdict(config.classifier_arch)},
            )
            # Only on a new best: these artifacts are large enough that pushing every
            # improvement would dominate the run.
            if config.hf_repo_id is not None:
                push_checkpoint(config.hf_repo_id, config.classifier_checkpoint_path)
        else:
            epochs_without_improvement += 1

        # Every epoch, not only on an improvement: a run that dies after a worse epoch
        # would otherwise resume from the last better one and repeat the work between.
        save_resume(
            resume_path,
            model,
            optimizer,
            epoch=epoch,
            best_metric=best_accuracy,
            fingerprint=fingerprint,
            metrics={"val_accuracy": val_accuracy, "val_loss": val_loss},
        )

        if epochs_without_improvement >= config.classifier_patience:
            logger.info(
                "Early stopping after %d epochs without improvement",
                epochs_without_improvement,
            )
            break

    # Score the checkpoint that ships, not whatever the last epoch left behind. With
    # early stopping those are `patience` epochs apart, so the test number reported for
    # the shipped model was the final model's -- a different set of weights.
    if config.classifier_checkpoint_path.exists():
        load_checkpoint(config.classifier_checkpoint_path, model, map_location=str(device))
    test_loss, test_accuracy = evaluate(model, test_loader, device)
    log_metrics({"test_loss": test_loss, "test_accuracy": test_accuracy})
    finish_tracking()

    plot_history(history, str(config.checkpoint_dir / "training_history.png"))
    (config.checkpoint_dir / "training_config.json").write_text(
        json.dumps(asdict(config), indent=2, default=str)
    )

    logger.info(
        "Best validation accuracy %.2f%% against baseline %.2f%% (%s); test accuracy %.2f%%",
        best_accuracy,
        baseline_accuracy,
        "beats baseline" if best_accuracy > baseline_accuracy else "does NOT beat baseline",
        test_accuracy,
    )
    return best_accuracy


def main() -> None:
    """Entry point for `make train-classifier`."""
    parser = argparse.ArgumentParser(description="Train the single-digit ViT classifier")
    parser.add_argument("--epochs", type=int, help="Override the epoch cap")
    parser.add_argument("--batch-size", type=int, help="Override the batch size")
    parser.add_argument("--learning-rate", type=float, help="Override the learning rate")
    parser.add_argument("--seed", type=int, help="Override the random seed")
    args = parser.parse_args()

    config = Config()
    if args.epochs is not None:
        config.classifier_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.seed is not None:
        config.seed = args.seed

    train(config)


if __name__ == "__main__":
    main()
