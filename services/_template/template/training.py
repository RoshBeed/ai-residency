"""Training loop.

This is the canonical shape every repo in the portfolio follows:

    Config -> set_seed -> baseline -> init_tracking
           -> epoch loop -> log_metrics
           -> on new best: save_checkpoint + push_checkpoint
           -> finish_tracking

Two parts of that are deliberate rather than incidental. `set_seed` runs before
anything touches the RNG, because unseeded runs in this portfolio were hiding real
bugs, not just producing wobbly numbers. And a baseline is computed before training
starts, because a loss with nothing to compare it against says very little — in one
repo the trained model turned out not to beat predicting the mean.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader

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
from template.data import Example, build_datasets
from template.models import Predictor

__all__ = ["compute_baseline", "evaluate", "main", "train"]

logger = get_logger(__name__)


def compute_baseline(train_loader: DataLoader[Example], val_loader: DataLoader[Example]) -> float:
    """Return the validation loss of predicting the training mean for every example.

    This is the number a trained model has to beat to have learned anything at all.
    Both splits are read through their loaders so the baseline is scored on exactly
    the data the model is scored on.

    Args:
        train_loader: Split the mean is taken from.
        val_loader: Split the baseline is scored on.

    Returns:
        Mean squared error of the constant prediction.
    """
    train_targets = torch.cat([targets for _, targets in train_loader])
    val_targets = torch.cat([targets for _, targets in val_loader])
    prediction = train_targets.mean()
    return float(((val_targets - prediction) ** 2).mean().item())


def evaluate(model: nn.Module, loader: DataLoader[Example], device: torch.device) -> float:
    """Return the model's mean loss over a loader.

    Args:
        model: Model to evaluate.
        loader: Data to evaluate on.
        device: Device to run on.

    Returns:
        Mean squared error across the loader.
    """
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    total_examples = 0

    with torch.no_grad():
        for features, targets in loader:
            features, targets = features.to(device), targets.to(device)
            loss = criterion(model(features), targets)
            total_loss += loss.item() * len(features)
            total_examples += len(features)

    return total_loss / total_examples


def train(config: Config) -> float:
    """Train the model, tracking metrics and publishing the best checkpoint.

    Args:
        config: Run configuration.

    Returns:
        The best validation loss reached.
    """
    set_seed(config.seed)
    device = get_device()
    logger.info("Training on %s", device)

    train_dataset, val_dataset = build_datasets(config)
    train_loader: DataLoader[Example] = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True
    )
    val_loader: DataLoader[Example] = DataLoader(val_dataset, batch_size=config.batch_size)

    baseline_loss = compute_baseline(train_loader, val_loader)
    logger.info("Baseline (predict train mean) validation loss: %.4f", baseline_loss)

    model = Predictor(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    init_tracking(
        config.wandb_project,
        config={"baseline_loss": baseline_loss, **vars(config)},
    )

    best_loss = float("inf")
    for epoch in range(1, config.num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        examples_seen = 0
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(features), targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(features)
            examples_seen += len(features)

        train_loss = epoch_loss / examples_seen
        val_loss = evaluate(model, val_loader, device)
        log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)
        logger.info(
            "Epoch %d/%d  train_loss=%.4f  val_loss=%.4f",
            epoch,
            config.num_epochs,
            train_loss,
            val_loss,
        )

        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(
                config.checkpoint_path,
                model,
                optimizer=optimizer,
                epoch=epoch,
                metrics={"val_loss": val_loss, "baseline_loss": baseline_loss},
                extra={"input_dim": config.input_dim, "hidden_dim": config.hidden_dim},
            )
            # Only on a new best: these artifacts are large and pushing every
            # improvement would dominate the run.
            if config.hf_repo_id is not None:
                push_checkpoint(config.hf_repo_id, config.checkpoint_path)

    finish_tracking()
    logger.info(
        "Best validation loss %.4f against baseline %.4f (%s)",
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
