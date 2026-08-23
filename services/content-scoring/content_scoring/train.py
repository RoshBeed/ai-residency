"""Train the content-score predictor over frozen CBOW embeddings.

Freezes the vectors trained in the sibling word-embeddings repo (pulled from its
Hugging Face repo), averages each post's title into one embedding, and trains a
small MLP to predict its score. Early stopping matters: with 782 labelled posts the
model overfits around epoch 5.

Run ``python -m content_scoring.train``.
"""

import json
from typing import Any

import torch
from ai_residency_common import (
    finish_tracking,
    get_device,
    get_logger,
    init_tracking,
    log_metrics,
    push_checkpoint,
    resolve_checkpoint,
    resolve_required,
    save_checkpoint,
    set_seed,
)
from torch import nn
from torch.utils.data import DataLoader, random_split

from config import Config
from content_scoring.data import TitleScoreDataset, create_title_embeddings
from content_scoring.embeddings import load_cbow_embeddings, load_cbow_vocabulary
from content_scoring.evaluate import compute_baseline, evaluate_scorer
from content_scoring.model import ScorePredictor

__all__ = ["main", "train_scorer"]

logger = get_logger(__name__)


def _train_scorer_loop(
    model: nn.Module,
    train_loader: DataLoader[Any],
    test_loader: DataLoader[Any],
    config: Config,
    *,
    input_dim: int,
    baseline_mse: float,
) -> float:
    """Train the predictor, publishing only the best checkpoint."""
    device = get_device()
    logger.info("Training on %s", device)
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    init_tracking(
        config.wandb_project,
        config={"baseline_mse": baseline_mse, **vars(config)},
    )

    best_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, config.num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        examples_seen = 0

        for embeddings, scores in train_loader:
            embeddings, scores = embeddings.to(device), scores.to(device)
            optimizer.zero_grad()
            loss = criterion(model(embeddings), scores)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(embeddings)
            examples_seen += len(embeddings)

        train_loss = epoch_loss / examples_seen
        test_loss = evaluate_scorer(model, test_loader, device)
        scheduler.step(test_loss)
        learning_rate = float(optimizer.param_groups[0]["lr"])

        log_metrics(
            {"train_loss": train_loss, "test_loss": test_loss, "learning_rate": learning_rate},
            step=epoch,
        )
        logger.info(
            "Epoch %d/%d  train_loss=%.4f  test_loss=%.4f  lr=%.6f",
            epoch,
            config.num_epochs,
            train_loss,
            test_loss,
            learning_rate,
        )

        if test_loss < best_loss:
            best_loss = test_loss
            epochs_without_improvement = 0

            save_checkpoint(
                config.checkpoint_path,
                model,
                optimizer=optimizer,
                epoch=epoch,
                metrics={"train_loss": train_loss, "test_loss": test_loss},
                extra={"input_dim": input_dim},
            )

            # Sidecar kept for the UI and README, which read it directly.
            info: dict[str, Any] = {
                "epoch": epoch,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "baseline_mse": baseline_mse,
                "input_dim": input_dim,
                "max_epochs": config.num_epochs,
                "learning_rate": learning_rate,
                "model_file": str(config.checkpoint_path),
            }
            with config.info_path.open("w") as handle:
                json.dump(info, handle, indent=2)

            if config.hf_repo_id is not None:
                push_checkpoint(config.hf_repo_id, config.checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                logger.info("Early stopping after %d epochs", epoch)
                break

    finish_tracking()
    logger.info(
        "Best test loss %.4f against baseline %.4f (%s)",
        best_loss,
        baseline_mse,
        "beats baseline" if best_loss < baseline_mse else "does NOT beat baseline",
    )
    return best_loss


def train_scorer(config: Config) -> float:
    """Train the score predictor over frozen CBOW embeddings and return best test loss."""
    set_seed(config.seed)

    # The frozen CBOW checkpoint (and the vocabulary inside it) come from the
    # word-embeddings repo's Hugging Face repo unless a local copy is present.
    cbow_path, cbow_source = resolve_required(config.cbow_checkpoint, config.cbow_repo_id)
    logger.info("CBOW checkpoint from %s", cbow_source)
    word_to_index = load_cbow_vocabulary(cbow_path)
    embeddings = load_cbow_embeddings(cbow_path)
    embedding_dim = int(embeddings.shape[1])

    # The labelled posts are the one input that cannot be regenerated locally, so
    # they come from a dataset repo rather than from git.
    posts_path, _ = resolve_checkpoint(
        config.posts_file, config.hf_dataset_repo_id, repo_type="dataset"
    )
    if posts_path is None:
        raise FileNotFoundError(
            f"{config.posts_file} not found locally or in {config.hf_dataset_repo_id}"
        )

    title_embeddings, scores = create_title_embeddings(posts_path, word_to_index, embeddings)
    logger.info("Created embeddings for %d titles", len(title_embeddings))

    with posts_path.open() as handle:
        posts: list[dict[str, Any]] = json.load(handle)

    dataset = TitleScoreDataset(title_embeddings, scores, posts)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_subset, test_subset = random_split(dataset, [train_size, test_size])
    logger.info("Split into %d training and %d test examples", train_size, test_size)

    baseline_mse = compute_baseline(dataset, train_subset, test_subset)

    train_loader: DataLoader[Any] = DataLoader(
        train_subset, batch_size=config.batch_size, shuffle=True
    )
    test_loader: DataLoader[Any] = DataLoader(test_subset, batch_size=config.batch_size)

    model = ScorePredictor(input_dim=embedding_dim)
    return _train_scorer_loop(
        model,
        train_loader,
        test_loader,
        config,
        input_dim=embedding_dim,
        baseline_mse=baseline_mse,
    )


def main() -> None:
    """Entry point: `python -m content_scoring.train`."""
    train_scorer(Config())


if __name__ == "__main__":
    main()
