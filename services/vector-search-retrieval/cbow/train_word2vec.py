"""Stage 2: train CBOW word embeddings.

Was `01_train_w2v.py`. This is the one stage in the repo that is genuinely batched and
matmul-heavy, so it is also the one that gains from an accelerator: on this repo's own
shape (vocabulary ~7.7k, batch 256) MPS benchmarked around 3.7x faster than CPU. The
tower training in `training.py` does not, which is why it is left alone — see the note
there.
"""

import datetime

import torch
from torch import nn
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
from cbow.dataset import Example, Wiki
from cbow.evaluate import nearest_words
from cbow.model import CBOW
from cbow.vocabulary import CANONICAL_CBOW_CHECKPOINT
from config import Config

__all__ = ["main", "train_word2vec"]

logger = get_logger(__name__)


def train_word2vec(config: Config) -> float:
    """Train CBOW embeddings over the preprocessed corpus.

    One checkpoint is written per epoch, named with the run timestamp, because the
    tower stage downstream picks up whichever CBOW checkpoint is newest.

    Args:
        config: Run configuration.

    Returns:
        The mean loss of the final epoch.
    """
    set_seed(config.seed)
    device = get_device()
    logger.info("Training CBOW on %s", device)

    dataset = Wiki(config)
    loader: DataLoader[Example] = DataLoader(dataset, batch_size=config.cbow_batch_size)

    model = CBOW(len(dataset.ids_to_words), config.embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.cbow_learning_rate)
    criterion = nn.CrossEntropyLoss()
    logger.info("Parameters: %d", sum(p.numel() for p in model.parameters()))

    timestamp = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    init_tracking(
        config.wandb_project,
        config={"stage": "cbow", **vars(config)},
        run_name=f"cbow-{timestamp}",
    )

    epoch_loss = 0.0
    best_loss = float("inf")
    for epoch in range(1, config.cbow_num_epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        progress = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
        for index, (context, target) in enumerate(progress):
            context, target = context.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(context), target.squeeze(1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1
            if index % config.cbow_eval_every == 0:
                neighbours = nearest_words(
                    model, config.cbow_eval_word, dataset.words_to_ids, dataset.ids_to_words
                )
                logger.info("Nearest to %r: %s", config.cbow_eval_word, neighbours)

        epoch_loss = total_loss / max(batches, 1)
        log_metrics({"cbow_loss": epoch_loss}, step=epoch)
        logger.info("Epoch %d/%d  loss=%.4f", epoch, config.cbow_num_epochs, epoch_loss)

        checkpoint_path = config.cbow_checkpoint_dir / f"{timestamp}.{epoch}.cbow.pth"
        save_checkpoint(
            checkpoint_path,
            model,
            optimizer=optimizer,
            epoch=epoch,
            metrics={"cbow_loss": epoch_loss},
            extra={
                "vocab_size": len(dataset.ids_to_words),
                "embedding_dim": config.embedding_dim,
            },
        )
        # Every epoch is kept locally, because the tower stage downstream reads
        # whichever CBOW checkpoint is newest. Only an improvement is worth the upload.
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            if config.hf_repo_id is not None:
                # Upload under a fixed name, not this run's timestamped one. A puller
                # cannot guess `2026_08_22__14_31_07.5.cbow.pth`, so uploading the local
                # filename left the artifact on the Hub but unreachable -- which is why
                # a host without a local copy could not start at all.
                push_checkpoint(
                    config.hf_repo_id,
                    checkpoint_path,
                    path_in_repo=CANONICAL_CBOW_CHECKPOINT,
                )

    finish_tracking()
    return epoch_loss


def main() -> None:
    """Entry point for `make train-cbow`."""
    train_word2vec(Config())


if __name__ == "__main__":
    main()
