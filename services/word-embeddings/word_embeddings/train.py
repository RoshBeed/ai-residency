"""Train the CBOW word embeddings.

Pretrains word vectors with negative sampling over the merged corpus (~100M tokens).
The vocabulary travels inside the checkpoint alongside the weights, because
embeddings without the exact mapping they were indexed by are meaningless.

Run ``python -m word_embeddings.train`` (add ``--sample`` to train on a tiny
built-in corpus in seconds instead of hours).
"""

import argparse

import torch
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
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from word_embeddings.data import (
    CBOWNegativeSamplingDataset,
    encode_corpus,
    load_corpus,
    load_vocabulary,
    sample_vocabulary,
)
from word_embeddings.evaluate import evaluate_embeddings
from word_embeddings.model import CBOWNS, CBOWConfig

__all__ = ["build_dataset", "main", "train_cbow"]

logger = get_logger(__name__)

Batch = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def build_dataset(
    config: Config, *, sample: bool = False
) -> tuple[CBOWNegativeSamplingDataset, dict[str, int], dict[str, int]]:
    """Assemble the CBOW training dataset and the vocabularies behind it."""
    if sample:
        word_to_index, word_to_lemma_index, text = sample_vocabulary()
    else:
        word_to_index, word_to_lemma_index = load_vocabulary(config)
        text = load_corpus(config)

    encoded = encode_corpus(text, word_to_index)
    dataset = CBOWNegativeSamplingDataset(
        encoded,
        window_size=config.cbow_window_size,
        num_negatives=config.cbow_num_negatives,
    )
    return dataset, word_to_index, word_to_lemma_index


def train_cbow(config: Config, *, sample: bool = False) -> float:
    """Train CBOW word vectors and return the best epoch loss."""
    set_seed(config.seed)
    device = get_device()

    dataset, word_to_index, word_to_lemma_index = build_dataset(config, sample=sample)
    loader: DataLoader[Batch] = DataLoader(dataset, batch_size=config.cbow_batch_size, shuffle=True)

    vocab_size = len(word_to_lemma_index)
    model_config = CBOWConfig(
        vocab_size=vocab_size,
        embed_size=config.cbow_embed_size,
        window_size=config.cbow_window_size,
        num_negatives=config.cbow_num_negatives,
    )
    model = CBOWNS(model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.cbow_learning_rate)
    logger.info("Training CBOW on %s over a %d-word vocabulary", device, vocab_size)

    init_tracking(config.wandb_project, config=vars(config), run_name="cbow")

    best_loss = float("inf")
    for epoch in range(1, config.cbow_num_epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0

        progress = tqdm(loader, desc=f"Epoch {epoch}", unit="batch")
        for batch_index, (context, target, negatives) in enumerate(progress):
            context, target, negatives = (
                context.to(device),
                target.to(device),
                negatives.to(device),
            )
            loss = model(context, target, negatives)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1
            progress.set_postfix(loss=loss.item())

            if batch_index % config.cbow_eval_every == 0:
                log_metrics(
                    {
                        "batch_loss": loss.item(),
                        **evaluate_embeddings(model.in_embed.weight.detach(), word_to_index),
                    }
                )

        epoch_loss = total_loss / max(batches, 1)
        log_metrics({"epoch_loss": epoch_loss}, step=epoch)
        logger.info("Epoch %d/%d  loss=%.4f", epoch, config.cbow_num_epochs, epoch_loss)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_checkpoint(
                config.cbow_checkpoint,
                model,
                optimizer=optimizer,
                epoch=epoch,
                metrics={"epoch_loss": epoch_loss},
                extra={
                    "word_to_index": word_to_index,
                    "word_to_lemma_index": word_to_lemma_index,
                    "config": model_config.to_dict(),
                },
            )
            if config.hf_repo_id is not None and not sample:
                push_checkpoint(config.hf_repo_id, config.cbow_checkpoint)

    finish_tracking()
    logger.info("Best CBOW loss %.4f", best_loss)
    return best_loss


def main() -> None:
    """Entry point: `python -m word_embeddings.train [--sample]`."""
    parser = argparse.ArgumentParser(description="Train CBOW word vectors")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Train on a tiny built-in corpus instead of the real one (seconds, not hours)",
    )
    args = parser.parse_args()

    if args.sample:
        # A sample run must never overwrite real embeddings, nor push them.
        config = Config(
            cbow_checkpoint=Config().cbow_checkpoint_dir / "cbow_model_sample.pt",
            hf_repo_id=None,
        )
    else:
        config = Config()

    train_cbow(config, sample=args.sample)


if __name__ == "__main__":
    main()
