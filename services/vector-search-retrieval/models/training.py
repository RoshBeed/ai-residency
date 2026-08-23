"""Stage 3: train the query and document towers jointly with triplet loss.

Was `src/models/simple_dual_encoder_rnn.py`. It follows the portfolio's training shape:

    set_seed -> load data -> compute_baseline -> init_tracking
             -> epoch loop -> log_metrics
             -> on new best: save_checkpoint + push_checkpoint
             -> finish_tracking

The baseline is the part worth keeping intact. A second, never-trained pair of towers —
same architecture, same frozen embeddings, random GRU weights — is scored on the held-out
validation triples before training starts. It answers the question a falling triplet loss
cannot: how much of the final number is training, and how much comes free from the CBOW
embeddings underneath? At this scale the answer is uncomfortable, around 52% against an
untrained 50%, and that is the honest result. The towers memorise 500 triples rather than
learning to rank unseen ones. Reporting it is the point; it is not a number to tune away.

The towers are trained one triple at a time. That is genuinely slow and genuinely not
batched, and it is why this stage stays on CPU while the CBOW stage uses MPS: per-op
dispatch overhead dominates at this tensor size, and MPS benchmarked slower here.
"""

import json
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from ai_residency_common import (
    finish_tracking,
    get_logger,
    init_tracking,
    load_checkpoint,
    log_metrics,
    push_checkpoint,
    save_checkpoint,
    set_seed,
)
from torch import nn

from cbow.vocabulary import build_embedding_layer
from config import Config
from data_prep.tokenize_triples import Triple
from models.towers import GruTower, encode

__all__ = [
    "TrainingResult",
    "build_towers",
    "compute_baseline",
    "load_triples",
    "main",
    "pairwise_accuracy",
    "train",
]

logger = get_logger(__name__)


@dataclass(frozen=True)
class TrainingResult:
    """What a training run produced.

    Attributes:
        baseline_accuracy: Held-out pairwise accuracy of untrained towers.
        trained_accuracy: Held-out pairwise accuracy of the saved towers.
        best_loss: Lowest mean triplet loss reached on the training triples.
        evaluated_triples: Validation triples that survived tokenization.
    """

    baseline_accuracy: float
    trained_accuracy: float
    best_loss: float
    evaluated_triples: int


def load_triples(config: Config) -> dict[str, list[Triple]]:
    """Load the tokenized MS MARCO triples.

    Args:
        config: Supplies `tokenized_triples_path`.

    Returns:
        Triples keyed by split.

    Raises:
        FileNotFoundError: If the data-prep stage has not been run.
    """
    path = config.tokenized_triples_path
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build the triples first with `make train-triples`."
        )
    with path.open() as handle:
        triples: dict[str, list[Triple]] = json.load(handle)
    return triples


def build_towers(
    config: Config, embedding_layer: nn.Embedding | None = None
) -> tuple[GruTower, GruTower]:
    """Build a fresh query and document tower over the frozen CBOW embeddings.

    Args:
        config: Supplies `hidden_size` and the embedding paths.
        embedding_layer: Word embeddings to sit on. Loaded from the newest CBOW
            checkpoint when omitted. `train` passes one in so that the baseline towers
            and the trained towers share a single layer: it saves reading a checkpoint
            twice, and it keeps the RNG draws identical to the run whose numbers the
            README reports, since constructing an `nn.Embedding` consumes the generator
            even when its weights are immediately overwritten.

    Returns:
        A `(query_tower, document_tower)` pair.
    """
    if embedding_layer is None:
        embedding_layer = build_embedding_layer(config)
    return (
        GruTower(embedding_layer, config.hidden_size),
        GruTower(embedding_layer, config.hidden_size),
    )


def pairwise_accuracy(
    query_tower: GruTower,
    document_tower: GruTower,
    triples: list[Triple],
) -> tuple[float, int]:
    """Fraction of triples where the query is closer to its positive document.

    A ranking metric rather than a loss: for each triple, ask whether
    `cos(query, positive) > cos(query, negative)`. Chance is 0.5, meaning the towers
    cannot tell a relevant document from an irrelevant one at all.

    Args:
        query_tower: Tower encoding the query.
        document_tower: Tower encoding both documents.
        triples: Triples to score, normally the held-out validation split.

    Returns:
        The accuracy and the number of triples it was computed over.
    """
    correct = 0
    total = 0
    with torch.no_grad():
        for triple in triples:
            query = encode(query_tower, triple["query_tokens"])
            positive = encode(document_tower, triple["positive_document_tokens"])
            negative = encode(document_tower, triple["negative_document_tokens"])
            if query is None or positive is None or negative is None:
                continue
            similarity_positive = F.cosine_similarity(query.unsqueeze(0), positive.unsqueeze(0))
            similarity_negative = F.cosine_similarity(query.unsqueeze(0), negative.unsqueeze(0))
            correct += int((similarity_positive > similarity_negative).item())
            total += 1
    return correct / max(total, 1), total


def compute_baseline(
    config: Config, triples: list[Triple], embedding_layer: nn.Embedding | None = None
) -> tuple[float, int]:
    """Score a never-trained pair of towers on the held-out triples.

    Args:
        config: Run configuration.
        triples: Held-out triples to score against.
        embedding_layer: Word embeddings to sit on, shared with the trained towers.

    Returns:
        The baseline accuracy and the number of triples scored.
    """
    query_tower, document_tower = build_towers(config, embedding_layer)
    query_tower.eval()
    document_tower.eval()
    return pairwise_accuracy(query_tower, document_tower, triples)


def train(config: Config) -> TrainingResult:
    """Train both towers and persist the best pair.

    Args:
        config: Run configuration.

    Returns:
        Baseline and trained held-out accuracy, plus the best training loss.
    """
    set_seed(config.seed)
    triples = load_triples(config)
    embedding_layer = build_embedding_layer(config)

    baseline_accuracy, evaluated = compute_baseline(config, triples["validation"], embedding_layer)
    logger.info(
        "Untrained baseline pairwise accuracy on %d held-out triples: %.4f",
        evaluated,
        baseline_accuracy,
    )

    query_tower, document_tower = build_towers(config, embedding_layer)
    # Only the GRUs are optimized; the CBOW embeddings underneath stay frozen.
    parameters = list(query_tower.rnn.parameters()) + list(document_tower.rnn.parameters())
    optimizer = torch.optim.Adam(parameters, lr=config.learning_rate)

    init_tracking(
        config.wandb_project,
        config={"stage": "towers", "baseline_accuracy": baseline_accuracy, **vars(config)},
    )
    log_metrics({"baseline_val_pairwise_accuracy": baseline_accuracy})

    best_loss = float("inf")
    for epoch in range(1, config.num_epochs + 1):
        query_tower.train()
        document_tower.train()
        total_loss = 0.0
        seen = 0

        for triple in triples["train"]:
            query = encode(query_tower, triple["query_tokens"])
            positive = encode(document_tower, triple["positive_document_tokens"])
            negative = encode(document_tower, triple["negative_document_tokens"])
            if query is None or positive is None or negative is None:
                continue

            similarity_positive = F.cosine_similarity(query.unsqueeze(0), positive.unsqueeze(0))
            similarity_negative = F.cosine_similarity(query.unsqueeze(0), negative.unsqueeze(0))
            loss = torch.clamp(config.margin - (similarity_positive - similarity_negative), min=0.0)

            optimizer.zero_grad()
            torch.autograd.backward(loss)
            optimizer.step()

            total_loss += loss.item()
            seen += 1

        epoch_loss = total_loss / max(seen, 1)
        log_metrics({"train_triplet_loss": epoch_loss}, step=epoch)
        logger.info("Epoch %d/%d  triplet_loss=%.4f", epoch, config.num_epochs, epoch_loss)

        # Selection is on training loss because it is the only per-epoch number this
        # loop produces. The held-out accuracy below is scored once, on whatever is
        # actually written to disk, so the reported figure always describes the saved
        # towers rather than some other epoch's weights.
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            for tower, path in (
                (query_tower, config.query_tower_path),
                (document_tower, config.document_tower_path),
            ):
                save_checkpoint(
                    path,
                    tower,
                    epoch=epoch,
                    metrics={"train_triplet_loss": epoch_loss},
                    extra={
                        "hidden_size": config.hidden_size,
                        "embedding_dim": config.embedding_dim,
                    },
                )
            if config.hf_repo_id is not None:
                push_checkpoint(config.hf_repo_id, config.query_tower_path)
                push_checkpoint(config.hf_repo_id, config.document_tower_path)

    # Score the checkpoints that were actually saved, not the last epoch's weights.
    load_checkpoint(config.query_tower_path, query_tower)
    load_checkpoint(config.document_tower_path, document_tower)
    query_tower.eval()
    document_tower.eval()
    trained_accuracy, _ = pairwise_accuracy(query_tower, document_tower, triples["validation"])
    log_metrics({"trained_val_pairwise_accuracy": trained_accuracy})
    finish_tracking()

    logger.info(
        "Trained pairwise accuracy on %d held-out triples: %.4f (untrained baseline %.4f) — %s",
        evaluated,
        trained_accuracy,
        baseline_accuracy,
        "beats baseline" if trained_accuracy > baseline_accuracy else "does NOT beat baseline",
    )
    return TrainingResult(
        baseline_accuracy=baseline_accuracy,
        trained_accuracy=trained_accuracy,
        best_loss=best_loss,
        evaluated_triples=evaluated,
    )


def main() -> None:
    """Entry point for `make train-encoder`."""
    train(Config())


if __name__ == "__main__":
    main()
