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

Training is batched, and under the default `in_batch` objective the batch *is* the
source of negatives -- see `models/losses.py`. The stage stays on CPU: per-op dispatch
overhead dominates at this tensor size, and MPS benchmarked slower here.
"""

import json
import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ai_residency_common import (
    BestMetric,
    finish_tracking,
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
from cbow.vocabulary import build_embedding_layer
from config import Config
from data_prep.tokenize_triples import Triple
from models.losses import in_batch_loss, triplet_batch_loss
from models.towers import GruTower, encode, encode_batch

__all__ = [
    "TrainingResult",
    "build_towers",
    "compute_baseline",
    "load_triples",
    "main",
    "pairwise_accuracy",
    "train",
    "usable_triples",
]

logger = get_logger(__name__)


@dataclass(frozen=True)
class TrainingResult:
    """What a training run produced.

    Attributes:
        baseline_accuracy: Held-out pairwise accuracy of untrained towers.
        trained_accuracy: Held-out pairwise accuracy of the saved towers.
        baseline_loss: Mean loss of untrained towers under the same objective. A loss
            is only readable next to this: in-batch cross-entropy has no fixed scale,
            and an untrained model is not the same thing as a uniform one -- at a low
            temperature it is confidently wrong and scores *worse* than chance.
        best_loss: Lowest mean loss reached on the training triples.
        evaluated_triples: Validation triples that survived tokenization.
    """

    baseline_accuracy: float
    trained_accuracy: float
    baseline_loss: float
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


def objective_loss(
    config: Config,
    query_tower: GruTower,
    document_tower: GruTower,
    batch: list[Triple],
) -> torch.Tensor:
    """Loss for one batch under whichever objective is configured.

    One place decides what "the loss" means, so the untrained baseline and the training
    step cannot drift into measuring different quantities.

    Args:
        config: Supplies `loss`, `temperature` and `margin`.
        query_tower: Tower encoding the queries.
        document_tower: Tower encoding the documents.
        batch: Triples, already filtered by `usable_triples`.

    Returns:
        The batch's mean loss.
    """
    queries = encode_batch(query_tower, [triple["query_tokens"] for triple in batch])
    positives = encode_batch(
        document_tower, [triple["positive_document_tokens"] for triple in batch]
    )
    if config.loss == "in_batch":
        return in_batch_loss(queries, positives, config.temperature)
    negatives = encode_batch(
        document_tower, [triple["negative_document_tokens"] for triple in batch]
    )
    return triplet_batch_loss(queries, positives, negatives, config.margin)


def baseline_objective_loss(
    config: Config,
    query_tower: GruTower,
    document_tower: GruTower,
    triples: list[Triple],
    *,
    max_batches: int = 50,
) -> float:
    """Mean loss of the given towers without any training, over a sample of batches.

    Args:
        config: Supplies the objective and the batch size.
        query_tower: Untrained query tower.
        document_tower: Untrained document tower.
        triples: Training triples, already filtered.
        max_batches: Batches to average over. The point is a stable reference, not a
            full pass, and the estimate stops moving well before fifty batches.

    Returns:
        The mean loss, or 0.0 if there was nothing to score.
    """
    total = 0.0
    batches = 0
    with torch.no_grad():
        for start in range(0, len(triples), config.batch_size):
            chunk = triples[start : start + config.batch_size]
            if config.loss == "in_batch" and len(chunk) < 2:
                continue
            total += float(objective_loss(config, query_tower, document_tower, chunk))
            batches += 1
            if batches >= max_batches:
                break
    return total / max(batches, 1)


def usable_triples(triples: list[Triple]) -> list[Triple]:
    """Drop triples that cannot be encoded.

    A short query against a small vocabulary can tokenize to nothing at all, and a GRU
    cannot consume a zero-length sequence. The old loop skipped these per triple; under
    the in-batch objective they have to go before batching, because a row removed from
    the middle of a batch shifts every later row and the loss's diagonal targets then
    point at the wrong documents.

    Args:
        triples: Triples to filter.

    Returns:
        Only those whose query, positive and negative all have tokens.
    """
    return [
        triple
        for triple in triples
        if triple["query_tokens"]
        and triple["positive_document_tokens"]
        and triple["negative_document_tokens"]
    ]


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

    # A triple whose query or document tokenized to nothing cannot be encoded, and
    # under `in_batch` it cannot be quietly skipped mid-batch either: dropping row i
    # would leave the similarity matrix's diagonal pointing at the wrong document. They
    # are filtered once, before batching, rather than inside the loop.
    train_triples = usable_triples(triples["train"])
    logger.info(
        "Training on %d of %d triples (%d dropped as untokenizable), %s loss, batch %d",
        len(train_triples),
        len(triples["train"]),
        len(triples["train"]) - len(train_triples),
        config.loss,
        config.batch_size,
    )

    # The loss these towers score before a single gradient step. Every metric in this
    # repo is reported next to the number an untrained model gets for free, because a
    # falling loss with no reference point is how four separate defects stayed hidden
    # in another service's trainer.
    baseline_loss = baseline_objective_loss(config, query_tower, document_tower, train_triples)
    logger.info("Untrained %s loss: %.4f", config.loss, baseline_loss)
    log_metrics({f"baseline_{config.loss}_loss": baseline_loss})

    # The scale fingerprint: resuming a triplet run into an in-batch one, or across a
    # different batch size, would be two experiments averaged together.
    resume_file = resume_path_for(config.query_tower_path)
    fingerprint = {
        "loss": config.loss,
        "batch_size": config.batch_size,
        "hidden_size": config.hidden_size,
        "learning_rate": config.learning_rate,
        "train_triples": len(train_triples),
    }
    resumed = resume_from(
        resume_file,
        nn.ModuleList([query_tower, document_tower]),
        optimizer,
        fingerprint=fingerprint,
        default_best=BestMetric.worst_for("min"),
    )
    best = BestMetric(
        mode="min",
        patience=config.early_stopping_patience or None,
        best=resumed.best_metric,
    )

    shuffler = random.Random(config.seed)
    for epoch in range(resumed.start_epoch, config.num_epochs + 1):
        query_tower.train()
        document_tower.train()
        # Reshuffled every epoch, which matters more under `in_batch` than it looks:
        # the batch composition *is* the negative sampling, so a fixed order would show
        # every query the same 63 negatives on every pass.
        order = list(range(len(train_triples)))
        shuffler.shuffle(order)

        total_loss = 0.0
        batches = 0
        for start in range(0, len(order), config.batch_size):
            chunk = [train_triples[index] for index in order[start : start + config.batch_size]]
            # A batch of one has no in-batch negatives at all, and cross-entropy over a
            # single logit is identically zero.
            if config.loss == "in_batch" and len(chunk) < 2:
                continue

            loss = objective_loss(config, query_tower, document_tower, chunk)

            optimizer.zero_grad()
            # `torch.autograd.backward` rather than `loss.backward()`: the method is
            # untyped in the stubs and this repo carries no `type: ignore`s.
            torch.autograd.backward(loss)
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        epoch_loss = total_loss / max(batches, 1)
        log_metrics({f"train_{config.loss}_loss": epoch_loss}, step=epoch)
        logger.info("Epoch %d/%d  %s_loss=%.4f", epoch, config.num_epochs, config.loss, epoch_loss)

        # Selection is on training loss because it is the only per-epoch number this
        # loop produces. The held-out accuracy below is scored once, on whatever is
        # actually written to disk, so the reported figure always describes the saved
        # towers rather than some other epoch's weights.
        if best.improved(epoch_loss):
            for tower, path in (
                (query_tower, config.query_tower_path),
                (document_tower, config.document_tower_path),
            ):
                save_checkpoint(
                    path,
                    tower,
                    epoch=epoch,
                    metrics={f"train_{config.loss}_loss": epoch_loss},
                    extra={
                        "hidden_size": config.hidden_size,
                        "embedding_dim": config.embedding_dim,
                        "loss": config.loss,
                    },
                )
            if config.hf_repo_id is not None:
                push_checkpoint(config.hf_repo_id, config.query_tower_path)
                push_checkpoint(config.hf_repo_id, config.document_tower_path)

        # Written every epoch, not only on an improvement: the point is to survive a
        # crash mid-run, and a run that has not improved for two epochs is exactly the
        # one with the most to lose.
        save_resume(
            resume_file,
            nn.ModuleList([query_tower, document_tower]),
            optimizer,
            epoch=epoch,
            best_metric=best.best,
            fingerprint=fingerprint,
            metrics={f"train_{config.loss}_loss": epoch_loss},
        )
        if best.should_stop:
            break

    best_loss = best.best

    # Score the checkpoints that were actually saved, not the last epoch's weights.
    load_checkpoint(config.query_tower_path, query_tower)
    load_checkpoint(config.document_tower_path, document_tower)
    query_tower.eval()
    document_tower.eval()
    trained_accuracy, _ = pairwise_accuracy(query_tower, document_tower, triples["validation"])
    log_metrics({"trained_val_pairwise_accuracy": trained_accuracy})
    finish_tracking()

    logger.info(
        "Best %s loss %.4f against untrained %.4f",
        config.loss,
        best_loss,
        baseline_loss,
    )
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
        baseline_loss=baseline_loss,
        best_loss=best_loss,
        evaluated_triples=evaluated,
    )


def main() -> None:
    """Entry point for `make train-encoder`."""
    train(Config())


if __name__ == "__main__":
    main()
