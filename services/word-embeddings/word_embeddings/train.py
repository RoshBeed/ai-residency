"""Train word vectors on text8.

Pretrains CBOW or skip-gram vectors with negative sampling. The vocabulary travels
inside the checkpoint alongside the weights, because embeddings without the exact
mapping they were indexed by are meaningless -- and for a long time this service
saved a *different* mapping from the one the corpus was encoded with, so every
neighbour it served was read out of the wrong index space.

    python -m word_embeddings.train                # a real run
    python -m word_embeddings.train --sample       # tiny corpus, seconds
    python -m word_embeddings.train --time-only    # measure throughput, train nothing

Metrics are logged in two families with separate x-axes: `train/*` against
`train/step`, `eval/*` against `epoch`. That is not cosmetic. Mixing unstepped
per-batch logs with `step=epoch` per-epoch logs, as this trainer used to, advances
W&B's shared counter into the thousands and then silently drops every epoch metric
for being behind it.
"""

import argparse
import json
import math
import time
from dataclasses import asdict

import torch
from tqdm import tqdm

from ai_residency_common import (
    define_metrics,
    finish_tracking,
    get_device,
    get_logger,
    init_tracking,
    log_metrics,
    log_table,
    push_checkpoint,
    save_checkpoint,
    set_seed,
    set_summary,
)
from config import Config
from word2vec import Word2Vec, nearest_words
from word_embeddings.benchmarks import evaluate_embeddings, load_benchmarks
from word_embeddings.corpus import Corpus, load_corpus
from word_embeddings.data import Word2VecBatcher
from word_embeddings.model import build_word2vec

__all__ = ["main", "measure_throughput", "random_baseline", "train"]

logger = get_logger(__name__)

_WARMUP_FRACTION = 0.05


def random_baseline(config: Config, vocab_size: int) -> float:
    """The loss an untrained model reports, before any learning has happened.

    Args:
        config: Supplies `num_negatives` and the architecture.
        vocab_size: Unused, but kept so the signature reads as the baseline of a
            particular model rather than of a formula.

    Returns:
        `positives * ln 2 + num_negatives * ln 2`. Both embedding matrices start at
        or near zero, so every score is 0 and every `logsigmoid` term is `ln 0.5`.

    Without this number a loss curve says nothing: the checkpoint this service
    shipped for months reported 10.02, which reads like a trained model until you
    know the untrained value was 11.09.
    """
    positives = 1 if config.architecture == "cbow" else 2 * config.window_size
    return (positives + positives * config.num_negatives) * math.log(2)


def _learning_rate(step: int, total: int, base: float) -> float:
    """Linear warmup then linear decay to zero, the usual word2vec schedule."""
    warmup = max(1, int(total * _WARMUP_FRACTION))
    if step < warmup:
        return base * step / warmup
    remaining = max(0.0, (total - step) / max(1, total - warmup))
    return base * remaining


def _probe_table(embeddings: torch.Tensor, corpus: Corpus, config: Config) -> list[list[str]]:
    """Nearest neighbours for a few fixed words, as table rows.

    The cheapest honest signal that a run is working. A loss curve falls for a model
    that has learned only word frequency; `king -> queen, prince, throne` does not
    happen by accident.
    """
    index_to_word = corpus.index_to_word
    rows = []
    for word in config.probe_words:
        if word not in corpus.word_to_index:
            continue
        neighbours = nearest_words(
            embeddings,
            embeddings[corpus.word_to_index[word]],
            index_to_word,
            k=5,
            exclude=word,
        )
        rows.append([word, ", ".join(f"{other} {score:.2f}" for other, score in neighbours)])
    return rows


def measure_throughput(config: Config, *, steps: int = 200) -> dict[str, float]:
    """Time `steps` optimizer steps and project a full run from them.

    Args:
        config: The run to measure.
        steps: Steps to time, after a short warmup.

    Returns:
        Measured `steps_per_second` and `tokens_per_second`, plus the projected
        `epoch_minutes` and `run_minutes`.
    """
    set_seed(config.seed)
    device = get_device()
    corpus = load_corpus(config)
    batcher = Word2VecBatcher(corpus, config, device=device)
    model = build_word2vec(corpus.vocab_size, config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    batches = batcher.epoch()
    timed = 0
    started = 0.0
    for index, (centre, context, negatives) in enumerate(batches):
        if index == 10:  # warmup: the first steps pay lazy device and kernel setup
            _synchronise(device)
            started = time.perf_counter()
        loss = model(centre, context, negatives)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if index >= 10:
            timed += 1
        if timed == steps:
            break
    _synchronise(device)

    elapsed = time.perf_counter() - started
    steps_per_second = timed / elapsed
    epoch_minutes = len(batcher) / steps_per_second / 60
    result = {
        "steps_per_second": steps_per_second,
        "tokens_per_second": steps_per_second * config.batch_size,
        "epoch_minutes": epoch_minutes,
        "run_minutes": epoch_minutes * config.num_epochs,
    }
    logger.info(
        "%s on %s, batch %d, dim %d: %.1f steps/s, %.0fk tokens/s, "
        "%.1f min/epoch, %.1f min for %d epochs",
        config.architecture,
        device,
        config.batch_size,
        config.embed_size,
        steps_per_second,
        steps_per_second * config.batch_size / 1000,
        epoch_minutes,
        result["run_minutes"],
        config.num_epochs,
    )
    return result


def _synchronise(device: torch.device) -> None:
    """Wait for queued device work, so a timing measurement is not of the queue."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def train(config: Config, *, corpus: Corpus | None = None) -> dict[str, float]:
    """Train word vectors and return the best epoch's metrics.

    Args:
        config: Hyperparameters, paths and tracking settings.
        corpus: A prepared corpus, for tests and for reusing one across runs.

    Returns:
        The benchmark metrics of the best epoch, by composite `score`.
    """
    set_seed(config.seed)
    device = get_device()
    generator = torch.Generator().manual_seed(config.seed)

    corpus = corpus if corpus is not None else load_corpus(config)
    benchmarks = load_benchmarks(config)
    batcher = Word2VecBatcher(corpus, config, device=device, generator=generator)

    model = build_word2vec(corpus.vocab_size, config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    total_steps = len(batcher) * config.num_epochs
    logger.info(
        "Training %s on %s: %d words, %d dims, %d steps over %d epochs",
        config.architecture,
        device,
        corpus.vocab_size,
        config.embed_size,
        total_steps,
        config.num_epochs,
    )

    init_tracking(
        config.wandb_project,
        config={
            "random_baseline_loss": random_baseline(config, corpus.vocab_size),
            **asdict(config),
        },
        run_name=f"{config.architecture}-d{config.embed_size}-w{config.window_size}",
    )
    define_metrics({"train/*": "train/step", "eval/*": "epoch"})

    # Epoch 0 is the untrained model, so every eval curve carries its own baseline
    # rather than needing one remembered from another run.
    baseline = evaluate_embeddings(
        model.embeddings.detach().cpu(),
        corpus.word_to_index,
        benchmarks,
        max_analogy_vocab=config.max_analogy_vocab,
    )
    log_metrics({"epoch": 0, **{f"eval/{k}": v for k, v in baseline.items()}})
    logger.info("Untrained baseline: %s", _format(baseline))

    step = 0
    best: dict[str, float] = {}
    for epoch in range(1, config.num_epochs + 1):
        model.train()
        running: torch.Tensor | float = 0.0
        counted = 0
        clock = time.perf_counter()

        progress = tqdm(batcher.epoch(), total=len(batcher), desc=f"Epoch {epoch}", unit="batch")
        for centre, context, negatives in progress:
            learning_rate = _learning_rate(step, total_steps, config.learning_rate)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate

            loss = model(centre, context, negatives)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulated on-device. `loss.item()` here would block until the queued
            # MPS work drains, turning an asynchronous pipeline into a synchronous
            # one every single step; it is read once per log instead.
            running += loss.detach()
            counted += 1
            step += 1

            if step % config.eval_every == 0:
                elapsed = time.perf_counter() - clock
                log_metrics(
                    {
                        "train/step": step,
                        "train/loss": float(running) / counted,
                        "train/lr": learning_rate,
                        "train/tokens_per_sec": counted * config.batch_size / max(elapsed, 1e-9),
                    }
                )
                progress.set_postfix(loss=float(running) / counted)
                running, counted, clock = 0.0, 0, time.perf_counter()

        embeddings = model.embeddings.detach().cpu()
        metrics = evaluate_embeddings(
            embeddings, corpus.word_to_index, benchmarks, max_analogy_vocab=config.max_analogy_vocab
        )
        log_metrics({"epoch": epoch, **{f"eval/{key}": value for key, value in metrics.items()}})
        log_table("eval/neighbours", ["word", "nearest"], _probe_table(embeddings, corpus, config))
        logger.info("Epoch %d/%d  %s", epoch, config.num_epochs, _format(metrics))

        # `not best` rather than a -inf sentinel: an unscoreable epoch gives NaN, and
        # every comparison against NaN is False, so a sentinel would mean a run whose
        # benchmarks are all uncovered never writes a checkpoint at all.
        if not best or metrics["score"] > best["score"]:
            best = {**metrics, "epoch": float(epoch)}
            if config.save_checkpoints:
                _save(config, model, corpus, epoch, metrics)

    # Pushed once, after the loop, rather than on every improvement. A 20-epoch run
    # improved seven times and uploaded 1.2GB of intermediate checkpoints that nothing
    # would ever fetch -- only the last one is reachable under a stable filename.
    if config.save_checkpoints and config.hf_repo_id is not None and best:
        push_checkpoint(config.hf_repo_id, config.embedding_checkpoint)

    set_summary({f"best/{key}": value for key, value in best.items()})
    finish_tracking()
    logger.info("Best epoch %d: %s", int(best["epoch"]), _format(best))
    return best


def _format(metrics: dict[str, float]) -> str:
    """One-line metric summary for the log, benchmarks first."""
    return "  ".join(
        f"{key}={metrics[key]:.4f}"
        for key in ("wordsim353_spearman", "simlex999_spearman", "analogy_total", "score")
        if key in metrics
    )


def _save(
    config: Config,
    model: Word2Vec,
    corpus: Corpus,
    epoch: int,
    metrics: dict[str, float],
) -> None:
    """Write the checkpoint locally. Pushing happens once, after the run.

    `extra` carries `word_to_index` -- the mapping the corpus was actually encoded
    with. It used to carry `word_to_lemma_index` instead, a different index space
    entirely, which is the bug this whole rewrite exists to fix.
    """
    # No optimizer state. Adam's two moment buffers are the same size as the two
    # embedding matrices, so carrying them tripled the artifact -- 515MB where 171MB
    # says everything -- and nothing reads them: serving and content-scoring want
    # `in_embed.weight`, and a run that is worth keeping has finished.
    save_checkpoint(
        config.embedding_checkpoint,
        model,
        epoch=epoch,
        metrics=metrics,
        extra={
            "word_to_index": corpus.word_to_index,
            "counts": corpus.counts,
            "architecture": config.architecture,
            "config": asdict(model.config),
        },
    )
    info = config.embedding_checkpoint.with_suffix(".json")
    info.write_text(json.dumps({"epoch": epoch, **metrics}, indent=2), encoding="utf-8")


def _sample_config(config: Config) -> Config:
    """A tiny-corpus configuration, for smoke runs that must not touch real vectors."""
    return Config(
        architecture=config.architecture,
        embed_size=32,
        window_size=2,
        num_negatives=3,
        min_count=1,
        batch_size=8,
        num_epochs=2,
        eval_every=5,
        checkpoint_dir=config.checkpoint_dir / "sample",
        hf_repo_id=None,
        wandb_project=None,
    )


def main() -> None:
    """Entry point: `python -m word_embeddings.train [--sample|--time-only]`."""
    parser = argparse.ArgumentParser(description="Train word vectors on text8")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Train on a tiny slice of the corpus instead of all of it (seconds, not minutes)",
    )
    parser.add_argument(
        "--time-only",
        type=int,
        nargs="?",
        const=200,
        metavar="STEPS",
        help="Measure throughput over STEPS steps and exit without training",
    )
    args = parser.parse_args()

    config = Config()
    if args.time_only:
        measure_throughput(config, steps=args.time_only)
        return

    if args.sample:
        config = _sample_config(config)
        corpus = load_corpus(config)
        corpus = Corpus(corpus.ids[:20_000], corpus.word_to_index, corpus.counts)
        train(config, corpus=corpus)
        return

    train(config)


if __name__ == "__main__":
    main()
