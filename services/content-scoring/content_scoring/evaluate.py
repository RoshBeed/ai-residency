"""The evaluation harness: one computation, two renderings.

`make evaluate` prints these numbers and `train.py` streams the same objects to W&B, so
a chart and the README table cannot drift apart -- there is nowhere for them to drift.
Everything the write-up quotes comes from here.

What it measures, and why each one is here:

**Spearman ρ** is the headline. The task is "will this post do well", which is a
ranking question; a model that predicts every score 20% low ranks perfectly and has an
RMSE that says it failed.

**Per-day NDCG@10 and precision@10** are the honest version of that. A global ρ over
three months rewards a model for learning that scores drifted upward, which is real but
is not what anyone wants to know. Ranking *within each day* controls for drift by
construction: it asks whether, among the posts competing for the same front page, the
model picks the ones that made it.

**Five baselines**, printed as rows of the same table. A metric without its floor is how
this repo once shipped a model reporting a loss of 10.02 against an untrained 11.09. The
TF-IDF + Ridge row is the one that matters most: it is a strong, cheap, entirely
conventional text model, and if the neural work does not beat it then the neural work
did not earn its keep.

**Slices** say where the model is wrong rather than only how wrong. One ρ over 69,275
posts hides everything interesting, and the OOV slice is what says whether extending the
vocabulary helped the titles it was built for or merely added rows.
"""

import argparse
import json
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, roc_auc_score

from ai_residency_common import get_device, get_logger, load_checkpoint, resolve_checkpoint
from config import Config, Fusion, Objective
from content_scoring.data import PostBatch, Splits, iterate_batches
from content_scoring.model import ScorePredictor, build_model, load_predictor_weights

__all__ = [
    "PROBE_TITLES",
    "VIRAL_THRESHOLD",
    "baseline_predictions",
    "evaluate_predictions",
    "leave_one_group_out",
    "predict",
    "score_extended_embeddings",
    "slice_table",
    "to_upvotes",
]

logger = get_logger(__name__)

PROBE_TITLES: tuple[tuple[str, str], ...] = (
    ("Show HN: I built a CLI that writes your commit messages", "github.com"),
    ("Show HN: A database that runs entirely in the browser", "github.com"),
    ("Ask HN: What are you working on this weekend?", ""),
    ("Ask HN: Is anyone else burned out?", ""),
    ("Attention Is All You Need", "arxiv.org"),
    ("A formal proof of the Kepler conjecture [pdf]", "arxiv.org"),
    ("Rust 1.90 released", "blog.rust-lang.org"),
    ("PostgreSQL 18 is out", "postgresql.org"),
    ("The unreasonable effectiveness of mathematics", "en.wikipedia.org"),
    ("Why we moved off Kubernetes", "example-startup-blog.com"),
    ("Our Series B", "example-startup-blog.com"),
    ("10 JavaScript tricks you won't believe", "medium.com"),
    ("This one weird trick will change your life", "medium.com"),
    ("Google lays off 12,000 employees", "nytimes.com"),
    ("Apple announces the M5 chip", "apple.com"),
    ("An intuitive explanation of the Fourier transform", "betterexplained.com"),
    ("I rewrote it in Rust and it got slower", "example-personal-site.dev"),
    ("The 2026 Advent of Code has begun", "adventofcode.com"),
    ("SQLite is not a toy database", "sqlite.org"),
    ("Tell HN: My startup failed. Here is everything I learned", ""),
)
"""Fixed probes, logged every epoch. The cheapest "is this learning anything" signal
there is: if a plausible `Show HN` and a piece of clickbait get the same number, the
model has learned the corpus mean and nothing else."""


def _spearman(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Spearman ρ, or NaN when there is nothing to correlate.

    Both sides are checked for being constant, not just the prediction. A model that
    predicts the mean has no ranking, and reporting 0.0 for it would be a claim rather
    than the truth -- but so does the bottom score quartile, where a quarter of Hacker
    News posts all scored exactly 1 and there is no order to recover. Guarding only the
    prediction leaves scipy to warn and return NaN anyway, one line per slice per epoch.
    """
    if len(predicted) < 3:
        return float("nan")
    if np.all(predicted == predicted[0]) or np.all(actual == actual[0]):
        return float("nan")
    correlation, _ = spearmanr(predicted, actual)
    return float(correlation)


def _ndcg_at_k(predicted: np.ndarray, actual: np.ndarray, k: int) -> float:
    """NDCG@k with the raw score as the gain.

    Args:
        predicted: Predicted values, higher is better.
        actual: True scores, used as gains.
        k: Cutoff.

    Returns:
        Discounted cumulative gain of the model's top k over the best achievable, or
        NaN when the group is smaller than the cutoff.
    """
    if len(predicted) < k:
        return float("nan")
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    chosen = actual[np.argsort(-predicted)[:k]]
    ideal = np.sort(actual)[::-1][:k]
    best = float((ideal * discounts).sum())
    return float((chosen * discounts).sum() / best) if best > 0 else float("nan")


def _per_day_ranking(frame: pd.DataFrame, predicted: np.ndarray, k: int = 10) -> dict[str, float]:
    """NDCG@k and precision@k averaged over calendar days.

    Args:
        frame: The split, carrying `posted_at` and `score`.
        predicted: Predictions aligned to `frame`.
        k: Cutoff.

    Returns:
        `ndcg@k` and `precision@k`.

    Precision here is the fraction of the model's daily top k that is genuinely in the
    day's true top k -- a number a reader can hold: "of the ten posts it flags, how many
    were actually among the day's ten best".
    """
    days = frame["posted_at"].dt.floor("D").to_numpy()
    actual = frame["score"].to_numpy(dtype=np.float64)
    ndcgs: list[float] = []
    precisions: list[float] = []
    for day in np.unique(days):
        rows = days == day
        if int(rows.sum()) < 2 * k:
            continue
        day_predicted = predicted[rows]
        day_actual = actual[rows]
        ndcgs.append(_ndcg_at_k(day_predicted, day_actual, k))
        chosen = set(np.argsort(-day_predicted)[:k].tolist())
        best = set(np.argsort(-day_actual)[:k].tolist())
        precisions.append(len(chosen & best) / k)
    return {
        f"ndcg@{k}": float(np.nanmean(ndcgs)) if ndcgs else float("nan"),
        f"precision@{k}": float(np.mean(precisions)) if precisions else float("nan"),
    }


def evaluate_predictions(
    frame: pd.DataFrame, predicted: np.ndarray, actual: np.ndarray
) -> dict[str, float]:
    """The headline metric block, in upvotes.

    Args:
        frame: The split the predictions belong to.
        predicted: Predicted **upvotes**.
        actual: True **upvotes**.

    Returns:
        Accuracy on the upvote scale, ranking quality, and calibration.

    Three families, because "predict the score" and "rank the posts" are different
    questions and a model can be good at one while useless at the other -- which is
    exactly what the first version of this service was.

    * **Accuracy in upvotes**: `mae`, `rmse`, and `mdae`, the median absolute error.
      The median is the one to read: with a handful of 3,000-point posts in the split,
      a mean error is mostly a report on how the tail went.
    * **Calibration**: `total_ratio` -- predicted upvotes over actual, summed across the
      split. A model that ranks perfectly but predicts a fifth of the traffic scores 0.2
      here, and nothing else in this block would have caught that.
    * **Ranking**: Spearman, per-day NDCG@10 and precision@10, top-decile lift. These
      are invariant to any monotone rescaling, which is why they cannot see a
      calibration failure and why the block needs the rows above them.
    """
    predicted = np.maximum(predicted, 0.0)
    log_predicted, log_actual = np.log1p(predicted), np.log1p(actual)
    residual = predicted - actual
    log_residual = log_predicted - log_actual
    variance = float(np.var(log_actual))
    cutoff = max(1, len(predicted) // 10)
    top = actual[np.argsort(-predicted)[:cutoff]]
    overall = float(actual.mean())
    total = float(actual.sum())

    metrics = {
        "mae": float(np.abs(residual).mean()),
        "rmse": float(np.sqrt((residual**2).mean())),
        "mdae": float(np.median(np.abs(residual))),
        "total_ratio": float(predicted.sum() / total) if total > 0 else float("nan"),
        "mean_predicted": float(predicted.mean()),
        "max_predicted": float(predicted.max()),
        "log_rmse": float(np.sqrt((log_residual**2).mean())),
        "r2": float(1 - np.mean(log_residual**2) / variance) if variance > 0 else float("nan"),
        "spearman": _spearman(predicted, actual),
        "top_decile_lift": float(top.mean() / overall) if overall > 0 else float("nan"),
    }
    metrics.update(_per_day_ranking(frame, predicted))
    metrics.update(_tail_metrics(predicted, actual))
    return metrics


VIRAL_THRESHOLD = 100
"""What counts as reaching the front page. 4.2% of the test period clears it."""


def _tail_metrics(predicted: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    """How well the ranking finds the posts that go viral.

    Args:
        predicted: Predicted upvotes.
        actual: True upvotes.

    Returns:
        `viral_roc_auc`, `viral_pr_auc` and `viral_lift@1pct`.

    Every other metric on this page can be satisfied by predicting near the middle for
    everything, because three quarters of Hacker News posts *are* near the middle. These
    cannot: they score only the model's ability to pick out the 4.2% of posts that reach
    100 upvotes, which is the part of the problem anyone actually cares about and the
    part a mean-seeking loss is least pushed to solve.

    `pr_auc` against the base rate is the number to read -- ROC-AUC looks respectable at
    0.68 on a 4% positive class while the precision it implies is still poor.
    """
    labels = (actual >= VIRAL_THRESHOLD).astype(int)
    base = float(labels.mean())
    if labels.sum() < 30 or base in (0.0, 1.0):
        return {}
    cutoff = max(1, len(predicted) // 100)
    top = labels[np.argsort(-predicted)[:cutoff]]
    return {
        "viral_base_rate": base,
        "viral_roc_auc": float(roc_auc_score(labels, predicted)),
        "viral_pr_auc": float(average_precision_score(labels, predicted)),
        "viral_lift@1pct": float(top.mean() / base),
    }


def viral_calibration(
    predicted: np.ndarray, actual: np.ndarray, bins: int = 20
) -> dict[str, list[float]]:
    """Empirical P(score >= 100) as a function of the predicted count.

    Args:
        predicted: Predicted upvotes on a split the model did not train on.
        actual: True upvotes for the same rows.
        bins: Quantile buckets of the prediction.

    Returns:
        `{"edges": [...], "rates": [...]}` -- the mean prediction of each bucket and the
        fraction of it that actually cleared 100 upvotes.

    Fitted rather than derived from the Poisson the model was trained under. A Poisson
    with a mean of 15 puts about 1e-25 of its mass above 100; Hacker News scores are
    wildly overdispersed, so that analytic tail is off by more than twenty orders of
    magnitude and reports 0.0% for every post ever submitted. The empirical rate on
    held-out data is the only version of this number worth serving.
    """
    order = np.argsort(predicted)
    chunks = np.array_split(order, bins)
    edges = [float(predicted[c].mean()) for c in chunks if len(c)]
    rates = [float((actual[c] >= VIRAL_THRESHOLD).mean()) for c in chunks if len(c)]
    return {"edges": edges, "rates": rates}


def calibration_table(
    predicted: np.ndarray, actual: np.ndarray, bins: int = 10
) -> tuple[list[str], list[list[Any]]]:
    """Predicted against actual upvotes, by predicted decile.

    Args:
        predicted: Predicted upvotes.
        actual: True upvotes.
        bins: How many quantile buckets.

    Returns:
        `(columns, rows)` for `log_table`.

    The figure that would have caught the original bug on day one: a well-calibrated
    model has `mean predicted` tracking `mean actual` down every row, and a model that
    ranks well but predicts the wrong magnitudes has two columns that diverge while
    every ranking metric on the page looks healthy.
    """
    order = np.argsort(predicted)
    edges = np.linspace(0, len(order), bins + 1).astype(int)
    rows: list[list[Any]] = []
    for index in range(bins):
        chunk = order[edges[index] : edges[index + 1]]
        if len(chunk) == 0:
            continue
        rows.append(
            [
                index + 1,
                len(chunk),
                round(float(predicted[chunk].mean()), 2),
                round(float(actual[chunk].mean()), 2),
                round(float(np.median(actual[chunk])), 1),
            ]
        )
    return ["decile", "n", "mean_predicted", "mean_actual", "median_actual"], rows


def baseline_predictions(splits: Splits, split: str) -> dict[str, np.ndarray]:
    """Every baseline's predicted upvotes on one split.

    Args:
        splits: The loaded splits; the training one fits each baseline.
        split: Which split to predict.

    Returns:
        Baseline name to predicted upvotes.

    All five are fitted on the training split only, and every one predicts the **mean
    upvote count** of its group rather than a mean of logs. That distinction is the
    whole point of this rewrite: a baseline that predicted the geometric mean would be
    biased low in the same way the first model was, and would flatter it.
    """
    train = splits.frames["train"]
    target = splits.frames[split]
    upvotes = train["score"].to_numpy(dtype=np.float64)
    global_mean = float(upvotes.mean())
    count = len(target)

    hour_means = (
        pd.Series(upvotes, index=train["posted_at"].dt.hour.to_numpy()).groupby(level=0).mean()
    )
    domain_means = pd.Series(upvotes, index=train["domain"].to_numpy()).groupby(level=0).mean()

    predictions = {
        "global_mean": np.full(count, global_mean),
        "hour_mean": target["posted_at"]
        .dt.hour.map(hour_means)
        .fillna(global_mean)
        .to_numpy(dtype=np.float64),
        "domain_mean": target["domain"]
        .map(domain_means)
        .fillna(global_mean)
        .to_numpy(dtype=np.float64),
    }

    # Ridge on log1p and back through expm1 would inherit exactly the retransformation
    # bias this rewrite exists to remove, so the baseline is fitted on the raw count.
    # It is a worse fit in squared error and an honest comparison in upvotes.
    vectoriser = TfidfVectorizer(
        ngram_range=(1, 2), min_df=3, max_features=200_000, sublinear_tf=True
    )
    matrix = vectoriser.fit_transform(train["title"])
    ridge = Ridge(alpha=1.0, solver="sparse_cg")
    ridge.fit(matrix, upvotes)
    predictions["tfidf_ridge"] = np.maximum(
        ridge.predict(vectoriser.transform(target["title"])), 0.0
    )
    return predictions


def to_upvotes(output: np.ndarray, objective: Objective) -> np.ndarray:
    """Turn a model's raw output into expected upvotes.

    Args:
        output: What the head emitted.
        objective: How it was trained.

    Returns:
        Predicted upvotes, on the scale the task actually asks about.

    This is the step the first version of this service got wrong. Under `log_mse` the
    head fits the conditional mean *of the log*, and `expm1` of that is close to a
    geometric mean -- on a distribution with a median of 3 and a maximum of 5,710 it
    lands about five times low. Under `poisson` the head emits `log(lambda)` and the
    exponential is the expected count directly, with no biased step in between.
    """
    clipped = np.clip(output, -20, 20)
    upvotes: np.ndarray = np.exp(clipped) if objective == "poisson" else np.expm1(clipped)
    return upvotes


@torch.no_grad()
def predict(
    model: ScorePredictor,
    batch: PostBatch,
    device: torch.device,
    objective: Objective = "poisson",
) -> np.ndarray:
    """Run a model over a whole split and return predicted upvotes.

    Args:
        model: A built scorer.
        batch: The split's tensors.
        device: Where to run.
        objective: How the model was trained, which decides the output transform.

    Returns:
        Predicted upvotes, aligned to the split's row order -- `iterate_batches` is
        called without shuffling, which is what keeps that alignment true.
    """
    model.eval()
    outputs = [
        model(tokens, mask, numeric, domain).cpu()
        for tokens, mask, numeric, domain, _ in iterate_batches(
            batch, 4096, shuffle=False, device=device
        )
    ]
    return to_upvotes(torch.cat(outputs).numpy().astype(np.float64), objective)


def _slice_masks(
    frame: pd.DataFrame, splits: Splits, word_to_index: dict[str, int]
) -> dict[str, dict[str, np.ndarray]]:
    """Row masks for every slice the harness reports.

    Args:
        frame: The split being sliced.
        splits: Supplies the fitted domain vocabulary, so "rare domain" means what the
            model thinks it means rather than what this function guesses.
        word_to_index: The extended vocabulary, for the OOV slice.

    Returns:
        Slice family to bucket name to boolean mask.
    """
    known = splits.spec.domain_to_index
    titles = frame["title"].str.lower()
    scores = frame["score"].to_numpy()
    oov = np.array(
        [
            (
                sum(1 for token in tokens if token not in word_to_index) / len(tokens)
                if len(tokens)
                else 1.0
            )
            for tokens in frame["tokens"]
        ]
    )
    # `duplicates="drop"`: a quarter of these posts score exactly 1, so the quartile
    # edges are not distinct and qcut would otherwise raise. Fewer than four buckets is
    # the honest outcome on a distribution this piled up at its floor.
    quartile = pd.Categorical(pd.qcut(scores, 4, labels=False, duplicates="drop"))
    return {
        "score_quartile": {
            f"q{int(bucket) + 1}": np.asarray(quartile == bucket) for bucket in quartile.categories
        },
        "domain": {
            "github.com": (frame["domain"] == "github.com").to_numpy(),
            "arxiv.org": (frame["domain"] == "arxiv.org").to_numpy(),
            "text_post": (frame["domain"] == "<none>").to_numpy(),
            "rare": ~frame["domain"].isin(known).to_numpy(),
        },
        "prefix": {
            "show_hn": titles.str.startswith("show hn").to_numpy(),
            "ask_hn": titles.str.startswith("ask hn").to_numpy(),
            "plain": ~titles.str.startswith(("show hn", "ask hn", "tell hn")).to_numpy(),
        },
        "title_oov": {
            "none": (oov == 0),
            "some": (oov > 0) & (oov <= 0.25),
            "high": oov > 0.25,
        },
        "weekday": {
            "weekday": (frame["posted_at"].dt.dayofweek < 5).to_numpy(),
            "weekend": (frame["posted_at"].dt.dayofweek >= 5).to_numpy(),
        },
    }


def slice_table(
    frame: pd.DataFrame,
    predicted: np.ndarray,
    actual: np.ndarray,
    splits: Splits,
    word_to_index: dict[str, int],
) -> tuple[list[str], list[list[Any]]]:
    """Spearman and lift per slice.

    Args:
        frame: The split.
        predicted: Predictions aligned to it.
        actual: True `log1p(score)`.
        splits: Supplies the fitted domain vocabulary.
        word_to_index: The extended vocabulary, for the OOV slice.

    Returns:
        `(columns, rows)` for `log_table`.
    """
    rows: list[list[Any]] = []
    for family, buckets in _slice_masks(frame, splits, word_to_index).items():
        for name, mask in buckets.items():
            count = int(mask.sum())
            if count < 50:
                continue
            rows.append(
                [
                    family,
                    name,
                    count,
                    round(_spearman(predicted[mask], actual[mask]), 4),
                    round(float(actual[mask].mean()), 2),
                ]
            )
    return ["family", "bucket", "n", "spearman", "mean_score"], rows


def probe_table(
    model: ScorePredictor, splits: Splits, config: Config, word_to_index: dict[str, int]
) -> tuple[list[str], list[list[Any]]]:
    """The fixed probe titles and what the model makes of them.

    Args:
        model: A built scorer.
        splits: Supplies the fitted feature spec.
        config: Supplies widths and the device.
        word_to_index: The extended vocabulary.

    Returns:
        `(columns, rows)` sorted by predicted score, so a glance says whether the
        ordering is one a human would recognise.
    """
    from content_scoring.data import make_tensors

    latest = splits.frames["test"]["posted_at"].max()
    frame = pd.DataFrame(
        {
            "title": [title for title, _ in PROBE_TITLES],
            "tokens": [_tokenise(title) for title, _ in PROBE_TITLES],
            "url": [f"https://{domain}/x" if domain else "" for _, domain in PROBE_TITLES],
            "domain": [domain or "<none>" for _, domain in PROBE_TITLES],
            "score": [1] * len(PROBE_TITLES),
            "time": [int(latest.timestamp())] * len(PROBE_TITLES),
            "posted_at": [latest] * len(PROBE_TITLES),
            "author": ["probe"] * len(PROBE_TITLES),
            "comments": [0] * len(PROBE_TITLES),
        }
    )
    batch = make_tensors(frame, splits.spec, word_to_index, config)
    predicted = predict(model, batch, config.device, config.objective)
    order = np.argsort(-predicted)
    rows = [
        [frame["title"].iloc[i], frame["domain"].iloc[i], round(float(predicted[i]), 1)]
        for i in order
    ]
    return ["title", "domain", "predicted_score"], rows


def _tokenise(title: str) -> list[str]:
    """Tokenise exactly as the ETL does, so serving and training agree."""
    from content_scoring.etl import normalise_text, tokenise

    return tokenise(normalise_text(title))


def leave_one_group_out(
    model: ScorePredictor, splits: Splits, split: str
) -> tuple[list[str], list[list[Any]]]:
    """Ablate each feature group in turn on a late-fusion model.

    Args:
        model: A `LateFusionScorer`. Other architectures have no separable branches, so
            this returns an empty table for them.
        splits: The loaded splits.
        split: Which split to ablate on.

    Returns:
        `(columns, rows)` giving each group's Spearman when its branch is zeroed, and
        the drop from the full model.

    Zeroing a branch's input rather than retraining without it: the question is what the
    trained model *uses*, not what a differently-trained model would learn to use, and
    the two answers differ.
    """
    from content_scoring.model import LateFusionScorer

    if not isinstance(model, LateFusionScorer):
        return ["group", "spearman", "delta"], []

    device = next(model.parameters()).device
    actual = splits.frames[split]["score"].to_numpy(dtype=np.float64)
    objective = model.config.objective
    full = _spearman(predict(model, splits.tensors[split], device, objective), actual)

    rows: list[list[Any]] = [["(none)", round(full, 4), 0.0]]
    for name in model.branches:
        model.disabled = {name}
        ablated = _spearman(predict(model, splits.tensors[split], device, objective), actual)
        rows.append([name, round(ablated, 4), round(ablated - full, 4)])
    model.disabled = set()
    return ["group", "spearman", "delta"], rows


def score_extended_embeddings(config: Config) -> dict[str, float]:
    """Score the extended matrix on the intrinsic word-vector benchmarks.

    Args:
        config: Supplies the embedding path and the benchmark repo.

    Returns:
        WordSim-353 and SimLex-999 Spearman with their coverage, or an empty dict if
        the benchmark files cannot be fetched.

    This is the check that adding 40,000 Hacker News rows did not damage the text8
    vectors underneath them. A number in the table, rather than something someone
    remembers to look at.
    """
    from word2vec import parse_similarity_pairs, similarity_score

    payload = load_checkpoint(config.embeddings_path)
    embeddings = payload["model_state_dict"]["in_embed.weight"]
    word_to_index = payload["extra"]["word_to_index"]

    similarity: dict[str, list[tuple[str, str, float]]] = {}
    for name, filename in (("wordsim353", "wordsim353.txt"), ("simlex999", "simlex999.txt")):
        path, _ = resolve_checkpoint(
            config.data_cache_dir / "eval" / filename,
            "roshbeed/ai-residency-word-embeddings-eval",
            repo_type="dataset",
        )
        if path is None:
            logger.warning("Could not fetch %s; skipping the intrinsic check", filename)
            return {}
        similarity[name] = parse_similarity_pairs(path)

    metrics: dict[str, float] = {}
    for name, pairs in similarity.items():
        correlation, coverage = similarity_score(embeddings, word_to_index, pairs)
        metrics[f"embed_{name}"] = correlation
        metrics[f"embed_{name}_coverage"] = coverage
    return metrics


def load_predictor(config: Config, splits: Splits, word_to_index: dict[str, int]) -> ScorePredictor:
    """Build and load one architecture's checkpoint.

    Args:
        config: Supplies `fusion` and the checkpoint path.
        splits: Supplies the number of domains.
        word_to_index: The extended vocabulary, sized to the embedding matrix.

    Returns:
        The loaded model on the configured device.

    Raises:
        FileNotFoundError: If the checkpoint is neither local nor on the Hub.
    """
    payload = load_checkpoint(config.embeddings_path)
    matrix = payload["model_state_dict"]["in_embed.weight"]
    model = build_model(matrix, splits.spec.num_domains, config)

    path, _ = resolve_checkpoint(
        config.served_checkpoint,
        config.hf_repo_id,
        filename=config.served_checkpoint.name,
    )
    if path is None:
        raise FileNotFoundError(
            f"No checkpoint for fusion={config.fusion} at {config.served_checkpoint} "
            f"or in {config.hf_repo_id}."
        )
    load_predictor_weights(model, load_checkpoint(path)["model_state_dict"])
    del word_to_index
    return model.to(config.device).eval()


def aggregate(values: list[float]) -> tuple[float, float]:
    """Mean and full spread of a metric across seeds.

    Args:
        values: One value per seed.

    Returns:
        `(mean, max - min)`. The full range rather than a standard deviation: three
        points do not support a variance estimate, and the range is what a reader needs
        to decide whether a gap between two architectures is real.
    """
    finite = [v for v in values if v == v]
    if not finite:
        return float("nan"), float("nan")
    return float(np.mean(finite)), float(max(finite) - min(finite))


def main() -> None:
    """Score every trained architecture against the test split and print the table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split", default="test", choices=("val", "test"), help="Which split to score."
    )
    parser.add_argument(
        "--untrained",
        action="store_true",
        help="Score randomly-initialised models instead of loading checkpoints. Run "
        "this before any training: it is the only way to tell a real score from a bug.",
    )
    parser.add_argument("--json", type=str, default="", help="Also write the table here.")
    arguments = parser.parse_args()

    config = Config()
    device = get_device()
    payload = load_checkpoint(config.embeddings_path)
    word_to_index = payload["extra"]["word_to_index"]
    matrix = payload["model_state_dict"]["in_embed.weight"]

    from content_scoring.data import build_splits

    splits = build_splits(config, word_to_index)

    frame = splits.frames[arguments.split]
    actual = splits.frames[arguments.split]["score"].to_numpy(dtype=np.float64)

    report: dict[str, dict[str, float]] = {}
    spreads: dict[str, dict[str, tuple[float, float]]] = {}
    for name, predicted in baseline_predictions(splits, arguments.split).items():
        report[name] = evaluate_predictions(frame, predicted, actual)

    fusions: Sequence[Fusion] = ("text_only", "features_only", "early", "late")
    ablation: tuple[list[str], list[list[Any]]] = (["group", "spearman", "delta"], [])
    slices: tuple[list[str], list[list[Any]]] = ([], [])
    for fusion in fusions:
        variant = Config(fusion=fusion)
        try:
            model = (
                build_model(matrix, splits.spec.num_domains, variant).to(device).eval()
                if arguments.untrained
                else load_predictor(variant, splits, word_to_index)
            )
        except FileNotFoundError as error:
            logger.warning("%s: %s", fusion, error)
            continue
        predicted = predict(model, splits.tensors[arguments.split], device, variant.objective)
        report[fusion] = evaluate_predictions(frame, predicted, actual)
        # Every seed, not just the promoted one. A single run cannot say whether a gap
        # between two architectures is bigger than the noise between two seeds of one.
        per_seed: dict[str, list[float]] = {}
        for seed_path in variant.seed_checkpoints():
            seed_model = build_model(matrix, splits.spec.num_domains, variant)
            load_predictor_weights(seed_model, load_checkpoint(seed_path)["model_state_dict"])
            seed_scores = predict(
                seed_model.to(device).eval(),
                splits.tensors[arguments.split],
                device,
                variant.objective,
            )
            for name, value in evaluate_predictions(frame, seed_scores, actual).items():
                per_seed.setdefault(name, []).append(value)
        if per_seed:
            spreads[fusion] = {name: aggregate(vals) for name, vals in per_seed.items()}
            report[fusion] = {name: mean for name, (mean, _) in spreads[fusion].items()}
        if fusion == "late":
            slices = slice_table(frame, predicted, actual, splits, word_to_index)
            ablation = leave_one_group_out(model, splits, arguments.split)
            for columns, rows in (probe_table(model, splits, variant, word_to_index),):
                logger.info("probe titles: %s", " | ".join(columns))
                for row in rows:
                    logger.info("  %-58s %-24s %8.1f", row[0][:58], row[1][:24], row[2])

    # Upvote accuracy first, then ranking. The service is named for predicting scores and
    # for a while reported only how well it ordered them, which is how it shipped
    # predicting a fifth of the upvotes that actually happen without anything noticing.
    logger.info(
        "%s split, %d posts, %d actual upvotes", arguments.split, len(frame), int(actual.sum())
    )
    logger.info(
        "%-16s%9s%9s%9s%9s%10s%10s%-9s%8s",
        "model",
        "MAE",
        "medAE",
        "mean",
        "max",
        "total",
        "spearman",
        " (spread)",
        "PR-AUC",
    )
    for name, metrics in report.items():
        spread = spreads.get(name, {})
        seeds = f" ±{spread['spearman'][1]:.4f}" if "spearman" in spread else ""
        logger.info(
            "%-16s%9.2f%9.2f%9.2f%9.0f%9.0f%%%10.4f%-9s%8.3f",
            name,
            metrics["mae"],
            metrics["mdae"],
            metrics["mean_predicted"],
            metrics["max_predicted"],
            100 * metrics["total_ratio"],
            metrics["spearman"],
            seeds,
            metrics.get("viral_pr_auc", float("nan")),
        )
    logger.info(
        "%-16s%9s%9s%9.2f%9.0f%9.0f%%",
        "(actual)",
        "-",
        "-",
        float(actual.mean()),
        float(actual.max()),
        100.0,
    )

    if slices[1]:
        logger.info("slices (late fusion): %s", " | ".join(slices[0]))
        for row in slices[1]:
            logger.info("  %-16s%-14s%8d%10.4f%10.2f", *row)
    if ablation[1]:
        logger.info("leave-one-group-out (late fusion):")
        for row in ablation[1]:
            logger.info("  %-12s%10.4f%10.4f", *row)
    for name, value in score_extended_embeddings(config).items():
        logger.info("  %-24s%10.4f", name, value)

    if arguments.json:
        with open(arguments.json, "w", encoding="utf-8") as handle:
            json.dump(
                {"metrics": report, "slices": slices[1], "ablation": ablation[1]}, handle, indent=2
            )


if __name__ == "__main__":
    main()
