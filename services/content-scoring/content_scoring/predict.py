"""Scoring a single post, the same way training scored a corpus.

One post is a one-row frame here rather than a special path. That is deliberate: the
feature code has fitted state -- domain vocabulary, scaling statistics, the author
fallback -- and the only way to be sure a request is transformed exactly as the training
corpus was is to send it through the same function with the same spec.

The spec travels inside the predictor checkpoint, so a served model cannot be paired
with statistics from a different run.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from ai_residency_common import get_logger, load_checkpoint, resolve_required
from config import Config, Fusion
from content_scoring.data import make_tensors
from content_scoring.embeddings import ExtendedEmbeddings, load_extended_embeddings
from content_scoring.etl import normalise_text, registered_domain, tokenise
from content_scoring.features import FeatureSpec
from content_scoring.model import ScorePredictor, build_model, load_predictor_weights

__all__ = [
    "LoadedPredictor",
    "Post",
    "load_embeddings",
    "load_predictor",
    "post_frame",
    "score_post",
]

logger = get_logger(__name__)


@dataclass
class Post:
    """One post as a caller supplies it.

    Attributes:
        title: The headline.
        url: Where it links, or empty for a text post.
        posted_at: When it was submitted. Defaults to now at the API layer.
        author: The submitter, used only to look up their history.
    """

    title: str
    url: str = ""
    posted_at: pd.Timestamp | None = None
    author: str = ""


@dataclass
class LoadedPredictor:
    """A model and everything needed to feed it.

    Attributes:
        model: The scorer, in eval mode.
        spec: The feature spec fitted during that model's training run.
        embeddings: The extended vectors and vocabulary.
        fusion: Which architecture this is.
        source: Where the checkpoint came from -- `local` or `hub`.
        viral_calibration: Held-out mapping from a predicted count to the empirical
            chance of clearing 100 upvotes. Empty for checkpoints written before it.
    """

    model: ScorePredictor
    spec: FeatureSpec
    embeddings: ExtendedEmbeddings
    fusion: Fusion
    source: str
    viral_calibration: dict[str, list[float]] = field(default_factory=dict)


def post_frame(post: Post) -> pd.DataFrame:
    """Turn one post into the frame shape the feature code expects.

    Args:
        post: The post.

    Returns:
        A one-row frame with every column `build_features` reads.

    The author's history cannot be looked up at serving time without a store of past
    posts, so a request gets the corpus mean and `author_is_first` set. That is honest:
    the model is told "an author we know nothing about", which is what is true.
    """
    timestamp = post.posted_at if post.posted_at is not None else pd.Timestamp.now(tz="UTC")
    title = normalise_text(post.title)
    return pd.DataFrame(
        {
            "id": [0],
            "title": [title],
            "tokens": [tokenise(title)],
            "url": [post.url],
            "domain": [registered_domain(post.url)],
            "score": [1],
            "time": [int(timestamp.timestamp())],
            "posted_at": [timestamp],
            "author": [post.author or ""],
            "comments": [0],
        }
    )


def load_embeddings(config: Config) -> ExtendedEmbeddings:
    """Resolve and load the extended word vectors.

    Args:
        config: Supplies the path and the repo.

    Returns:
        The matrix and its vocabulary.

    Raises:
        FileNotFoundError: If they are neither local nor on the Hub.

    Separate from `load_predictor` so a caller holding several architectures loads the
    matrix once. It is 104MB and identical for all four; the API serves all four.
    """
    path, source = resolve_required(
        config.embeddings_path,
        config.hf_repo_id,
        rebuild_hint="Build it with `make -C services/content-scoring extend-vocab`.",
    )
    logger.info("Extended embeddings from %s", source)
    return load_extended_embeddings(path)


def load_predictor(config: Config, embeddings: ExtendedEmbeddings | None = None) -> LoadedPredictor:
    """Resolve and load one architecture, with its embeddings and feature spec.

    Args:
        config: Supplies `fusion` and the repos to pull from.
        embeddings: Already-loaded vectors to share. Loaded here when omitted.

    Returns:
        The loaded predictor.

    Raises:
        FileNotFoundError: If the embeddings or the checkpoint are missing everywhere.
    """
    embeddings = embeddings if embeddings is not None else load_embeddings(config)

    checkpoint_path, source = resolve_required(
        config.predictor_checkpoint,
        config.hf_repo_id,
        rebuild_hint="Train it with `make -C services/content-scoring train`.",
    )
    payload = load_checkpoint(checkpoint_path)
    if payload["extra"].get("embeddings_finetuned"):
        raise RuntimeError(
            f"{checkpoint_path} was trained with fine-tuned word vectors, which it does "
            "not carry. Serving it against the base vectors would answer 200 with numbers "
            "unrelated to what was evaluated. Retrain with freeze_embeddings=true."
        )
    stored = payload["extra"]["feature_spec"]
    # Rebuilt field by field rather than by `FeatureSpec(**stored)`: `asdict` turned the
    # two arrays into nested lists on the way in, and a spec whose `means` is a list
    # broadcasts differently from one whose `means` is an array -- quietly, and only on
    # some shapes.
    spec = FeatureSpec(
        domain_to_index=dict(stored["domain_to_index"]),
        means=np.asarray(stored["means"], dtype=np.float64),
        stds=np.asarray(stored["stds"], dtype=np.float64),
        epoch_start=int(stored["epoch_start"]),
        epoch_span_days=float(stored["epoch_span_days"]),
        global_author_mean=float(stored["global_author_mean"]),
    )
    model = build_model(embeddings.matrix, spec.num_domains, config)
    load_predictor_weights(model, payload["model_state_dict"])
    logger.info("Loaded %s predictor from %s", config.fusion, source)
    return LoadedPredictor(
        model=model.to(config.device).eval(),
        spec=spec,
        embeddings=embeddings,
        fusion=config.fusion,
        source=source,
        viral_calibration=payload["extra"].get("viral_calibration", {}),
    )


@torch.no_grad()
def score_post(loaded: LoadedPredictor, post: Post, config: Config) -> tuple[float, list[bool]]:
    """Predict a post's score.

    Args:
        loaded: A loaded predictor.
        post: The post.
        config: Supplies `max_title_tokens` and the device.

    Returns:
        `(predicted_score, known)` where the score is on the raw upvote scale and
        `known` marks which of the title's tokens the vocabulary contains.

    The prediction is made in `log1p` space, as trained, and only then converted back.
    Training on the raw scale would let a handful of 5,000-point posts dominate the loss.
    """
    frame = post_frame(post)
    tokens, mask, numeric, domain, _ = make_tensors(
        frame, loaded.spec, loaded.embeddings.word_to_index, config
    )
    device = config.device
    predicted = loaded.model(
        tokens.to(device), mask.to(device), numeric.to(device), domain.to(device)
    )
    log_score = float(predicted.reshape(-1)[0].cpu())
    known = [token in loaded.embeddings.word_to_index for token in frame["tokens"].iloc[0]]
    return float(np.expm1(log_score)), known
