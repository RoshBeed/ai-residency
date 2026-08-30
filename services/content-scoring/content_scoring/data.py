"""Turning the derived splits into tensors the models read.

A post reaches a model as four things: padded token ids, a padding mask, the scaled
numeric features, and a domain id. Titles are short -- the median is nine tokens -- so
padding them to a fixed width costs almost nothing and lets the whole split live as four
dense tensors rather than as a collated `Dataset`.

The target is `log1p(score)`. Raw scores run from 1 to 5,710 with a median of 3, so a
squared error on the raw scale is dominated by a handful of front-page posts and the
model learns to predict the mean of the tail. In log space the residuals are roughly
symmetric and every post contributes.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from ai_residency_common import get_logger
from config import Config
from content_scoring.features import FeatureSpec, build_features, fit_feature_spec

__all__ = [
    "PostBatch",
    "Splits",
    "build_splits",
    "encode_titles",
    "iterate_batches",
    "load_split",
    "make_tensors",
]

logger = get_logger(__name__)

PostBatch = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
"""`(tokens, mask, numeric, domain, target)`."""


@dataclass
class Splits:
    """The three splits as tensors, plus the frames the eval harness slices on.

    Attributes:
        tensors: Split name to its `PostBatch`.
        frames: Split name to the frame it came from, kept so the eval harness can
            group by domain, hour or title shape without recomputing anything.
        spec: The feature spec fitted on the training split.
    """

    tensors: dict[str, PostBatch]
    frames: dict[str, pd.DataFrame]
    spec: FeatureSpec


def load_split(config: Config, split: str) -> pd.DataFrame:
    """Read one derived split from the local cache.

    Args:
        config: Supplies the cache path.
        split: `train`, `val` or `test`.

    Returns:
        The split's frame.

    Raises:
        FileNotFoundError: If the ETL has not run.
    """
    path = config.split_path(split)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run `make -C services/content-scoring etl` first -- the "
            "splits are a local cache derived from the Hub mirror, not a published artifact."
        )
    return pd.read_parquet(path)


def encode_titles(
    frame: pd.DataFrame, word_to_index: dict[str, int], max_tokens: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad a split's titles into row ids and a mask.

    Args:
        frame: A split carrying a `tokens` column.
        word_to_index: The extended vocabulary.
        max_tokens: Width to pad or truncate to.

    Returns:
        `(tokens, mask)` -- `(n, max_tokens)` int64 and float32.

    Out-of-vocabulary tokens are dropped rather than mapped to a shared row, matching
    what the extension stage does when it builds the vectors. The mask, not the id, is
    what tells the model a position is empty, so id 0 being a real word is harmless.
    """
    tokens = np.zeros((len(frame), max_tokens), dtype=np.int64)
    mask = np.zeros((len(frame), max_tokens), dtype=np.float32)
    for row, title_tokens in enumerate(frame["tokens"]):
        ids = [word_to_index[token] for token in title_tokens if token in word_to_index]
        ids = ids[:max_tokens]
        if ids:
            tokens[row, : len(ids)] = ids
            mask[row, : len(ids)] = 1.0
    empty = int((mask.sum(axis=1) == 0).sum())
    if empty:
        logger.info(
            "%d of %d titles (%.2f%%) have no in-vocabulary words and pool to zero",
            empty,
            len(frame),
            100 * empty / max(len(frame), 1),
        )
    return torch.from_numpy(tokens), torch.from_numpy(mask)


def make_tensors(
    frame: pd.DataFrame, spec: FeatureSpec, word_to_index: dict[str, int], config: Config
) -> PostBatch:
    """Turn one split into the five tensors a model and its loss need.

    Args:
        frame: The split.
        spec: The spec fitted on the training split.
        word_to_index: The extended vocabulary.
        config: Supplies `max_title_tokens`.

    Returns:
        `(tokens, mask, numeric, domain, target)`.
    """
    tokens, mask = encode_titles(frame, word_to_index, config.max_title_tokens)
    numeric, domain = build_features(frame, spec)
    target = np.log1p(frame["score"].to_numpy(dtype=np.float32))
    return (
        tokens,
        mask,
        torch.from_numpy(numeric),
        torch.from_numpy(domain),
        torch.from_numpy(target),
    )


def build_splits(config: Config, word_to_index: dict[str, int]) -> Splits:
    """Load all three splits and fit the feature spec on the training one.

    Args:
        config: Supplies paths and widths.
        word_to_index: The extended vocabulary.

    Returns:
        Tensors, frames and the fitted spec.
    """
    frames = {name: load_split(config, name) for name in ("train", "val", "test")}
    spec = fit_feature_spec(frames["train"], config)
    tensors = {
        name: make_tensors(frame, spec, word_to_index, config) for name, frame in frames.items()
    }
    logger.info(
        "Splits: train %d, val %d, test %d",
        len(frames["train"]),
        len(frames["val"]),
        len(frames["test"]),
    )
    return Splits(tensors=tensors, frames=frames, spec=spec)


def iterate_batches(
    batch: PostBatch, batch_size: int, *, shuffle: bool, device: torch.device
) -> list[PostBatch]:
    """Slice a split into batches on the target device.

    Args:
        batch: A whole split's tensors.
        batch_size: Rows per batch.
        shuffle: Permute before slicing. True for training, false everywhere else so a
            prediction array lines up with the frame it came from.
        device: Where to deliver.

    Returns:
        The batches, in order.
    """
    tokens, mask, numeric, domain, target = batch
    count = len(target)
    order = torch.randperm(count) if shuffle else torch.arange(count)
    return [
        (
            tokens[slice_].to(device),
            mask[slice_].to(device),
            numeric[slice_].to(device),
            domain[slice_].to(device),
            target[slice_].to(device),
        )
        for slice_ in (order[start : start + batch_size] for start in range(0, count, batch_size))
    ]
