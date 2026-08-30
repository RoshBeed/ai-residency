"""Post metadata, in four groups, all of it knowable at posting time.

The groups exist because the point of the comparison is *where* the signal is, not just
how much of it there is. Late fusion gives each its own tower; early fusion flattens
them into one vector and lets the network find the grouping itself; `features_only`
drops the title; `text_only` drops all of this.

| group | what it carries |
|---|---|
| `temporal` | hour and weekday as sin/cos pairs, weekend flag, and a drift term |
| `domain` | which site the post links to, as a learned embedding |
| `title_surface` | shape of the title: length, case, punctuation, `Show HN` and friends |
| `author` | how the author's earlier posts did |

**`comments` is not a feature and never becomes one.** The comment count correlates
strongly with the score and is not known when the post is submitted, so a model using
it would score beautifully offline and be useless for the question the service asks.
It stays in the data as a secondary label.

**The author features are the leak risk in this problem**, and the reason `fit` takes
the training frame while `transform` takes any frame. An author's mean score is the
single strongest signal on Hacker News, and computing it over the whole dataset would
let a test row's own score influence its own feature. `author_history` instead walks the
corpus forward in time and, for each row, uses only what was already published --
`expanding().mean().shift(1)` per author, over a frame the ETL sorted by `(time, id)`.
Val and test rows legitimately see train-era history, because that is information a real
submitter would have had.
"""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ai_residency_common import get_logger
from config import Config

__all__ = [
    "COLUMN_ORDER",
    "FEATURE_GROUPS",
    "GROUP_COLUMNS",
    "GROUP_DIMS",
    "GROUP_SLICES",
    "FeatureSpec",
    "author_history",
    "build_features",
    "fit_feature_spec",
]

logger = get_logger(__name__)

RARE_DOMAIN = "<rare>"
NO_DOMAIN = "<none>"
_PREFIXES = ("show hn", "ask hn", "tell hn", "launch hn")
_BRACKET_TAGS = ("[pdf]", "[video]", "[audio]", "[slides]")

GROUP_COLUMNS: dict[str, tuple[str, ...]] = {
    "temporal": (
        "hour_sin",
        "hour_cos",
        "weekday_sin",
        "weekday_cos",
        "is_weekend",
        "days_elapsed",
    ),
    "domain": ("has_url",),
    "title_surface": (
        "n_tokens",
        "n_chars",
        "mean_token_len",
        "has_digit",
        "is_question",
        "n_caps_words",
        "has_bracket_tag",
        *(f"prefix_{prefix.replace(' ', '_')}" for prefix in _PREFIXES),
    ),
    "author": ("author_mean", "author_count", "author_is_first"),
}
"""Which numeric columns belong to which branch.

Written out rather than counted. The widths the model slices with are derived from this
one place, so adding a feature cannot leave a branch reading its neighbour's columns --
which is the kind of bug that trains fine and simply learns less.

`domain` looks thin here because its real content is the domain *id*, carried separately
and embedded; `has_url` is the part of it that is already a number.
"""

FEATURE_GROUPS = tuple(GROUP_COLUMNS)
"""The four branches, in the order their columns are concatenated."""

COLUMN_ORDER: tuple[str, ...] = tuple(
    column for group in FEATURE_GROUPS for column in GROUP_COLUMNS[group]
)
"""Every numeric column, in the order `build_features` emits them."""

GROUP_SLICES: dict[str, slice] = {}
_offset = 0
for _group in FEATURE_GROUPS:
    _width = len(GROUP_COLUMNS[_group])
    GROUP_SLICES[_group] = slice(_offset, _offset + _width)
    _offset += _width

GROUP_DIMS: dict[str, int] = {group: len(GROUP_COLUMNS[group]) for group in FEATURE_GROUPS}
"""Numeric width of each group, derived from `GROUP_COLUMNS`."""


@dataclass
class FeatureSpec:
    """Everything fitted on the training split that `transform` needs.

    Attributes:
        domain_to_index: Domain to embedding row. Index 0 is `<rare>`.
        means: Per-column mean of the numeric features.
        stds: Per-column standard deviation, floored so a constant column cannot
            divide by zero.
        epoch_start: Unix time of the earliest training post, the origin of the drift
            term.
        epoch_span_days: Training span in days, which normalises that term.
        global_author_mean: Fallback for an author with no history.
    """

    domain_to_index: dict[str, int]
    means: np.ndarray
    stds: np.ndarray
    epoch_start: int
    epoch_span_days: float
    global_author_mean: float

    @property
    def num_domains(self) -> int:
        """Rows the domain embedding needs."""
        return len(self.domain_to_index)

    @property
    def numeric_dim(self) -> int:
        """Width of the numeric feature block, excluding the domain id."""
        return int(self.means.shape[0])


def author_history(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-row author statistics computed from strictly earlier posts.

    Args:
        frame: A frame sorted by `(time, id)`, carrying `author` and `score`.

    Returns:
        A frame with `author_mean`, `author_count` and `author_is_first`, aligned to
        `frame`'s index.

    The `shift(1)` inside the group is what makes this causal: without it a row's own
    score is inside its own mean, and a model reads the answer off the feature. With it,
    an author's first post has no history at all, which `author_is_first` marks so the
    model can tell "no history" from "history of zero".

    Anonymous or missing authors are pooled under the empty string rather than each
    getting their own group. They are a handful of rows and giving them a shared history
    is less wrong than giving each a singleton one.
    """
    scores = np.log1p(frame["score"].to_numpy(dtype=np.float64))
    working = pd.DataFrame(
        {"author": frame["author"].to_numpy(), "log_score": scores},
        index=frame.index,
    )
    grouped = working.groupby("author", sort=False)["log_score"]
    prior_mean = grouped.transform(lambda values: values.expanding().mean().shift(1))
    prior_count = grouped.cumcount()
    return pd.DataFrame(
        {
            "author_mean": prior_mean,
            "author_count": np.log1p(prior_count.to_numpy(dtype=np.float64)),
            "author_is_first": (prior_count == 0).astype(np.float64),
        },
        index=frame.index,
    )


def _numeric_block(frame: pd.DataFrame, spec: FeatureSpec | None) -> pd.DataFrame:
    """Assemble every numeric feature, unscaled, in a fixed column order."""
    posted = frame["posted_at"]
    hour = posted.dt.hour.to_numpy(dtype=np.float64)
    weekday = posted.dt.dayofweek.to_numpy(dtype=np.float64)
    titles = frame["title"].to_numpy()
    lowered = np.array([title.lower() for title in titles])
    tokens = frame["tokens"]

    epoch_start = spec.epoch_start if spec else int(frame["time"].min())
    span = (
        spec.epoch_span_days if spec else max(1.0, float(frame["time"].max() - epoch_start) / 86400)
    )

    columns: dict[str, np.ndarray] = {
        # Cyclic encodings, because hour 23 and hour 0 are adjacent and a raw integer
        # tells the model they are twenty-three apart.
        "hour_sin": np.sin(2 * math.pi * hour / 24),
        "hour_cos": np.cos(2 * math.pi * hour / 24),
        "weekday_sin": np.sin(2 * math.pi * weekday / 7),
        "weekday_cos": np.cos(2 * math.pi * weekday / 7),
        "is_weekend": (weekday >= 5).astype(np.float64),
        # Scores drift over three years. Without a time term the model has to express
        # that drift through whatever else correlates with it, usually the domain mix.
        "days_elapsed": (frame["time"].to_numpy(dtype=np.float64) - epoch_start) / 86400 / span,
        "has_url": (frame["domain"].to_numpy() != NO_DOMAIN).astype(np.float64),
        "n_tokens": np.array([len(token_list) for token_list in tokens], dtype=np.float64),
        "n_chars": np.array([len(title) for title in titles], dtype=np.float64),
        "mean_token_len": np.array(
            [
                float(np.mean([len(token) for token in token_list])) if len(token_list) else 0.0
                for token_list in tokens
            ]
        ),
        "has_digit": np.array(
            [float(any(character.isdigit() for character in title)) for title in titles]
        ),
        "is_question": np.array([float(title.endswith("?")) for title in titles]),
        "n_caps_words": np.array(
            [
                float(sum(1 for word in title.split() if word.isupper() and len(word) > 1))
                for title in titles
            ]
        ),
        "has_bracket_tag": np.array(
            [float(any(tag in title for tag in _BRACKET_TAGS)) for title in lowered]
        ),
    }
    for prefix in _PREFIXES:
        columns[f"prefix_{prefix.replace(' ', '_')}"] = np.array(
            [float(title.startswith(prefix)) for title in lowered]
        )

    history = author_history(frame)
    fallback = spec.global_author_mean if spec else float(np.nanmean(history["author_mean"]))
    columns["author_mean"] = history["author_mean"].fillna(fallback).to_numpy(dtype=np.float64)
    columns["author_count"] = history["author_count"].to_numpy(dtype=np.float64)
    columns["author_is_first"] = history["author_is_first"].to_numpy(dtype=np.float64)

    # Reindexed to COLUMN_ORDER rather than trusting insertion order. This is what makes
    # GROUP_SLICES true: a column added above but not declared in GROUP_COLUMNS raises
    # here instead of silently shifting every branch's view of the matrix.
    missing = set(COLUMN_ORDER) - set(columns)
    extra = set(columns) - set(COLUMN_ORDER)
    if missing or extra:
        raise ValueError(
            f"feature columns disagree with GROUP_COLUMNS: missing {sorted(missing)}, "
            f"unexpected {sorted(extra)}"
        )
    return pd.DataFrame(columns, index=frame.index)[list(COLUMN_ORDER)]


def fit_feature_spec(train: pd.DataFrame, config: Config) -> FeatureSpec:
    """Fit the domain vocabulary and the scaling statistics on the training split.

    Args:
        train: The training frame.
        config: Supplies `min_domain_count`.

    Returns:
        The fitted spec, which travels inside the checkpoint so serving scales a
        request exactly as training scaled the corpus.
    """
    counts = train["domain"].value_counts()
    kept = sorted(
        str(domain) for domain, count in counts.items() if count >= config.min_domain_count
    )
    domain_to_index = {RARE_DOMAIN: 0}
    for domain in kept:
        domain_to_index.setdefault(domain, len(domain_to_index))

    epoch_start = int(train["time"].min())
    span = max(1.0, float(train["time"].max() - epoch_start) / 86400)
    history = author_history(train)
    global_mean = float(np.nanmean(history["author_mean"].to_numpy(dtype=np.float64)))

    provisional = FeatureSpec(
        domain_to_index=domain_to_index,
        means=np.zeros(1),
        stds=np.ones(1),
        epoch_start=epoch_start,
        epoch_span_days=span,
        global_author_mean=global_mean,
    )
    block = _numeric_block(train, provisional).to_numpy(dtype=np.float64)
    means = block.mean(axis=0)
    # Floored rather than left as measured: a column that is constant across the
    # training split divides by zero and turns the whole feature matrix into NaN, which
    # surfaces as a loss of nan several minutes into a run rather than as an error here.
    stds = np.maximum(block.std(axis=0), 1e-6)

    logger.info(
        "Fitted %d domains (of %d seen, min_count=%d) and %d numeric features",
        len(domain_to_index),
        len(counts),
        config.min_domain_count,
        means.shape[0],
    )
    return FeatureSpec(
        domain_to_index=domain_to_index,
        means=means,
        stds=stds,
        epoch_start=epoch_start,
        epoch_span_days=span,
        global_author_mean=global_mean,
    )


def build_features(frame: pd.DataFrame, spec: FeatureSpec) -> tuple[np.ndarray, np.ndarray]:
    """Turn a frame into the model's numeric block and domain ids.

    Args:
        frame: Any split, or a single-row frame at serving time.
        spec: The spec fitted on the training split.

    Returns:
        `(numeric, domain_ids)` -- `(n, numeric_dim)` float32 and `(n,)` int64.

    Domains unseen during training map to `<rare>` rather than being dropped: a new site
    appearing in the test period is exactly the case the model has to handle, and it
    handles it by having learned what an unfamiliar domain is worth.
    """
    block = _numeric_block(frame, spec).to_numpy(dtype=np.float64)
    numeric = (block - spec.means) / spec.stds
    domain_ids = np.array(
        [spec.domain_to_index.get(domain, 0) for domain in frame["domain"]], dtype=np.int64
    )
    return numeric.astype(np.float32), domain_ids
