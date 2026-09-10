"""On Hacker News, what does the title actually interact with?

    cd services/content-scoring
    uv run python ../../notebooks/content-scoring/hn_interactions.py

The early-versus-late fusion question is only interesting if the title and the metadata
*interact* -- if the effect of a good title depends on when you posted, where you linked,
or who you are. Measured over all metadata at once the interaction is real but small
(+0.006 Spearman). This splits it, and asks the question one group at a time:

    does the title's effect depend on WHEN you post?
    ... on WHERE you link?
    ... on the SHAPE of the title?
    ... on WHO you are?

**A null needs its alternatives ruled out**, and several could produce one here:

- *the target transform*. Additivity is not transform-invariant: a model additive in
  `log1p` space is multiplicative in raw counts. `--target` runs raw, log1p and rank.
- *under-training*. Interactions are harder to learn than main effects, so a short budget
  finds the additive part and stops. `--epochs` is swept rather than fixed.
- *the metric*. Spearman only sees ordering; an interaction could move magnitudes without
  reordering anything. Both rho and MSE are reported.
- *subgroups*. An interaction confined to `Show HN` posts or the top decile disappears in
  an average over 69,000 of them. `--slice` restricts the test set.

For each pairing it fits two models on the real upvote target. One is additive by
construction -- a tower per input reduced to a score, then the scores added, which is late
fusion in Snoek's sense and cannot represent any interaction. The other concatenates the
raw features and can. The difference between them is that pairing's interaction, in the
units the rest of the write-up uses.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from scipy.stats import mannwhitneyu, spearmanr
from torch import nn

sys.path.insert(0, str(Path.cwd()))

from ai_residency_common import get_logger, load_checkpoint, set_seed
from config import Config
from content_scoring.data import Splits, build_splits
from content_scoring.features import GROUP_SLICES

logger = get_logger(__name__)

Combine = Literal["sum", "product"]

# The title against each metadata group, then every metadata pair against each other.
# The first set asks whether a good title is worth more in some contexts; the second asks
# whether the metadata features interact among themselves -- a question the fusion
# comparison never touches, because early and late fusion both keep the metadata together.
TITLE_PAIRINGS = {
    "title x when you post": ("temporal",),
    "title x where it links": ("domain",),
    "title x title shape": ("title_surface",),
    "title x the author": ("author",),
    "title x all metadata": ("temporal", "domain", "title_surface", "author"),
}

METADATA_PAIRINGS = {
    "when x where": ("temporal", "domain"),
    "when x author": ("temporal", "author"),
    "when x title shape": ("temporal", "title_surface"),
    "where x author": ("domain", "author"),
    "where x title shape": ("domain", "title_surface"),
    "author x title shape": ("author", "title_surface"),
}


def pooled_title(
    splits: Splits, split: str, matrix: torch.Tensor, chunk: int = 4096
) -> torch.Tensor:
    """Mean of each title's word vectors, gathered in chunks.

    The one-line version allocates `rows x max_tokens x dims` -- 27 GB on this split --
    before reducing it, which is enough to take the machine down.
    """
    tokens, mask, _, _, _ = splits.tensors[split]
    out = torch.empty((len(tokens), matrix.shape[1]), dtype=torch.float32)
    for start in range(0, len(tokens), chunk):
        stop = min(start + chunk, len(tokens))
        window = mask[start:stop].unsqueeze(-1)
        vectors = matrix[tokens[start:stop]] * window
        out[start:stop] = vectors.sum(dim=1) / window.sum(dim=1).clamp(min=1.0)
    return out


def metadata(splits: Splits, split: str, groups: tuple[str, ...]) -> torch.Tensor:
    """The named feature groups' columns, concatenated."""
    _, _, numeric, _, _ = splits.tensors[split]
    return torch.cat([numeric[:, GROUP_SLICES[g]] for g in groups], dim=1)


class Additive(nn.Module):
    """A tower per input reduced to one score, the scores summed. Cannot interact.

    `capacity` widens both towers without letting them meet, which is what separates "the
    joint model found an interaction" from "the joint model simply had more parameters".
    The `title x all metadata` pairing scored +0.005 against a same-hidden-size additive
    model while every one of its constituent pairs was zero or negative -- exactly the
    pattern extra capacity would produce on its own.

    `combine="product"` multiplies the two scores instead of adding them. That is still one
    number per tower and still no crosstalk, but it can express exactly one interaction: the
    claim that a good title *multiplies* upvotes rather than adding to them. Whatever fraction
    of the joint model's advantage it recovers is the fraction that single multiplication
    explains -- and whatever it leaves on the table is structure richer than one product.
    """

    def __init__(
        self,
        a_dim: int,
        b_dim: int,
        hidden: int,
        capacity: int = 1,
        combine: Combine = "sum",
    ) -> None:
        super().__init__()
        wide = hidden * capacity
        self.combine = combine
        self.a = nn.Sequential(nn.Linear(a_dim, wide), nn.ReLU(), nn.Linear(wide, 1))
        self.b = nn.Sequential(nn.Linear(b_dim, wide), nn.ReLU(), nn.Linear(wide, 1))

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        left, right = self.a(a), self.b(b)
        if self.combine == "product":
            return (left * right).squeeze(-1)
        return (left + right).squeeze(-1)


class Joint(nn.Module):
    """One network over both inputs concatenated. Can represent any interaction."""

    def __init__(self, a_dim: int, b_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(a_dim + b_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([a, b], dim=1)).squeeze(-1)


def parameters_of(model: nn.Module) -> int:
    """Trainable parameter count, for reporting what each comparison actually held equal."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fit(
    model: nn.Module,
    data: dict[str, tuple[torch.Tensor, ...]],
    seed: int,
    epochs: int,
    batch: int,
    lr: float,
) -> float:
    """Train on the real target and return test Spearman at the best validation epoch.

    **Each model is scored at its own optimum, not at a fixed epoch.** Without that, a
    long budget compares two models that are both past their best, and the two
    architectures do not get there at the same time: at 25 epochs the additive model had
    fallen from 0.291 to 0.270 while the joint model held at 0.285, which reads as the
    joint model "finding an interaction" when it is really the additive one overfitting
    sooner. Early stopping is what separates *more training reveals structure* from
    *these two curves peak at different places*, and the service's own trainer has had it
    all along.
    """
    set_seed(seed)
    a, b, y = data["train"]
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_state = -float("inf"), None
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(a))
        for start in range(0, len(order), batch):
            index = order[start : start + batch]
            optimiser.zero_grad()
            nn.functional.mse_loss(model(a[index], b[index]), y[index]).backward()
            optimiser.step()
        model.eval()
        with torch.no_grad():
            scored = model(*data["val"][:2]).numpy()
        current = float(spearmanr(scored, data["val"][2].numpy()).statistic)
        if current > best:
            best = current
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        predicted = model(*data["test"][:2]).numpy()
    return float(spearmanr(predicted, data["test"][2].numpy()).statistic)


def transform_target(splits: Splits, split: str, kind: str) -> torch.Tensor:
    """The upvote count, on the scale the fit is done in.

    Args:
        splits: The loaded splits.
        split: Which split.
        kind: `raw`, `log1p` or `rank`.

    Returns:
        `(n,)` targets.

    **This choice can create or destroy an interaction on its own.** A model that is
    additive in `log1p` space is multiplicative in raw counts, so "the title and the
    metadata do not interact" is a statement about a coordinate system until it has been
    checked in more than one. `rank` is the scale Spearman actually scores, and is
    immune to the long tail that dominates a squared-error fit on raw counts.
    """
    raw = torch.tensor(splits.frames[split]["score"].to_numpy(dtype="float32"))
    if kind == "raw":
        return raw / raw.std().clamp(min=1e-6)
    if kind == "rank":
        order = raw.argsort().argsort().float()
        return order / len(order)
    return torch.log1p(raw)


def slice_mask(splits: Splits, split: str, kind: str) -> torch.Tensor:
    """Which rows a slice keeps. An interaction can live in a subgroup and vanish overall."""
    frame = splits.frames[split]
    if kind == "show_hn":
        return torch.tensor(
            frame["title"].str.startswith(("Show HN", "Ask HN")).to_numpy(dtype=bool)
        )
    if kind == "top_decile":
        scores = frame["score"].to_numpy()
        return torch.tensor(scores >= np.quantile(scores, 0.9))
    if kind == "has_url":
        return torch.tensor(frame["url"].fillna("").str.len().to_numpy() > 0)
    return torch.ones(len(frame), dtype=torch.bool)


def main() -> None:
    """Measure the title's interaction with each metadata group in turn."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 52)))
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--match-capacity",
        type=int,
        default=3,
        help="Tower width multiplier for the capacity-matched additive control.",
    )
    parser.add_argument("--target", choices=["log1p", "raw", "rank"], default="log1p")
    parser.add_argument(
        "--slice", choices=["all", "show_hn", "top_decile", "has_url"], default="all"
    )
    parser.add_argument("--out", default="assets/hn-interactions.json")
    arguments = parser.parse_args()

    config = Config()
    payload = load_checkpoint(config.embeddings_path)
    splits = build_splits(config, payload["extra"]["word_to_index"])
    matrix = payload["model_state_dict"]["in_embed.weight"]
    title = {s: pooled_title(splits, s, matrix) for s in ("train", "val", "test")}
    splits_used = ("train", "val", "test")
    target = {s: transform_target(splits, s, arguments.target) for s in splits_used}
    keep = {s: slice_mask(splits, s, arguments.slice) for s in splits_used}
    logger.info(
        "target=%s slice=%s (%d train, %d test rows)",
        arguments.target,
        arguments.slice,
        int(keep["train"].sum()),
        int(keep["test"].sum()),
    )

    results: dict[str, dict[str, list[float]]] = {}
    pairings: dict[str, tuple[tuple[str, ...], tuple[str, ...] | None]] = {
        label: (groups, None) for label, groups in TITLE_PAIRINGS.items()
    }
    pairings.update({label: ((a,), (b,)) for label, (a, b) in METADATA_PAIRINGS.items()})

    for label, (left, right) in pairings.items():
        if right is None:
            data = {
                s: (title[s][keep[s]], metadata(splits, s, left)[keep[s]], target[s][keep[s]])
                for s in splits_used
            }
        else:
            data = {
                s: (
                    metadata(splits, s, left)[keep[s]],
                    metadata(splits, s, right)[keep[s]],
                    target[s][keep[s]],
                )
                for s in splits_used
            }
        a_dim, b_dim = data["train"][0].shape[1], data["train"][1].shape[1]
        # Defaults bind the loop variables; without them these close over whatever
        # `a_dim` happens to be at call time, which is only safe by accident here.
        builders = {
            "additive": lambda a=a_dim, b=b_dim: Additive(a, b, arguments.hidden),
            # Same architecture, towers widened until it has at least as many parameters
            # as the joint model. If the joint model still wins, it is the interaction.
            "additive-matched": lambda a=a_dim, b=b_dim: Additive(
                a, b, arguments.hidden, capacity=arguments.match_capacity
            ),
            # One tower each, one number each, multiplied rather than summed. Tests the
            # stated mechanism directly rather than inferring it from the transforms.
            "multiplicative": lambda a=a_dim, b=b_dim: Additive(
                a,
                b,
                arguments.hidden,
                capacity=arguments.match_capacity,
                combine="product",
            ),
            "joint": lambda a=a_dim, b=b_dim: Joint(a, b, arguments.hidden),
        }
        for name, build in builders.items():
            scores = [
                fit(
                    build(),
                    data,
                    seed,
                    arguments.epochs,
                    arguments.batch,
                    arguments.learning_rate,
                )
                for seed in arguments.seeds
            ]
            results.setdefault(label, {})[name] = scores
        sizes = {n: parameters_of(b()) for n, b in builders.items()}
        results[label]["_parameters"] = [float(sizes[n]) for n in builders]
        add, joint = results[label]["additive"], results[label]["joint"]
        logger.info(
            "%-20s additive %.4f  joint %.4f  interaction %+.4f  p=%.3f",
            label,
            statistics.median(add),
            statistics.median(joint),
            statistics.median(joint) - statistics.median(add),
            mannwhitneyu(add, joint).pvalue,
        )

    Path(arguments.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Wrote %s", arguments.out)

    print("\n  does the first input's effect depend on the second?")
    print(f"{'':<24}{'additive':>10}{'joint':>9}{'interaction':>13}{'p':>8}")
    for label in pairings:
        add = results[label]["additive-matched"]
        joint = results[label]["joint"]
        gap = statistics.median(joint) - statistics.median(add)
        p = mannwhitneyu(add, joint).pvalue
        mark = "  yes" if p < 0.05 and gap > 0 else "  no"
        print(
            f"{label:<24}{statistics.median(add):>10.4f}{statistics.median(joint):>9.4f}"
            f"{gap:>+13.4f}{p:>8.3f}{mark}"
        )


if __name__ == "__main__":
    main()
