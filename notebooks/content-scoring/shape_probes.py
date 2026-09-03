"""Does each input group prefer a different tower shape?

    cd services/content-scoring
    uv run python ../../notebooks/content-scoring/shape_probes.py --budget 64

The fusion experiments so far gave every input the same tower shape, either by splitting
one layer proportionally or by splitting it evenly. Neither asks whether the inputs *want*
the same shape. A 300-d pooled title and three author scalars are different enough that
the arrangement of a fixed neuron budget -- wide and shallow against narrow and deep --
might suit them differently.

That matters for the fusion question. If every input prefers the same shape, late fusion's
one structural advantage over early fusion is worth nothing and the comparison is only
about connectivity. If they prefer different shapes, late fusion can give each input the
arrangement it wants and early fusion cannot, which is a reason to expect late fusion to
win at small budgets and lose once the model is large enough to find the structure itself.

Each probe trains ONE group against the target, with the same neuron budget arranged
several ways. Nothing is shared with the service's own models: this is a question about
the inputs, not about a trained artifact, so it lives here rather than in the package.

Two things could fake a result here, and both are controlled for:

**Starvation.** At a fixed budget the deep shapes are also the narrow ones, so `8x8`
gives the 300-d title a first layer of `300 -> 8`. A "the title prefers wide" finding
would then be a bottleneck rather than a preference. So every budget is run at several
sizes: at 1024 the same `8x128` shape has a first layer of `300 -> 128`, which is not a
bottleneck. A preference that survives the large budgets is a preference.

**Noise.** The seed spread on this task reaches 0.02-0.05 Spearman, which is larger than
the differences being looked for. Shapes are compared on medians over many seeds, and the
report prints the spread beside them so a difference smaller than the noise is visible as
such rather than being read as a result.

Selection is on validation; test is reported but never chosen on.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

# Run from the service directory, where `Config`'s relative paths resolve -- but the
# interpreter puts *this* directory on the path, not that one, so the service's top-level
# `config` package needs adding by hand.
sys.path.insert(0, str(Path.cwd()))

import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn

from ai_residency_common import get_logger, load_checkpoint, set_seed
from config import Config
from content_scoring.data import Splits, build_splits
from content_scoring.features import GROUP_SLICES

logger = get_logger(__name__)


def shapes_for(budget: int) -> dict[str, tuple[int, ...]]:
    """Ways to arrange one neuron budget, from widest-shallowest to narrowest-deepest.

    Args:
        budget: Total hidden units to spend.

    Returns:
        A label to layer widths, every entry summing to `budget`.
    """
    out: dict[str, tuple[int, ...]] = {}
    for depth in (1, 2, 3, 4, 8):
        if budget % depth or budget // depth < 2:
            continue
        out[f"{depth}x{budget // depth}"] = tuple([budget // depth] * depth)
    return out


def load_everything(config: Config) -> tuple[Splits, torch.Tensor]:
    """The splits and the extended embedding matrix, loaded the way the trainer does."""
    payload = load_checkpoint(config.embeddings_path)
    matrix = payload["model_state_dict"]["in_embed.weight"]
    return build_splits(config, payload["extra"]["word_to_index"]), matrix


def _inputs(group: str, splits: Splits, split: str, matrix: torch.Tensor) -> torch.Tensor:
    """One group's raw input columns for a split, the title pooled to its mean vector."""
    tokens, mask, numeric, domain, _ = splits.tensors[split]
    if group == "text":
        vectors = matrix[tokens] * mask.unsqueeze(-1)
        return vectors.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    if group == "domain":
        # A flag plus the *id*, which the caller embeds. A one-hot would hand the probe a
        # 1004-column input where the model sees nine, and the shape it preferred would
        # not be the shape the model wants.
        return torch.cat([numeric[:, GROUP_SLICES["domain"]], domain.unsqueeze(1).float()], dim=1)
    return numeric[:, GROUP_SLICES[group]]


def probe(
    group: str,
    shape: tuple[int, ...],
    config: Config,
    splits: Splits,
    matrix: torch.Tensor,
    seed: int,
    epochs: int,
) -> dict[str, float]:
    """Train one group through one tower shape and score it on the test split.

    Args:
        group: Which input group.
        shape: Hidden layer widths.
        config: Supplies the learning rate and batch size.
        splits: Loaded splits.
        matrix: The extended embedding matrix, for pooling the title.
        seed: Torch seed.
        epochs: Passes over the training split.

    Returns:
        Validation and test Spearman between predicted and actual score.
    """
    set_seed(seed)
    train_x = _inputs(group, splits, "train", matrix)
    train_y = torch.tensor(splits.frames["train"]["score"].to_numpy(dtype="float32"))

    layers: list[nn.Module] = []
    width = train_x.shape[1] + (config.domain_embed_dim - 1 if group == "domain" else 0)
    for out_width in shape:
        layers += [nn.Linear(width, out_width), nn.ReLU()]
        width = out_width
    stack = nn.Sequential(*layers, nn.Linear(width, 1))
    embedding = (
        nn.Embedding(splits.spec.num_domains, config.domain_embed_dim)
        if group == "domain"
        else None
    )

    def model(x: torch.Tensor) -> torch.Tensor:
        """The stack, with the domain id embedded first, as the service's model does."""
        if embedding is not None:
            x = torch.cat([x[:, :-1], embedding(x[:, -1].long())], dim=1)
        return stack(x)

    parameters = list(stack.parameters()) + (
        list(embedding.parameters()) if embedding is not None else []
    )
    optimiser = torch.optim.Adam(parameters, lr=config.learning_rate)
    loss_fn = nn.PoissonNLLLoss(log_input=True)

    order = torch.randperm(len(train_x))
    for _ in range(epochs):
        for start in range(0, len(order), config.batch_size):
            batch = order[start : start + config.batch_size]
            optimiser.zero_grad()
            loss = loss_fn(model(train_x[batch]).squeeze(-1), train_y[batch])
            loss.backward()
            optimiser.step()

    stack.eval()
    scores = {}
    with torch.no_grad():
        for split in ("val", "test"):
            predicted = model(_inputs(group, splits, split, matrix)).squeeze(-1).numpy()
            actual = splits.frames[split]["score"].to_numpy(dtype=np.float64)
            scores[split] = float(spearmanr(predicted, actual).statistic)
    return scores


def main() -> None:
    """Score every group against every shape at one budget."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=[64, 256, 1024],
        help="Total hidden units. Several, so a narrow-first-layer bottleneck can be "
        "told apart from a real preference for wide towers.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 52)))
    parser.add_argument(
        "--text-seeds",
        type=int,
        default=5,
        help="The title probe is ~30x slower than the others, so it gets fewer seeds.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--out", default="assets/shape-probes.json")
    arguments = parser.parse_args()

    config = Config()
    splits, matrix = load_everything(config)
    groups = ["text", "temporal", "domain", "title_surface", "author"]

    results: dict[str, dict[str, dict[str, list[float]]]] = {}
    for budget in arguments.budgets:
        shapes = shapes_for(budget)
        logger.info("budget %d, shapes %s", budget, list(shapes))
        for group in groups:
            seeds = arguments.seeds[: arguments.text_seeds] if group == "text" else arguments.seeds
            for label, shape in shapes.items():
                runs = [
                    probe(group, shape, config, splits, matrix, seed, arguments.epochs)
                    for seed in seeds
                ]
                results.setdefault(str(budget), {}).setdefault(group, {})[label] = [
                    run["val"] for run in runs
                ]
                results[str(budget)].setdefault(f"{group}__test", {})[label] = [
                    run["test"] for run in runs
                ]
                vals = sorted(run["val"] for run in runs)
                logger.info(
                    "budget %-5d %-14s %-8s val %.4f  spread %.4f  n=%d",
                    budget,
                    group,
                    label,
                    statistics.median(vals),
                    vals[-1] - vals[0],
                    len(vals),
                )

    Path(arguments.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Wrote %s", arguments.out)

    for budget in arguments.budgets:
        shapes = list(shapes_for(budget))
        print(f"\n=== budget {budget} — median validation rho, (spread) ===")
        print(f"{'group':<16}" + "".join(f"{label:>18}" for label in shapes) + "   best")
        for group in groups:
            row = results[str(budget)][group]
            medians = {label: statistics.median(row[label]) for label in shapes}
            best = max(medians, key=lambda label: medians[label])
            spread = {label: max(row[label]) - min(row[label]) for label in shapes}
            cells = "".join(
                f"{medians[label]:>11.4f} ({spread[label]:.3f})"[-18:] for label in shapes
            )
            # Only call a winner when it clears the noise it was measured against.
            margin = medians[best] - sorted(medians.values())[-2]
            verdict = best if margin > max(spread.values()) / 2 else "(not separated)"
            print(f"{group:<16}{cells}   {verdict}")


if __name__ == "__main__":
    main()
