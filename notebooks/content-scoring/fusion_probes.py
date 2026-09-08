"""Reproduce the fusion literature's claims on Hacker News features.

    cd services/content-scoring
    uv run python ../../notebooks/content-scoring/fusion_probes.py interaction
    uv run python ../../notebooks/content-scoring/fusion_probes.py robustness

On the real target the two modalities turn out to be additive: a combiner over two
unimodal scores reaches 0.2348 where the joint model reaches 0.2392, a gap of 0.004
against a seed spread of 0.013. So early and late fusion score the same, which is the
correct answer to the question and no answer at all to whether the *setup* could tell
them apart. These probes supply the missing half.

**interaction** builds a synthetic target over the real feature distributions:

    y = g(title) + h(metadata) + alpha * g(title) * h(metadata) + noise

`g` and `h` are fixed random projections. At `alpha = 0` the task is additive; as alpha
grows the signal moves into the product, which is recoverable only by a model that sees
both modalities together. Snoek et al. name "the potential loss of correlation in mixed
feature space" as late fusion's central disadvantage but never vary it -- this turns it
into a dial.

Three architectures, because the word "late" covers two very different things:

    early        concatenate the raw features, then one network
    late-deep    a tower per modality, concatenated, then a *shared* network
    late-snoek   a tower per modality reduced to one score each, combined linearly

Only `late-snoek` matches Snoek's definition, and only it is structurally unable to
represent the product term. `late-deep` can, one layer later. The distinction matters:
every late-fusion model in this service is the `late-deep` kind, so the null result on
real data was never a test of the thing the papers warn about.

**robustness** takes the other half of the literature -- late fusion's advantage under
missing and noisy modalities. The surveys credit late fusion with degrading gracefully
when a modality goes missing or turns noisy, because it can lean on the surviving branch
where early fusion has entangled the two in its first layer. This trains each
architecture once on clean data, then at *inference* zeroes or corrupts one modality and
measures what is left. No retraining, so every degradation point is one forward pass.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch
from scipy.stats import mannwhitneyu, spearmanr
from torch import nn

sys.path.insert(0, str(Path.cwd()))

from ai_residency_common import get_logger, load_checkpoint, set_seed
from config import Config
from content_scoring.data import Splits, build_splits

logger = get_logger(__name__)

TASK_SEED = 20260905
"""Fixes `g` and `h`, so every run faces the same task and only the model seed varies."""


def modalities(
    splits: Splits, split: str, matrix: torch.Tensor, chunk: int = 4096
) -> tuple[torch.Tensor, torch.Tensor]:
    """The two modalities: the pooled title, and every numeric metadata column.

    Args:
        splits: The loaded splits.
        split: Which split to pool.
        matrix: `(vocab, dims)` word vectors.
        chunk: Rows gathered at a time.

    Returns:
        `(pooled_title, numeric)`.

    **Pooled in chunks, because the obvious one-liner asks for 27 GB.**
    `matrix[tokens] * mask` gathers `(rows, max_tokens, dims)` before reducing it, which
    on the training split is 710,722 x 32 x 300 floats -- 27.3 GB to produce an 0.85 GB
    result, on a 32 GB machine. It does not raise: macOS compresses and swaps until the
    compressor runs out of segments, the VM subsystem stalls, and the watchdog panics the
    kernel because `watchdogd` has not been scheduled for 91 seconds. That is a machine
    crash caused by one line of tensor indexing.

    Chunked, the same computation peaks at 157 MB.
    """
    tokens, mask, numeric, _, _ = splits.tensors[split]
    pooled = torch.empty((len(tokens), matrix.shape[1]), dtype=torch.float32)
    for start in range(0, len(tokens), chunk):
        stop = min(start + chunk, len(tokens))
        window = mask[start:stop].unsqueeze(-1)
        vectors = matrix[tokens[start:stop]] * window
        pooled[start:stop] = vectors.sum(dim=1) / window.sum(dim=1).clamp(min=1.0)
    return pooled, numeric


def synthetic_target(
    title: torch.Tensor,
    meta: torch.Tensor,
    alpha: float,
    projections: tuple[torch.Tensor, torch.Tensor],
    kind: str = "product",
) -> torch.Tensor:
    """`g + h + alpha * interaction(g, h) + noise`, on a scale comparable across kinds.

    Args:
        title: `(n, 300)` pooled title vectors.
        meta: `(n, k)` metadata columns.
        alpha: Interaction strength. Zero makes the task purely additive.
        projections: The fixed `(w_g, w_h)` random projections.
        kind: Which interaction to use. See `INTERACTIONS`.

    Returns:
        `(n,)` targets.

    `tanh` keeps both terms in [-1, 1] so the product is on the same scale as the main
    effects and `alpha` means the same thing at every setting. The noise is what stops a
    large enough model from simply memorising the projections.
    """
    w_g, w_h = projections
    g = torch.tanh(title @ w_g)
    h = torch.tanh(meta @ w_h)
    term = INTERACTIONS[kind](g, h)
    # Normalised so `alpha` means the same thing whichever interaction is used --
    # `max` and `product` have very different natural scales.
    term = (term - term.mean()) / term.std().clamp(min=1e-6)
    signal = g + h + alpha * term
    return signal + 0.5 * torch.randn(len(signal), generator=torch.Generator().manual_seed(0))


class Early(nn.Module):
    """Concatenate the raw modalities, then one network."""

    def __init__(self, title_dim: int, meta_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(title_dim + meta_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, title: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([title, meta], dim=1)).squeeze(-1)


class LateDeep(nn.Module):
    """A tower per modality, concatenated, then a shared network.

    This is what every "late fusion" model in the service actually is. The modalities do
    meet, one layer later than in early fusion, so the shared trunk can still represent an
    interaction between them.
    """

    def __init__(self, title_dim: int, meta_dim: int, hidden: int) -> None:
        super().__init__()
        half = hidden // 2
        self.title = nn.Sequential(nn.Linear(title_dim, half), nn.ReLU())
        self.meta = nn.Sequential(nn.Linear(meta_dim, half), nn.ReLU())
        self.trunk = nn.Sequential(nn.Linear(2 * half, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, title: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        return self.trunk(torch.cat([self.title(title), self.meta(meta)], dim=1)).squeeze(-1)


class LateSnoek(nn.Module):
    """Each modality reduced to one score, the scores combined linearly.

    Snoek's definition: "first reduces unimodal features to separately learned concept
    scores, then these scores are integrated". A linear combination of two scalars cannot
    represent their product, so this architecture is *structurally* unable to model an
    interaction between the modalities -- which is the disadvantage the paper names and
    the thing this probe is built to show.
    """

    def __init__(self, title_dim: int, meta_dim: int, hidden: int) -> None:
        super().__init__()
        half = hidden // 2
        self.title = nn.Sequential(
            nn.Linear(title_dim, half),
            nn.ReLU(),
            nn.Linear(half, half),
            nn.ReLU(),
            nn.Linear(half, 1),
        )
        self.meta = nn.Sequential(
            nn.Linear(meta_dim, half),
            nn.ReLU(),
            nn.Linear(half, half),
            nn.ReLU(),
            nn.Linear(half, 1),
        )
        self.combine = nn.Linear(2, 1)

    def forward(self, title: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        scores = torch.cat([self.title(title), self.meta(meta)], dim=1)
        return self.combine(scores).squeeze(-1)


class LateDot(nn.Module):
    """A tower per modality, combined by a dot product. Two-tower retrieval.

    This is what production retrieval actually does -- encode the query, encode the
    document, take a dot product -- and it is *not* the same as adding two scores. A dot
    product is a bilinear form, so it can represent any interaction expressible as
    `sum_k u_k(a) * v_k(b)`: strictly more than a generalised additive model, strictly
    less than an arbitrary function of both.

    That middle position is the reason two-tower retrieval works as well as it does, and
    the reason it still needs a cross-encoder to rerank. Which interactions it can and
    cannot reach is measurable rather than a matter of opinion, which is what this is for.
    """

    def __init__(self, title_dim: int, meta_dim: int, hidden: int, rank: int = 16) -> None:
        super().__init__()
        half = hidden // 2
        self.title = nn.Sequential(nn.Linear(title_dim, half), nn.ReLU(), nn.Linear(half, rank))
        self.meta = nn.Sequential(nn.Linear(meta_dim, half), nn.ReLU(), nn.Linear(half, rank))
        # A bias per branch, so the model keeps the additive part a sum would have given
        # it and the dot product is free to carry only the interaction.
        self.title_bias = nn.Sequential(nn.Linear(title_dim, half), nn.ReLU(), nn.Linear(half, 1))
        self.meta_bias = nn.Sequential(nn.Linear(meta_dim, half), nn.ReLU(), nn.Linear(half, 1))

    def forward(self, title: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        dot = (self.title(title) * self.meta(meta)).sum(dim=1)
        return dot + self.title_bias(title).squeeze(-1) + self.meta_bias(meta).squeeze(-1)


ARCHITECTURES = {
    "early": Early,
    "late-deep": LateDeep,
    "late-dot": LateDot,
    "late-snoek": LateSnoek,
}

# How the two modalities are made to interact. `product` is bilinear and a dot product
# should reach it; the other two are not, and should separate two-tower fusion from the
# architectures that see the raw features together.
INTERACTIONS = {
    "product": lambda g, h: g * h,
    "max": lambda g, h: torch.maximum(g, h),
    "disagree": lambda g, h: -(g - h).abs(),
}


def fit(
    architecture: str,
    data: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    hidden: int,
    seed: int,
    epochs: int,
    batch: int,
    learning_rate: float,
) -> float:
    """Train one architecture on one task and return its test Spearman."""
    set_seed(seed)
    title, meta, target = data["train"]
    model = ARCHITECTURES[architecture](title.shape[1], meta.shape[1], hidden)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for _ in range(epochs):
        order = torch.randperm(len(title))
        for start in range(0, len(order), batch):
            index = order[start : start + batch]
            optimiser.zero_grad()
            loss = nn.functional.mse_loss(model(title[index], meta[index]), target[index])
            loss.backward()
            optimiser.step()

    model.eval()
    with torch.no_grad():
        predicted = model(*data["test"][:2]).numpy()
    return float(spearmanr(predicted, data["test"][2].numpy()).statistic)


def run_threshold(arguments: argparse.Namespace) -> None:
    """How much interaction would the real data need before we could see it?

    Args:
        arguments: Parsed command line.

    "The task is additive" is a claim about a null, and a null is only as strong as the
    smallest effect the setup could have found. This bounds it: the *real* upvote target
    gets a synthetic interaction added on top, scaled to a fraction of the target's own
    standard deviation, and the sweep finds the smallest addition at which early fusion
    separates from an architecture that cannot represent it.

    That threshold is the answer to "how do you know there is no interaction" -- there is
    none stronger than the smallest one this would have caught.
    """
    config = Config()
    payload = load_checkpoint(config.embeddings_path)
    splits = build_splits(config, payload["extra"]["word_to_index"])
    matrix = payload["model_state_dict"]["in_embed.weight"]
    raw = {split: modalities(splits, split, matrix) for split in ("train", "test")}

    generator = torch.Generator().manual_seed(TASK_SEED)
    projections = (
        torch.randn(raw["train"][0].shape[1], generator=generator)
        / raw["train"][0].shape[1] ** 0.5,
        torch.randn(raw["train"][1].shape[1], generator=generator)
        / raw["train"][1].shape[1] ** 0.5,
    )

    results: dict[str, dict[str, list[float]]] = {}
    for strength in arguments.strengths:
        data = {}
        for split in ("train", "test"):
            title, meta = raw[split]
            # log1p, because the raw counts are so skewed that an interaction scaled to
            # their standard deviation would be dominated by a handful of viral posts.
            real = torch.log1p(
                torch.tensor(splits.frames[split]["score"].to_numpy(dtype="float32"))
            )
            product = torch.tanh(title @ projections[0]) * torch.tanh(meta @ projections[1])
            # Normalised by the product's *own* spread, not just the target's. Two tanh
            # terms multiplied have a standard deviation around 0.043, so scaling by the
            # target's std alone made `strength=1.0` an interaction worth 4% of the
            # target's spread rather than 100% -- the axis was mislabelled by about 23x
            # and the sweep could not mean what it said.
            product = product / product.std().clamp(min=1e-6)
            data[split] = (title, meta, real + strength * real.std() * product)
        for architecture in ARCHITECTURES:
            scores = [
                fit(
                    architecture,
                    data,
                    arguments.hidden[0],
                    seed,
                    arguments.epochs,
                    arguments.batch,
                    arguments.learning_rate,
                )
                for seed in arguments.seeds
            ]
            results.setdefault(f"{strength}", {})[architecture] = scores
            logger.info(
                "strength %-5s %-11s median rho %.4f  spread %.4f",
                strength,
                architecture,
                statistics.median(scores),
                max(scores) - min(scores),
            )

    Path(arguments.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Wrote %s", arguments.out)

    print(
        f"\n{'added':>7}" + "".join(f"{a:>13}" for a in ARCHITECTURES) + "   early-snoek        p"
    )
    for strength in arguments.strengths:
        row = results[f"{strength}"]
        medians = {a: statistics.median(row[a]) for a in ARCHITECTURES}
        gap = medians["early"] - medians["late-snoek"]
        p = mannwhitneyu(row["early"], row["late-snoek"]).pvalue
        mark = "  <- detected" if p < 0.05 and gap > 0 else ""
        print(
            f"{strength:>7}"
            + "".join(f"{medians[a]:>13.4f}" for a in ARCHITECTURES)
            + f"{gap:>+14.4f}{p:>9.3f}{mark}"
        )


def run_interaction(arguments: argparse.Namespace) -> None:
    """Sweep the interaction strength and watch the architectures separate."""
    config = Config()
    payload = load_checkpoint(config.embeddings_path)
    splits = build_splits(config, payload["extra"]["word_to_index"])
    matrix = payload["model_state_dict"]["in_embed.weight"]

    raw = {split: modalities(splits, split, matrix) for split in ("train", "test")}
    generator = torch.Generator().manual_seed(TASK_SEED)
    projections = (
        torch.randn(raw["train"][0].shape[1], generator=generator)
        / raw["train"][0].shape[1] ** 0.5,
        torch.randn(raw["train"][1].shape[1], generator=generator)
        / raw["train"][1].shape[1] ** 0.5,
    )

    results: dict[str, dict[str, list[float]]] = {}
    for alpha in arguments.alphas:
        data = {
            split: (*raw[split], synthetic_target(*raw[split], alpha, projections, arguments.kind))
            for split in ("train", "test")
        }
        for hidden in arguments.hidden:
            for architecture in ARCHITECTURES:
                scores = [
                    fit(
                        architecture,
                        data,
                        hidden,
                        seed,
                        arguments.epochs,
                        arguments.batch,
                        arguments.learning_rate,
                    )
                    for seed in arguments.seeds
                ]
                results.setdefault(f"{alpha}", {})[f"{architecture}@{hidden}"] = scores
                logger.info(
                    "alpha %-6s h=%-4d %-11s median rho %.4f  spread %.4f  n=%d",
                    alpha,
                    hidden,
                    architecture,
                    statistics.median(scores),
                    max(scores) - min(scores),
                    len(scores),
                )

    Path(arguments.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Wrote %s", arguments.out)

    for hidden in arguments.hidden:
        print(f"\n=== hidden {hidden} ===")
        print(f"{'alpha':>7}" + "".join(f"{a:>13}" for a in ARCHITECTURES) + "   early-snoek")
        for alpha in arguments.alphas:
            row = results[f"{alpha}"]
            medians = {a: statistics.median(row[f"{a}@{hidden}"]) for a in ARCHITECTURES}
            gap = medians["early"] - medians["late-snoek"]
            print(
                f"{alpha:>7}"
                + "".join(f"{medians[a]:>13.4f}" for a in ARCHITECTURES)
                + f"{gap:>+14.4f}"
            )


def run_robustness(arguments: argparse.Namespace) -> None:
    """Train clean, then damage one modality at inference and measure what survives.

    Args:
        arguments: Parsed command line.

    The literature's claim for late fusion is robustness, not accuracy: a model that
    reduces each modality separately can ignore a branch that has gone bad, where early
    fusion mixed them in its first layer and has no separate handle on either. Damage is
    applied only at test time so every architecture is compared on the same trained
    weights -- the question is what the *architecture* does with a broken input, not what
    it would learn if trained on broken inputs.
    """
    config = Config()
    payload = load_checkpoint(config.embeddings_path)
    splits = build_splits(config, payload["extra"]["word_to_index"])
    matrix = payload["model_state_dict"]["in_embed.weight"]

    raw = {split: modalities(splits, split, matrix) for split in ("train", "test")}
    generator = torch.Generator().manual_seed(TASK_SEED)
    projections = (
        torch.randn(raw["train"][0].shape[1], generator=generator)
        / raw["train"][0].shape[1] ** 0.5,
        torch.randn(raw["train"][1].shape[1], generator=generator)
        / raw["train"][1].shape[1] ** 0.5,
    )
    data = {
        split: (*raw[split], synthetic_target(*raw[split], arguments.alpha, projections))
        for split in ("train", "test")
    }

    results: dict[str, dict[str, list[float]]] = {}
    for architecture in ARCHITECTURES:
        for seed in arguments.seeds:
            set_seed(seed)
            title, meta, target = data["train"]
            model = ARCHITECTURES[architecture](title.shape[1], meta.shape[1], arguments.hidden[0])
            optimiser = torch.optim.Adam(model.parameters(), lr=arguments.learning_rate)
            for _ in range(arguments.epochs):
                order = torch.randperm(len(title))
                for start in range(0, len(order), arguments.batch):
                    index = order[start : start + arguments.batch]
                    optimiser.zero_grad()
                    nn.functional.mse_loss(
                        model(title[index], meta[index]), target[index]
                    ).backward()
                    optimiser.step()

            model.eval()
            test_title, test_meta, test_target = data["test"]
            actual = test_target.numpy()
            with torch.no_grad():
                for damage in arguments.damage:
                    for level in arguments.levels:
                        damaged_title, damaged_meta = test_title, test_meta
                        noise = torch.Generator().manual_seed(seed)
                        if damage == "missing-title":
                            damaged_title = test_title * (1.0 - level)
                        elif damage == "missing-meta":
                            damaged_meta = test_meta * (1.0 - level)
                        elif damage == "noisy-title":
                            damaged_title = test_title + level * test_title.std() * torch.randn(
                                test_title.shape, generator=noise
                            )
                        else:
                            damaged_meta = test_meta + level * test_meta.std() * torch.randn(
                                test_meta.shape, generator=noise
                            )
                        predicted = model(damaged_title, damaged_meta).numpy()
                        key = f"{damage}@{level}"
                        results.setdefault(key, {}).setdefault(architecture, []).append(
                            float(spearmanr(predicted, actual).statistic)
                        )
        logger.info("%s trained on %d seeds", architecture, len(arguments.seeds))

    Path(arguments.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Wrote %s", arguments.out)

    for damage in arguments.damage:
        print(f"\n=== {damage} (alpha={arguments.alpha}) ===")
        print(f"{'level':>7}" + "".join(f"{a:>13}" for a in ARCHITECTURES) + "   snoek-early")
        for level in arguments.levels:
            row = results[f"{damage}@{level}"]
            medians = {a: statistics.median(row[a]) for a in ARCHITECTURES}
            edge = medians["late-snoek"] - medians["early"]
            print(
                f"{level:>7}"
                + "".join(f"{medians[a]:>13.4f}" for a in ARCHITECTURES)
                + f"{edge:>+14.4f}"
            )


def main() -> None:
    """Run one probe."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe", choices=["interaction", "robustness", "threshold"])
    parser.add_argument(
        "--strengths",
        type=float,
        nargs="+",
        default=[0.0, 0.05, 0.1, 0.2, 0.4, 0.8],
        help="Interaction added to the real target, in units of the target's own standard "
        "deviation. The product term is normalised first, so 1.0 really is one std.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite an existing results file.")
    parser.add_argument(
        "--kind",
        choices=list(INTERACTIONS),
        default="product",
        help="Which interaction the synthetic target carries. `product` is bilinear and a "
        "dot product can reach it; `max` and `disagree` are not.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="Interaction strength for the robustness probe. Zero keeps the task additive "
        "so the comparison is purely about damage, not about who can model a product.",
    )
    parser.add_argument(
        "--damage",
        nargs="+",
        default=["missing-title", "missing-meta", "noisy-title", "noisy-meta"],
    )
    parser.add_argument(
        "--levels",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
        help="Fraction removed, or noise as a multiple of the modality's own std.",
    )
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0, 2.0, 4.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 52)))
    parser.add_argument(
        "--hidden",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512],
        help="Swept, because a structural limit does not go away with capacity: if "
        "late-snoek trails at every width, the product term is unreachable rather "
        "than merely hard.",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--out",
        default="",
        help="Defaults per probe. A single shared default let a robustness smoke test "
        "overwrite 1,440 interaction runs, because both wrote to the same file.",
    )
    arguments = parser.parse_args()
    if not arguments.out:
        arguments.out = f"assets/fusion-{arguments.probe}.json"
    if Path(arguments.out).exists() and not arguments.force:
        raise SystemExit(
            f"{arguments.out} exists; pass --force to overwrite or --out to write elsewhere"
        )
    if arguments.probe == "interaction":
        run_interaction(arguments)
    elif arguments.probe == "threshold":
        run_threshold(arguments)
    else:
        run_robustness(arguments)


if __name__ == "__main__":
    main()
