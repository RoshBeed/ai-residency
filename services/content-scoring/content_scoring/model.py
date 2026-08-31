"""Four ways to get from a Hacker News post to a predicted score.

All four share the same inputs, the same target and the same head width, so the only
thing that differs between them is where the title and the metadata meet:

```
text_only        title -> pool -> MLP ------------------> score
features_only                     MLP(features) --------> score
early            title -> pool -> MLP(concat(both)) ----> score
late             title -> pool -> MLP ---.
                 temporal  -> MLP -------|
                 domain    -> MLP -------+-> head MLP --> score
                 surface   -> MLP -------|
                 author    -> MLP -------'
```

The two ablations are not filler. `features_only` is a genuinely strong model on Hacker
News -- the domain and the hour carry a lot -- so `early` and `late` have to beat it
before "the title matters" is a claim rather than an assumption, and `text_only` is what
says how much the title carries on its own.

**Early versus late is a question about capacity, not about information.** Both see
exactly the same inputs. Early fusion has one MLP over everything, so it can model any
interaction between a word in the title and the hour it was posted, but it has to find
the group structure itself and spends parameters doing so. Late fusion hands each group
its own tower, so each learns a representation without interference and the head sees
five summaries -- cheaper, and better when the groups genuinely are separable, worse
when the interactions matter.

The embedding matrix is not saved with the model. It is 104MB (86,438 x 300 floats)
shared by all four architectures, published once by `vocab.py`, and `save_checkpoint`'s
`state_dict` override exists for this case.
"""

import torch
from torch import nn

from config import Config, Fusion
from content_scoring.features import FEATURE_GROUPS, GROUP_SLICES

__all__ = [
    "BlockDiagonalScorer",
    "BranchMLP",
    "EarlyFusionScorer",
    "FeaturesOnlyScorer",
    "LateFusionScorer",
    "ScorePredictor",
    "TextOnlyScorer",
    "build_model",
    "describe",
    "load_predictor_weights",
    "predictor_state_dict",
]


class BranchMLP(nn.Module):
    """A stack of `num_layers` hidden layers, ReLU and dropout between them."""

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
        """Build the stack.

        Args:
            in_dim: Input width.
            hidden_dim: Hidden width, the same at every layer.
            num_layers: Hidden layers. Zero makes this an identity, which is what a
                one-feature branch usually wants.
            dropout: Dropout probability between layers.
        """
        super().__init__()
        layers: list[nn.Module] = []
        width = in_dim
        for _ in range(num_layers):
            layers += [nn.Linear(width, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            width = hidden_dim
        self.network = nn.Sequential(*layers)
        self.out_dim = width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the stack.

        Args:
            x: `(batch, in_dim)`.

        Returns:
            `(batch, out_dim)`.
        """
        output: torch.Tensor = self.network(x)
        return output


class ScorePredictor(nn.Module):
    """Base class holding the parts every architecture shares.

    Attributes:
        embeddings: The extended word vectors, frozen unless the run fine-tunes them.
        domain_embed: Learned domain representation, row 0 being `<rare>`.
    """

    def __init__(
        self,
        embedding_matrix: torch.Tensor,
        num_domains: int,
        config: Config,
    ) -> None:
        """Wire the shared inputs.

        Args:
            embedding_matrix: `(vocab_size, embed_size)` extended word vectors.
            num_domains: Rows the domain embedding needs.
            config: Supplies widths, dropout and whether the vectors are frozen.
        """
        super().__init__()
        # Cloned only when this run will train the vectors. Frozen, the matrix is shared
        # with whoever passed it in -- which matters at serving time, where the API holds
        # all four architectures at once: cloning unconditionally meant four copies of an
        # 86,438 x 300 table, 415MB of a pod's memory spent to say the same thing four
        # times. Nothing writes to it in the frozen case, so sharing is safe.
        weight = embedding_matrix if config.freeze_embeddings else embedding_matrix.clone()
        self.embeddings = nn.Embedding(
            embedding_matrix.shape[0], embedding_matrix.shape[1], _weight=weight
        )
        self.embeddings.weight.requires_grad_(not config.freeze_embeddings)
        self.domain_embed = nn.Embedding(num_domains, config.domain_embed_dim)
        self.embed_size = int(embedding_matrix.shape[1])
        self.config = config

    def pool_title(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mean of a title's word vectors, ignoring padding.

        Args:
            tokens: `(batch, max_tokens)` row ids, padded with zeros.
            mask: `(batch, max_tokens)` float mask, 1 for a real token.

        Returns:
            `(batch, embed_size)`. A title with no in-vocabulary words pools to the
            zero vector rather than to whatever row 0 happens to be -- the clamp on the
            denominator is what stops it dividing by zero and producing NaN.
        """
        vectors: torch.Tensor = self.embeddings(tokens) * mask.unsqueeze(-1)
        pooled: torch.Tensor = vectors.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        return pooled

    def feature_block(self, numeric: torch.Tensor, domain: torch.Tensor) -> torch.Tensor:
        """Every feature as one flat vector, domain embedded in place.

        Args:
            numeric: `(batch, numeric_dim)` scaled features.
            domain: `(batch,)` domain ids.

        Returns:
            `(batch, numeric_dim + domain_embed_dim)`.
        """
        embedded: torch.Tensor = self.domain_embed(domain)
        return torch.cat([numeric, embedded], dim=1)

    def forward(
        self, tokens: torch.Tensor, mask: torch.Tensor, numeric: torch.Tensor, domain: torch.Tensor
    ) -> torch.Tensor:
        """Predict `log1p(score)`.

        Args:
            tokens: `(batch, max_tokens)` padded row ids.
            mask: `(batch, max_tokens)` padding mask.
            numeric: `(batch, numeric_dim)` scaled features.
            domain: `(batch,)` domain ids.

        Returns:
            `(batch,)` predictions.

        Raises:
            NotImplementedError: On the base class.
        """
        raise NotImplementedError


class TextOnlyScorer(ScorePredictor):
    """The pooled title and nothing else."""

    def __init__(self, embedding_matrix: torch.Tensor, num_domains: int, config: Config) -> None:
        super().__init__(embedding_matrix, num_domains, config)
        self.branch = BranchMLP(
            self.embed_size, config.hidden_dim, config.num_layers, config.dropout
        )
        self.head = nn.Linear(self.branch.out_dim, 1)

    def forward(
        self, tokens: torch.Tensor, mask: torch.Tensor, numeric: torch.Tensor, domain: torch.Tensor
    ) -> torch.Tensor:
        del numeric, domain
        predicted: torch.Tensor = self.head(self.branch(self.pool_title(tokens, mask))).squeeze(-1)
        return predicted


class FeaturesOnlyScorer(ScorePredictor):
    """Every metadata feature, no text."""

    def __init__(self, embedding_matrix: torch.Tensor, num_domains: int, config: Config) -> None:
        super().__init__(embedding_matrix, num_domains, config)
        numeric_dim = sum(slot.stop - slot.start for slot in GROUP_SLICES.values())
        self.branch = BranchMLP(
            numeric_dim + config.domain_embed_dim,
            config.hidden_dim,
            config.num_layers,
            config.dropout,
        )
        self.head = nn.Linear(self.branch.out_dim, 1)

    def forward(
        self, tokens: torch.Tensor, mask: torch.Tensor, numeric: torch.Tensor, domain: torch.Tensor
    ) -> torch.Tensor:
        del tokens, mask
        predicted: torch.Tensor = self.head(
            self.branch(self.feature_block(numeric, domain))
        ).squeeze(-1)
        return predicted


class EarlyFusionScorer(ScorePredictor):
    """One MLP over the title and the features concatenated at the input."""

    def __init__(self, embedding_matrix: torch.Tensor, num_domains: int, config: Config) -> None:
        super().__init__(embedding_matrix, num_domains, config)
        numeric_dim = sum(slot.stop - slot.start for slot in GROUP_SLICES.values())
        self.branch = BranchMLP(
            self.embed_size + numeric_dim + config.domain_embed_dim,
            config.hidden_dim,
            config.num_layers,
            config.dropout,
        )
        self.head = nn.Linear(self.branch.out_dim, 1)

    def forward(
        self, tokens: torch.Tensor, mask: torch.Tensor, numeric: torch.Tensor, domain: torch.Tensor
    ) -> torch.Tensor:
        combined = torch.cat(
            [self.pool_title(tokens, mask), self.feature_block(numeric, domain)], dim=1
        )
        predicted: torch.Tensor = self.head(self.branch(combined)).squeeze(-1)
        return predicted


class LateFusionScorer(ScorePredictor):
    """One tower per branch, joined at the head.

    Attributes:
        branches: Text plus one tower per feature group, keyed by group name.
        disabled: Groups zeroed at the input, which is how the leave-one-group-out
            ablation runs without retraining or rebuilding anything.
    """

    def __init__(self, embedding_matrix: torch.Tensor, num_domains: int, config: Config) -> None:
        super().__init__(embedding_matrix, num_domains, config)
        widths = {"text": self.embed_size}
        for group in FEATURE_GROUPS:
            slot = GROUP_SLICES[group]
            width = slot.stop - slot.start
            if group == "domain":
                width += config.domain_embed_dim
            widths[group] = width

        # A branch narrower than the hidden width gets one layer rather than the full
        # stack: three 128-wide layers over a three-column author block is capacity
        # spent on nothing, and it makes the late model slower than early for no gain.
        self.branches = nn.ModuleDict(
            {
                name: BranchMLP(
                    width,
                    config.hidden_dim,
                    max(1, config.num_layers - 1) if name != "text" else config.num_layers,
                    config.dropout,
                )
                for name, width in widths.items()
            }
        )
        self.head = nn.Sequential(
            nn.Linear(config.hidden_dim * len(self.branches), config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )
        self.disabled: set[str] = set()

    def forward(
        self, tokens: torch.Tensor, mask: torch.Tensor, numeric: torch.Tensor, domain: torch.Tensor
    ) -> torch.Tensor:
        parts = {"text": self.pool_title(tokens, mask)}
        for group in FEATURE_GROUPS:
            block = numeric[:, GROUP_SLICES[group]]
            if group == "domain":
                block = torch.cat([block, self.domain_embed(domain)], dim=1)
            parts[group] = block

        outputs = []
        for name, branch in self.branches.items():
            value = parts[name]
            if name in self.disabled:
                value = torch.zeros_like(value)
            outputs.append(branch(value))
        predicted: torch.Tensor = self.head(torch.cat(outputs, dim=1)).squeeze(-1)
        return predicted


class BlockDiagonalScorer(ScorePredictor):
    """Early fusion with the cross-group connections removed, and nothing else changed.

    `LateFusionScorer` gives each group a full-width tower and adds a head to combine
    them, so against early fusion it varies connectivity, width *and* depth at once --
    three explanations for any difference it shows. This varies only connectivity.

    The first layer is the same `329 -> hidden` map early fusion uses, with the weight
    matrix constrained to be block-diagonal: each group's inputs reach only its own share
    of the hidden units, and the shares are proportional to how many inputs each group
    brings. Everything after the first layer is identical to early fusion. It is
    therefore strictly *smaller* -- it is early fusion with weights deleted -- which is
    what makes "does crosstalk help?" the only question being asked.

    Attributes:
        disabled: Groups zeroed at the input, for the leave-one-group-out ablation.
    """

    def __init__(self, embedding_matrix: torch.Tensor, num_domains: int, config: Config) -> None:
        super().__init__(embedding_matrix, num_domains, config)
        widths = {"text": self.embed_size}
        for group in FEATURE_GROUPS:
            slot = GROUP_SLICES[group]
            widths[group] = (slot.stop - slot.start) + (
                config.domain_embed_dim if group == "domain" else 0
            )
        total_in = sum(widths.values())

        # Hidden units split in proportion to inputs, with a floor of one so a
        # three-column group still reaches the head. The remainder goes to the widest
        # group, so the shares sum to exactly `hidden_dim` and the layer is the same size
        # as early fusion's.
        shares = {
            name: max(1, round(config.hidden_dim * w / total_in)) for name, w in widths.items()
        }
        widest = max(shares, key=lambda name: widths[name])
        shares[widest] += config.hidden_dim - sum(shares.values())

        self.blocks = nn.ModuleDict(
            {name: BranchMLP(widths[name], shares[name], 1, config.dropout) for name in widths}
        )
        self.rest = BranchMLP(
            config.hidden_dim, config.hidden_dim, max(0, config.num_layers - 1), config.dropout
        )
        self.head = nn.Linear(config.hidden_dim, 1)
        self.disabled: set[str] = set()

    def forward(
        self, tokens: torch.Tensor, mask: torch.Tensor, numeric: torch.Tensor, domain: torch.Tensor
    ) -> torch.Tensor:
        parts = {"text": self.pool_title(tokens, mask)}
        for group in FEATURE_GROUPS:
            block = numeric[:, GROUP_SLICES[group]]
            if group == "domain":
                block = torch.cat([block, self.domain_embed(domain)], dim=1)
            parts[group] = block

        pieces = []
        for name, projection in self.blocks.items():
            value = parts[name]
            if name in self.disabled:
                value = torch.zeros_like(value)
            pieces.append(projection(value))
        predicted: torch.Tensor = self.head(self.rest(torch.cat(pieces, dim=1))).squeeze(-1)
        return predicted


_ARCHITECTURES: dict[Fusion, type[ScorePredictor]] = {
    "text_only": TextOnlyScorer,
    "features_only": FeaturesOnlyScorer,
    "early": EarlyFusionScorer,
    "late": LateFusionScorer,
    "block_diagonal": BlockDiagonalScorer,
}


def build_model(embedding_matrix: torch.Tensor, num_domains: int, config: Config) -> ScorePredictor:
    """Build whichever architecture `config.fusion` names.

    Args:
        embedding_matrix: The extended word vectors.
        num_domains: Rows the domain embedding needs.
        config: Supplies `fusion` and every width.

    Returns:
        The model, on CPU.

    Raises:
        ValueError: If `fusion` names an architecture that does not exist. Unreachable
            through `Config`, which validates the Literal, but reachable by a caller
            constructing one directly.
    """
    if config.fusion not in _ARCHITECTURES:
        raise ValueError(
            f"unknown fusion {config.fusion!r}; expected one of {list(_ARCHITECTURES)}"
        )
    return _ARCHITECTURES[config.fusion](embedding_matrix, num_domains, config)


def load_predictor_weights(model: ScorePredictor, state_dict: dict[str, torch.Tensor]) -> None:
    """Load a checkpoint that deliberately omits the embedding table.

    Args:
        model: A built scorer.
        state_dict: The saved weights, without `embeddings.weight`.

    Raises:
        RuntimeError: If anything other than the embedding table failed to match.

    `strict=False` is required -- the table is stored separately and is 104MB -- but on
    its own it is dangerous: a renamed layer makes every key miss, `load_state_dict`
    returns quietly, and the service happily serves a randomly-initialised head. Nothing
    fails, the API answers, and the numbers are noise. So the omission is checked to be
    exactly the one that was intended.
    """
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    unexplained = set(missing) - {"embeddings.weight"}
    if unexplained or unexpected:
        raise RuntimeError(
            f"checkpoint does not fit this architecture: missing {sorted(unexplained)}, "
            f"unexpected {sorted(unexpected)}"
        )


def describe(model: ScorePredictor) -> dict[str, object]:
    """What a model is, in numbers, for anything that reports the comparison.

    Args:
        model: A built scorer.

    Returns:
        `{"parameters", "hidden", "first_layer", "connectivity"}`.

    Read off the built modules rather than written down beside them. The four
    architectures differ in ways a prose label hides -- `late` carries five times the
    hidden units of `early` and an extra layer, so "late versus early" is not only a
    question about connectivity -- and a UI that states the difference should be reading
    it from the model rather than from a comment that can go stale.
    """
    linear = [
        (name, layer) for name, layer in model.named_modules() if isinstance(layer, nn.Linear)
    ]
    first = [
        f"{layer.in_features}→{layer.out_features}"
        for name, layer in linear
        if "head" not in name and "rest" not in name
    ]
    hidden = sum(
        layer.out_features for name, layer in linear if "head" not in name and "rest" not in name
    )
    return {
        "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "hidden": hidden,
        "first_layer": " | ".join(first) or "—",
        "connectivity": "dense" if len(first) == 1 else f"{len(first)} separate blocks",
    }


def predictor_state_dict(model: ScorePredictor) -> dict[str, torch.Tensor]:
    """The model's weights without the word-embedding table.

    Args:
        model: A built scorer.

    Returns:
        Its `state_dict` minus `embeddings.weight`.

    The table is 104MB and identical across all four architectures; it is published once
    by `vocab.py` and resolved separately at load time. Saving it with each model would
    put 415MB on the Hub to say the same thing four times.
    """
    return {key: value for key, value in model.state_dict().items() if key != "embeddings.weight"}
