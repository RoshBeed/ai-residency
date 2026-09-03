"""Four ways to get from a Hacker News post to a predicted score.

All four share the same inputs, the same target and the same head width, so the only
thing that differs between them is where the title and the metadata meet:

```
text_only        title -> pool -> MLP ------------------> score
features_only                     MLP(features) --------> score
early            title -> pool -> MLP(concat(both)) ----> score
late             title -> pool -.
                 temporal  ------|  block-diagonal
                 domain    ------+  MLP(concat(both)) --> score
                 surface   ------|  (no cross-group weights)
                 author    ------'
```

The two ablations are not filler. `features_only` is a genuinely strong model on Hacker
News -- the domain and the hour carry a lot -- so `early` and `late` have to beat it
before "the title matters" is a claim rather than an assumption, and `text_only` is what
says how much the title carries on its own.

**Early versus late is a question about connectivity, and only about connectivity.**
Both see the same inputs, and `late` is built as `early` with the cross-group weights
deleted -- same input width, same hidden width, same depth, same head. Early fusion can
model an interaction between a word in the title and the hour it was posted; late fusion
structurally cannot, because no group reaches another's hidden units until the head. So
the gap between the two measures what that crosstalk is worth, with nothing else varying
to explain it.

The embedding matrix is not saved with the model. It is 104MB (86,438 x 300 floats)
shared by all four architectures, published once by `vocab.py`, and `save_checkpoint`'s
`state_dict` override exists for this case.
"""

from collections.abc import Callable, Sequence
from typing import get_args

import torch
from torch import nn

from config import Allocation, Config, Fusion
from content_scoring.features import FEATURE_GROUPS, GROUP_SLICES

__all__ = [
    "BranchMLP",
    "FeaturesOnlyScorer",
    "FusionScorer",
    "ScorePredictor",
    "TextOnlyScorer",
    "build_model",
    "describe",
    "graph_dot",
    "load_predictor_weights",
    "predictor_state_dict",
]


class BranchMLP(nn.Module):
    """A stack of hidden layers of given widths, ReLU and dropout between them."""

    def __init__(self, in_dim: int, widths: Sequence[int], dropout: float) -> None:
        """Build the stack.

        Args:
            in_dim: Input width.
            widths: Output width of each hidden layer, in order. Empty makes this an
                identity, which is what a group with no private layers wants.
            dropout: Dropout probability between layers.
        """
        super().__init__()
        layers: list[nn.Module] = []
        width = in_dim
        for out_width in widths:
            layers += [nn.Linear(width, out_width), nn.ReLU(), nn.Dropout(dropout)]
            width = out_width
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
        disabled: Input groups zeroed before anything reads them, for the
            leave-one-group-out ablation. On the base class so the ablation runs on any
            architecture rather than only on the one with separable blocks -- measured
            on the block-diagonal model alone it reports which group was handed the most
            hidden units as much as which group carries the most signal, because that
            model allocates its width in proportion to input width.
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
        self.disabled: set[str] = set()

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

        A group named in `disabled` is zeroed here, which for the domain means its
        embedding as well as its flag -- zeroing only the numeric column would leave the
        model an 8-d representation of exactly the thing being ablated.
        """
        embedded: torch.Tensor = self.domain_embed(domain)
        if self.disabled:
            numeric = numeric.clone()
            for group in self.disabled:
                if group in GROUP_SLICES:
                    numeric[:, GROUP_SLICES[group]] = 0.0
            if "domain" in self.disabled:
                embedded = torch.zeros_like(embedded)
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
        self.branch = BranchMLP(self.embed_size, config.hidden_widths, config.dropout)
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
            numeric_dim + config.domain_embed_dim, config.hidden_widths, config.dropout
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


class FusionScorer(ScorePredictor):
    """Title and metadata, meeting at a configurable layer.

    Layers before `fusion_layer` are **block-diagonal**: each input group is projected
    into its own share of that layer's width and reaches nothing else. Layers from
    `fusion_layer` on are dense and shared. `fusion_layer=1` therefore *is* early
    fusion -- one dense stack over everything -- and a larger value delays the meeting
    without changing anything else about the network.

    This is one class rather than an `EarlyFusionScorer` and a `LateFusionScorer`
    because they were always the same network with a different fusion point, and keeping
    them apart is what let an earlier version drift: it gave each group a full-width
    tower plus an extra layer, so against early fusion it varied connectivity, width and
    depth at once, with 4.5x the parameters and three explanations for any gap.

    Shares are proportional to how many input columns each group brings, floored at one
    and summing exactly to the layer's width. That is what makes the first layer the
    same shape as early fusion's, so a block-diagonal model is early fusion with weights
    deleted rather than a differently-sized network -- and the gap between them measures
    the crosstalk and nothing else.

    At least one shared layer always remains. Without it the model would be
    `sum_g w_g . f_g(x_g)`, a generalised additive model that cannot represent any
    interaction between the title and the timestamp at all -- which an earlier
    single-layer version silently was.

    Attributes:
        fusion_layer: Where the groups meet, after clamping to the stack's depth.
    """

    def __init__(
        self,
        embedding_matrix: torch.Tensor,
        num_domains: int,
        config: Config,
        fusion_layer: int | None = None,
    ) -> None:
        """Build the stack.

        Args:
            embedding_matrix: `(vocab_size, embed_size)` extended word vectors.
            num_domains: Rows the domain embedding needs.
            config: Supplies the widths, dropout and the default fusion point.
            fusion_layer: Overrides `config.fusion_layer`. `build_model` pins this to 1
                for `fusion="early"`.

        Raises:
            ValueError: If the fusion point is below 1, or the stack has no layers.
        """
        super().__init__(embedding_matrix, num_domains, config)
        widths = list(config.hidden_widths)
        if not widths:
            raise ValueError("hidden_widths must name at least one layer")
        requested = config.fusion_layer if fusion_layer is None else fusion_layer
        if requested < 1:
            raise ValueError(f"fusion_layer must be at least 1, got {requested}")
        # Clamped rather than rejected: a Bayes sweep draws fusion_layer independently of
        # the shape, so `5` against a two-layer stack is a normal sample, not an error. It
        # costs a run either way; clamping makes that run useful. The effective value goes
        # to W&B so the analysis groups on what was built, not what was asked for.
        self.fusion_layer = min(requested, len(widths))

        # Resize the groups before they meet, so the comparison is not dominated by the
        # title arriving 100x wider than the author's history for reasons that have
        # nothing to do with how much either one says.
        raw_widths = _input_widths(self)
        # Every group gets an entry, `Identity` where nothing is resized, so `forward`
        # has no branch and the two ModuleDicts always share their keys.
        self.project = nn.ModuleDict(
            {
                name: (
                    nn.Identity()
                    if (target := _projected_width(name, size, config)) == size
                    else nn.Linear(size, target)
                )
                for name, size in raw_widths.items()
            }
        )
        group_widths = {
            name: _projected_width(name, size, config) for name, size in raw_widths.items()
        }

        towers = _parse_towers(config.tower_shapes)
        if towers:
            # Explicit per-group towers: each input gets its own depth and widths, and
            # `hidden_widths` becomes the shared trunk.
            self.blocks = nn.ModuleDict(
                {
                    name: BranchMLP(size, towers.get(name, ()), config.dropout)
                    for name, size in group_widths.items()
                }
            )
            joined = sum(
                towers[name][-1] if towers.get(name) else size
                for name, size in group_widths.items()
            )
            shared = widths
        else:
            private = widths[: self.fusion_layer - 1]
            shared = widths[self.fusion_layer - 1 :]
            self.blocks = nn.ModuleDict(
                {
                    name: BranchMLP(
                        size,
                        [_shares(group_widths, w, config.block_allocation)[name] for w in private],
                        dropout=config.dropout,
                    )
                    for name, size in group_widths.items()
                }
            )
            joined = sum(private[-1:]) or sum(group_widths.values())
        self.rest = BranchMLP(joined, shared, config.dropout)
        self.head = nn.Linear(self.rest.out_dim, 1)

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
        # Not `block`: that name is bound to a tensor above, and reusing it makes the
        # module look like a tensor to the type checker.
        for name, layer in self.blocks.items():
            value = parts[name]
            if name in self.disabled:
                value = torch.zeros_like(value)
            pieces.append(layer(self.project[name](value)))
        predicted: torch.Tensor = self.head(self.rest(torch.cat(pieces, dim=1))).squeeze(-1)
        return predicted


def _parse_towers(spec: str) -> dict[str, tuple[int, ...]]:
    """Per-group tower shapes from `text:128;temporal:16,16`.

    Args:
        spec: Semicolon-separated `group:widths` pairs. Empty means no explicit towers.

    Returns:
        Group name to its layer widths. A group the spec omits gets no tower.

    Raises:
        ValueError: If a clause is malformed or names an unknown group.

    A string rather than five config fields because it has to survive a shell
    environment: `AI_RESIDENCY_TOWER_SHAPES=text:128;author:8,8` is one variable a sweep
    can set, where five would need five.
    """
    if not spec.strip():
        return {}
    towers: dict[str, tuple[int, ...]] = {}
    for clause in spec.split(";"):
        if not clause.strip():
            continue
        name, _, widths = clause.partition(":")
        name = name.strip()
        if name not in INPUT_LABELS:
            raise ValueError(f"unknown input group {name!r} in tower_shapes")
        towers[name] = tuple(int(part) for part in widths.split(",") if part.strip())
    return towers


def _projected_width(name: str, size: int, config: Config) -> int:
    """The width a group arrives at the fusion layer with.

    Args:
        name: The input group.
        size: How many columns it brings natively.
        config: Supplies `input_projection` and `projection_dim`.

    Returns:
        The width after any resizing.

    The three modes answer the same objection from different directions: the title is
    300-d because word2vec chose 300, not because a title says a hundred times more than
    an author's history does. `text_down` shrinks the title to meet the metadata,
    `features_up` grows the metadata to meet the title, and `all` puts everything at one
    width. None of them adds information -- they change only how much of the fusion layer
    each input can reach.
    """
    mode = config.input_projection
    if mode == "none":
        return size
    if mode == "all":
        return config.projection_dim
    if mode == "text_down":
        return config.projection_dim if name == "text" else size
    return size if name == "text" else config.projection_dim


def _shares(
    group_widths: dict[str, int], width: int, allocation: Allocation = "proportional"
) -> dict[str, int]:
    """Split one layer's width between the input groups.

    Args:
        group_widths: How many input columns each group brings.
        width: The layer's total width.
        allocation: `proportional` to columns, or `equal` between groups.

    Returns:
        Each group's share, at least one each and summing to exactly `width`.

    **Proportional is not the neutral rule it looks like.** The title arrives as a 300-d
    mean of word vectors, so its width reflects the embedding size rather than how much
    it says; the author's history is three engineered scalars. Splitting by column count
    gives the title 233 of 256 units and the author 2 -- and on this data the 29 metadata
    columns out-predict the 300 title columns, so the rule hands the weaker input eight
    times the capacity, then a fusion-point sweep measures that instead of fusion.

    `equal` divides the width evenly. Either way the shares sum to exactly `width`, so
    the block-diagonal layer stays the same size as the dense layer it is carved from and
    late fusion remains early fusion with weights deleted.
    """
    names = list(group_widths)
    if allocation == "equal":
        shares = {name: max(1, width // len(names)) for name in names}
    else:
        total = sum(group_widths.values())
        shares = {name: max(1, round(width * size / total)) for name, size in group_widths.items()}
    widest = max(shares, key=lambda name: group_widths[name])
    shares[widest] += width - sum(shares.values())
    return shares


# `early` and `late` are the same class at different fusion points. Early fusion is
# pinned to 1 rather than reading the config, so it stays the fixed reference the curve
# is measured against however `fusion_layer` is swept.
_FUSION_POINTS: dict[Fusion, int | None] = {"early": 1, "late": None}


def build_model(embedding_matrix: torch.Tensor, num_domains: int, config: Config) -> ScorePredictor:
    """Build whichever architecture `config.fusion` names.

    Args:
        embedding_matrix: The extended word vectors.
        num_domains: Rows the domain embedding needs.
        config: Supplies `fusion`, the layer widths and the fusion point.

    Returns:
        The model, on CPU.

    Raises:
        ValueError: If `fusion` names an architecture that does not exist. Unreachable
            through `Config`, which validates the Literal, but reachable by a caller
            constructing one directly.
    """
    if config.fusion == "text_only":
        return TextOnlyScorer(embedding_matrix, num_domains, config)
    if config.fusion == "features_only":
        return FeaturesOnlyScorer(embedding_matrix, num_domains, config)
    if config.fusion in _FUSION_POINTS:
        return FusionScorer(embedding_matrix, num_domains, config, _FUSION_POINTS[config.fusion])
    raise ValueError(f"unknown fusion {config.fusion!r}; expected one of {list(get_args(Fusion))}")


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

    Two different mismatches reach here now that every fused model is one class. A
    checkpoint from a different *architecture* misses keys; one from the same
    architecture at a different width or fusion point has all the keys and the wrong
    shapes, which `load_state_dict` raises itself. Both are the same failure to a caller,
    so both leave by the same message -- serving logs it verbatim.
    """
    try:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
    except RuntimeError as error:
        raise RuntimeError(f"checkpoint does not fit this architecture: {error}") from error
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
        `{"parameters", "hidden", "depth", "fusion_layer", "first_layer",
        "shared_layers", "connectivity"}`.

    Read off the built modules rather than written down beside them. The first layer is
    the only place the fused architectures differ, so it is reported block by block;
    everything the groups share after meeting is reported separately. A UI that states
    the difference should be reading it from the model rather than from a comment that
    can go stale.
    """
    blocks = first_layer_blocks(model)
    return {
        "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "hidden": sum(out for _, _, out in blocks),
        "depth": len(model.config.hidden_widths),
        "fusion_layer": getattr(model, "fusion_layer", 1),
        "first_layer": " | ".join(f"{i}→{o}" for _, i, o in blocks) or "—",
        "shared_layers": " → ".join(_shared_shapes(model)) or "—",
        "connectivity": "dense" if len(blocks) == 1 else f"{len(blocks)} separate blocks",
    }


def _shared_shapes(model: ScorePredictor) -> list[str]:
    """The layers every input group passes through together, after the first.

    A `FusionScorer` fused at layer one has no private layers, so its shared stack is
    everything in `rest` *after* the first -- which `first_layer_blocks` already
    reported. Fused later, the whole of `rest` is shared and the private layers live in
    `blocks`.
    """
    source = model.rest if isinstance(model, FusionScorer) else getattr(model, "branch", None)
    network = getattr(source, "network", None)
    if not isinstance(network, nn.Sequential):
        return []
    linears = [layer for layer in network if isinstance(layer, nn.Linear)]
    fused_at_one = isinstance(model, FusionScorer) and model.fusion_layer == 1
    if fused_at_one or not isinstance(model, FusionScorer):
        linears = linears[1:]
    return [f"{layer.in_features}→{layer.out_features}" for layer in linears]


def _graph_nodes(model: ScorePredictor, prefix: str = "") -> list[str]:
    """The DOT nodes and edges for one architecture, ids optionally prefixed.

    Args:
        model: A built scorer.
        prefix: Prepended to every node id, so several graphs can share one digraph
            without their nodes colliding.

    Returns:
        DOT statement lines, without the enclosing braces.
    """
    inputs = {name: (INPUT_LABELS[name], width) for name, width in _input_widths(model).items()}
    box = "style=filled fontcolor=white fillcolor="

    def node(name: str) -> str:
        """This graph's id for a node, unique across the graphs sharing a digraph."""
        return f"{prefix}{name}"

    lines = [f'  {node("score")} [label="expected\\nupvotes" shape=oval {box}"#eb6834"];']

    blocks = model.blocks if isinstance(model, FusionScorer) and model.fusion_layer > 1 else None
    if blocks is None:
        # Fused at the first layer: every input reaches one dense stack.
        used = [key for key in inputs if _uses(model, key)]
        source = model.rest if isinstance(model, FusionScorer) else model.branch
        width = _first_width(source)
        for key in used:
            label, size = inputs[key]
            lines.append(
                f'  {node(key)} [label="{label}\\n{size}" style=filled fillcolor="#e8f0fb"];'
            )
            lines.append(f"  {node(key)} -> {node('dense')};")
        total = sum(inputs[key][1] for key in used)
        lines.append(
            f'  {node("dense")} [label="one dense layer\\n{total} → {width}" {box}"#2a78d6"];'
        )
        lines.extend(_shared_nodes(model, node, "dense", box))
        return lines

    for key, module in blocks.items():
        label, size = inputs[key]
        out = _first_width(module) or size
        lines.append(f'  {node(key)} [label="{label}\\n{size}" style=filled fillcolor="#e8f0fb"];')
        lines.append(f'  {node(key)}_l [label="{size} → {out}" {box}"#2a78d6"];')
        lines.append(f"  {node(key)} -> {node(key)}_l -> {node('concat')};")
    width = sum(_first_width(module) for module in blocks.values())
    lines.append(f'  {node("concat")} [label="concat\\n{width}" shape=point width=0.12];')
    lines.extend(_shared_nodes(model, node, "concat", box))
    return lines


def _shared_nodes(
    model: ScorePredictor, node: Callable[[str], str], source: str, box: str
) -> list[str]:
    """The layers every input group passes through together, from the join to the score.

    Args:
        model: A built scorer.
        node: This graph's id-maker, so ids stay unique inside a shared digraph.
        source: Id of the node the shared stack reads from.
        box: The DOT attribute prefix for a filled node.

    Returns:
        DOT statements for the shared stack and the edge into the score.

    Drawn for every architecture because it *is* the same stack in each: they differ in
    where the groups meet and in nothing else, and a diagram showing the shared part for
    only one of them would hide the thing that makes them comparable.
    """
    shared = _shared_shapes(model)
    if not shared:
        return [f"  {node(source)} -> {node('score')};"]
    label = " → ".join(shared)
    lines = [f'  {node("shared")} [label="shared\\n{label}" {box}"#4a3aa7"];']
    lines.append(f"  {node(source)} -> {node('shared')} -> {node('score')};")
    return lines


def _graph_header(rankdir: str) -> list[str]:
    """The digraph preamble every architecture diagram shares."""
    return [
        "digraph {",
        f"  rankdir={rankdir};",
        "  node [shape=box style=rounded fontname=Helvetica fontsize=10];",
        '  edge [color="#898781"];',
    ]


def graph_dot(model: ScorePredictor, rankdir: str = "BT") -> str:
    """The model's structure as Graphviz DOT, for rendering in the UI.

    Args:
        model: A built scorer.
        rankdir: Graphviz direction. `BT` puts the inputs at the bottom and the score at
            the top, matching the course diagram in the write-up.

    Returns:
        A DOT digraph from the inputs to the predicted score.

    Built from the model's own modules, the same way `describe` is, so a diagram cannot
    drift from the network it claims to show. `make onnx` writes the full computation
    graph for Netron; this is the shape a reader needs while looking at four predictions.
    """
    return "\n".join([*_graph_header(rankdir), *_graph_nodes(model), "}"])


def graph_dot_all(models: dict[str, ScorePredictor], rankdir: str = "BT") -> str:
    """Every architecture in one digraph, so they are drawn to one scale.

    Args:
        models: Fusion name to built scorer, in the order they should appear.
        rankdir: Graphviz direction, as for `graph_dot`.

    Returns:
        One DOT digraph holding a labelled cluster per architecture.

    Four separate graphs each scaled to the same column width is not a comparison: the
    tall narrow one (title only, two nodes stacked) is blown up and the wide one (early
    fusion, five inputs side by side) is shrunk, so the same "expected upvotes" node
    comes out a different size in each and the diagrams read as though they were drawn at
    different zoom levels. Laying them out as clusters of a single graph makes Graphviz
    size every node once, which is what makes the four visually comparable.
    """
    lines = _graph_header(rankdir)
    lines.append("  compound=true;")
    lines.append('  graph [fontname=Helvetica fontsize=11 fontcolor="#52514e" color="#c3c2b7"];')
    for name, model in models.items():
        lines.append(f"  subgraph cluster_{name} {{")
        lines.append(f'    label="{ARCHITECTURE_LABELS.get(name, name)}";')
        lines.append("    labelloc=b;")
        lines.extend(f"  {line}" for line in _graph_nodes(model, prefix=f"{name}_"))
        lines.append("  }")
    lines.append("}")
    return "\n".join(lines)


# What a reader calls each architecture, for a diagram that shows more than one.
ARCHITECTURE_LABELS = {
    "text_only": "Title only",
    "features_only": "Metadata only",
    "early": "Early fusion",
    "late": "Late fusion",
}

# What a reader calls each input group. `graph_dot` and the connectivity figure label
# the same five things, so the names live here rather than in whichever drew them first.
INPUT_LABELS = {
    "text": "title",
    "temporal": "when posted",
    "domain": "where it links",
    "title_surface": "title shape",
    "author": "author history",
}


def _input_widths(model: ScorePredictor) -> dict[str, int]:
    """How many columns each input group contributes, in input order."""
    widths = {"text": model.embed_size}
    for group in FEATURE_GROUPS:
        slot = GROUP_SLICES[group]
        widths[group] = (slot.stop - slot.start) + (
            model.config.domain_embed_dim if group == "domain" else 0
        )
    return widths


def first_layer_blocks(model: ScorePredictor) -> list[tuple[str, int, int]]:
    """The first layer as `(label, in_features, out_features)` per block.

    Args:
        model: A built scorer.

    Returns:
        One entry per independent block, in input order. A dense architecture has a
        single entry spanning every input; the block-diagonal one has five.

    This is what makes the connectivity figure a *reading* of the model rather than a
    drawing beside it. The two are the same layer shape and differ only in how many
    blocks it is cut into, which is the whole claim the figure exists to show.
    """
    if isinstance(model, FusionScorer) and model.fusion_layer > 1:
        blocks = []
        for name, block in model.blocks.items():
            in_features, out_features = _linear_shape(getattr(block, "network", None))
            blocks.append((INPUT_LABELS.get(name, name), in_features, out_features))
        return blocks
    # Fused at layer one: the shared stack *is* the first layer, and every input reaches
    # all of it. `blocks` is a stack of identities in that case, so it has no shape to
    # report and the dense `rest` is what the diagram should show.
    source = model.rest if isinstance(model, FusionScorer) else getattr(model, "branch", None)
    in_features, out_features = _linear_shape(getattr(source, "network", None))
    return [("all inputs", in_features, out_features)]


def _first_width(module: object) -> int:
    """Output width of a BranchMLP's first Linear, or 0 if it has no layers.

    `nn.Module` attribute access is typed `Tensor | Module`, so every read of
    `branch.network[0].out_features` needs narrowing. Done once here rather than at each
    of the five call sites.
    """
    network = getattr(module, "network", None)
    if not isinstance(network, nn.Sequential) or len(network) == 0:
        return 0
    layer = network[0]
    return layer.out_features if isinstance(layer, nn.Linear) else 0


def _linear_shape(module: object) -> tuple[int, int]:
    """`(in_features, out_features)` of a Sequential's first Linear."""
    if isinstance(module, nn.Sequential) and len(module) and isinstance(module[0], nn.Linear):
        return module[0].in_features, module[0].out_features
    if isinstance(module, nn.Linear):
        return module.in_features, module.out_features
    return 0, 0


def _uses(model: ScorePredictor, group: str) -> bool:
    """Whether an architecture reads a given input at all."""
    fusion = model.config.fusion
    if fusion == "text_only":
        return group == "text"
    if fusion == "features_only":
        return group != "text"
    return True


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
