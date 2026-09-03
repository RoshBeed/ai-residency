"""Build matched-neuron specs for the tower-shape experiment.

Every model at a budget gets the *same number of hidden units*; the only thing that
differs is how they are arranged. That is what makes the comparison about shape rather
than about capacity, and it is easy to get wrong by a few units through rounding, so it
is computed here and asserted rather than written out by hand.

    early         all B units in one shared stack, every input concatenated at the input
    late-uniform  half the units as five equal depth-1 towers, half as the shared trunk
    late-tuned    the same half, the same per-tower budget, each tower at the depth its
                  input preferred in the shape probe

late-uniform and late-tuned differ *only* in tower depth, which isolates the claim: that
some inputs want narrow-and-deep and others wide-and-shallow, and that late fusion's one
structural advantage is being able to give them that.
"""

GROUPS = ("text", "temporal", "domain", "title_surface", "author")


def split_evenly(total: int, parts: int) -> list[int]:
    """`total` divided into `parts` whole numbers summing to exactly `total`."""
    base, remainder = divmod(total, parts)
    return [base + (1 if index < remainder else 0) for index in range(parts)]


def tower_spec(depths: dict[str, int], tower_budget: int) -> str:
    """A `tower_shapes` string giving every group the same budget at its own depth.

    Args:
        depths: Group name to how many layers its tower should have.
        tower_budget: Hidden units to share across all five towers.

    Returns:
        A spec string for `Config.tower_shapes`.

    Each group gets the same number of units; only the depth differs. Units within a
    tower are split as evenly as they divide, so the totals stay exact.
    """
    per_group = split_evenly(tower_budget, len(GROUPS))
    clauses = []
    for group, units in zip(GROUPS, per_group, strict=True):
        widths = split_evenly(units, max(1, depths.get(group, 1)))
        clauses.append(f"{group}:{','.join(str(w) for w in widths)}")
    return ";".join(clauses)


def specs_for(budget: int, depths: dict[str, int]) -> dict[str, dict[str, object]]:
    """The three architectures at one budget, all with exactly `budget` hidden units.

    Args:
        budget: Total hidden units every model must have.
        depths: Per-group preferred depth, from the shape probe.

    Returns:
        Architecture name to the config overrides that build it.
    """
    trunk, towers = budget // 2, budget - budget // 2
    return {
        "early": {
            "AI_RESIDENCY_FUSION_LAYER": "1",
            "AI_RESIDENCY_HIDDEN_WIDTHS": ",".join(str(w) for w in split_evenly(budget, 2)),
            "AI_RESIDENCY_TOWER_SHAPES": "",
        },
        "late-uniform": {
            "AI_RESIDENCY_FUSION_LAYER": "2",
            "AI_RESIDENCY_HIDDEN_WIDTHS": str(trunk),
            "AI_RESIDENCY_TOWER_SHAPES": tower_spec(dict.fromkeys(GROUPS, 1), towers),
        },
        "late-tuned": {
            "AI_RESIDENCY_FUSION_LAYER": "2",
            "AI_RESIDENCY_HIDDEN_WIDTHS": str(trunk),
            "AI_RESIDENCY_TOWER_SHAPES": tower_spec(depths, towers),
        },
    }


def units(spec: dict[str, object]) -> int:
    """Total hidden units a spec builds, for the assertion that they match."""
    total = sum(int(w) for w in str(spec["AI_RESIDENCY_HIDDEN_WIDTHS"]).split(",") if w)
    towers = str(spec["AI_RESIDENCY_TOWER_SHAPES"])
    if towers:
        for clause in towers.split(";"):
            total += sum(int(w) for w in clause.partition(":")[2].split(",") if w)
    return total
