"""Choose each input's tower depth from the shape probe, and emit Step 2's specs.

    cd services/content-scoring
    uv run python ../../notebooks/content-scoring/pick_depths.py

A depth is only taken from the probe when it wins by more than half the spread it was
measured against, and only from the *large* budgets, where a deep shape is not also a
narrow first layer. Anything that fails either test falls back to depth 1, so a shape
preference has to be real to reach the experiment.
"""

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shape_specs import GROUPS, specs_for, units

# Small budgets make the deep shapes narrow, so a preference measured there could be the
# bottleneck rather than the shape. Only the budgets where that is not true get a vote.
TRUSTED_BUDGETS = ("256", "1024")

# Step 2 sweeps further up than the probe did, to test the other half of the claim: that
# a large enough early-fusion model finds the structure late fusion is handed.
BUDGETS = (128, 256, 512, 1024, 2048)


def main() -> None:
    """Write assets/tower-depths.env from assets/shape-probes.json."""
    results = json.loads(Path("assets/shape-probes.json").read_text(encoding="utf-8"))
    depths: dict[str, int] = {}
    for group in GROUPS:
        votes: dict[int, list[float]] = {}
        spreads: list[float] = []
        for budget in TRUSTED_BUDGETS:
            row = results.get(budget, {}).get(group)
            if not row:
                continue
            medians = {label: statistics.median(v) for label, v in row.items()}
            spreads += [max(v) - min(v) for v in row.values()]
            best = max(medians, key=lambda label: medians[label])
            margin = medians[best] - sorted(medians.values())[-2]
            if margin > max(max(v) - min(v) for v in row.values()) / 2:
                votes.setdefault(int(best.split("x")[0]), []).append(margin)
        depths[group] = max(votes, key=lambda d: sum(votes[d])) if votes else 1
        note = "probe" if votes else "not separated, default"
        print(f"  {group:<15} depth {depths[group]}   ({note})")

    # When the probe separates nothing, every tower is depth 1 and `late-tuned` is
    # `late-uniform` under another name. Running both would spend two hours measuring one
    # model against itself, so the runner is told to skip it and the budget goes into the
    # question that is still open: whether a large enough early-fusion model closes the
    # gap on its own.
    degenerate = set(depths.values()) == {1}
    if degenerate:
        print("\n  no input separated a shape: late-tuned == late-uniform, skipping it")

    lines = [f"TUNED_IS_DEGENERATE={1 if degenerate else 0}"]
    for budget in BUDGETS:
        for arch, spec in specs_for(budget, depths).items():
            assert units(spec) == budget, (arch, budget, units(spec))
            key = arch.replace("-", "_")
            lines.append(f'{key}_{budget}_widths="{spec["AI_RESIDENCY_HIDDEN_WIDTHS"]}"')
            lines.append(f'{key}_{budget}_towers="{spec["AI_RESIDENCY_TOWER_SHAPES"]}"')
            lines.append(f'{key}_{budget}_fuse="{spec["AI_RESIDENCY_FUSION_LAYER"]}"')
    Path("assets/tower-depths.env").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote assets/tower-depths.env ({len(lines)} variables)")


if __name__ == "__main__":
    main()
