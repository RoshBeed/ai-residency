"""One resolution and one palette for every figure in the portfolio.

Nine services draw charts and every one of them used to carry its own `savefig` call
with its own `dpi=`. That produced assets at 110, 140, 150, 200, 300 and 600 DPI across
the write-ups, so a reader opening two figures from the same README got two different
physical sizes for the same nominal width. The number is not interesting; having one of
it is, which makes it a constant rather than a keyword argument repeated at fourteen
call sites.

`DPI = 200` because it is what the two trained services' figures already used and what
renders sharply on a retina display at the width a README column gives them. A figure
declared at 8x4 inches comes out 1600x800 -- wide enough to stay legible when GitHub
scales it down, small enough not to add a megabyte to the repository per chart.

Importing this module needs matplotlib, which the serving images deliberately do not
install: it is the `figures` extra rather than a hard dependency, and
`ai_residency_common/__init__.py` does not re-export it, so nothing pulls matplotlib
into an API container by importing the package.
"""

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

__all__ = [
    "AQUA",
    "BLUE",
    "DPI",
    "INK",
    "INK_SOFT",
    "MUTED",
    "ORANGE",
    "RULE",
    "SERIES",
    "SURFACE",
    "VIOLET",
    "apply_style",
    "save_figure",
]

# Every figure in the portfolio is written at this resolution. See the module docstring
# for why it is one number, and `deploy/test_workspace.py` for the test that keeps the
# committed assets agreeing with it.
DPI = 200

# The reference categorical set, in assignment order. Validated on the all-pairs list
# rather than only on adjacent pairs, which is what a scatter needs -- a bar chart only
# ever puts neighbours side by side, a scatter can put any two colours together. Aqua
# sits below 3:1 against the surface, so a chart using it carries direct value labels.
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
SERIES = (BLUE, ORANGE, AQUA, VIOLET)

INK, INK_SOFT, MUTED, RULE = "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
SURFACE = "#fcfcfb"


def apply_style() -> None:
    """Recessive axes and grid, so the data is the only assertive thing on the page."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": RULE,
            "axes.labelcolor": INK_SOFT,
            "axes.titlecolor": INK,
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": RULE,
            "grid.alpha": 0.4,
            "grid.linewidth": 0.6,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "font.size": 10,
            "legend.frameon": False,
        }
    )


def save_figure(figure: Figure, output: Path | str, *, tight: bool = True, **kwargs: Any) -> None:
    """Write a figure at the portfolio's resolution.

    Args:
        figure: The figure to write.
        output: Where to write it. Parent directories are created.
        tight: Crop to the drawn content. Off for a figure whose own layout already
            positions the axes -- cropping one of those moves the panels around.
        **kwargs: Passed through to `savefig`.
    """
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    if tight:
        kwargs.setdefault("bbox_inches", "tight")
    kwargs.setdefault("facecolor", SURFACE)
    figure.savefig(output, dpi=DPI, **kwargs)
