"""Render the architecture diagram the demo page draws, outside the demo page.

    make -C services/content-scoring graph            # every architecture, one PNG
    make -C services/content-scoring graph FORMAT=svg
    make -C services/content-scoring graph FUSION=late

The page builds this from the running models on each `/compare`, so there is no file to
open when the stack is down -- and no file to go stale, which is the point. This writes
one on demand from the same function, so the diagram in a slide is the diagram the
service would draw.

Writes the DOT unconditionally and renders it only if Graphviz is installed, because the
`dot` binary is a system package rather than something `uv sync` can supply. `brew
install graphviz` on a Mac.

Like `make onnx`, this exports the *architecture*: shapes are the point, and a randomly
initialised model has the same graph, so it runs before anything is trained.
"""

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import get_args

import torch

from ai_residency_common import get_logger
from config import Config, Fusion
from content_scoring.model import build_model, graph_dot, graph_dot_all

logger = get_logger(__name__)

# Only layer widths are drawn, so a stub matrix of the right width is enough -- the real
# one is 86,438 x 300 and would be a 104MB download to read a number off.
EMBED_SIZE = 300
ASSETS = Path(__file__).resolve().parent.parent / "assets" / "graphs"


def main() -> None:
    """Write the architecture diagram as DOT, and as an image if Graphviz is present."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fusion",
        choices=get_args(Fusion),
        help="One architecture. Omitted, every architecture goes in one diagram.",
    )
    parser.add_argument("--format", default="png", help="Graphviz output format.")
    parser.add_argument(
        "--rankdir",
        default="BT",
        help="Graphviz direction: BT puts the inputs at the bottom, LR at the left.",
    )
    arguments = parser.parse_args()

    stub = torch.zeros(64, EMBED_SIZE)
    if arguments.fusion:
        name = arguments.fusion
        config = Config(fusion=arguments.fusion)
        dot = graph_dot(build_model(stub, 8, config), rankdir=arguments.rankdir)
    else:
        name = "architectures"
        models = {
            fusion: build_model(stub, 8, Config(fusion=fusion)) for fusion in get_args(Fusion)
        }
        dot = graph_dot_all(models, rankdir=arguments.rankdir)

    ASSETS.mkdir(parents=True, exist_ok=True)
    source = ASSETS / f"{name}.dot"
    source.write_text(dot, encoding="utf-8")
    logger.info("Wrote %s", source)

    if shutil.which("dot") is None:
        logger.warning("Graphviz is not installed, so only the DOT was written.")
        logger.warning("Install it with `brew install graphviz`, or paste the DOT into")
        logger.warning("https://dreampuf.github.io/GraphvizOnline to view it.")
        return

    output = ASSETS / f"{name}.{arguments.format}"
    # -Gdpi matches the portfolio's figure resolution, so this drops into a write-up
    # beside the matplotlib figures without being a different size.
    subprocess.run(
        ["dot", f"-T{arguments.format}", "-Gdpi=200", str(source), "-o", str(output)],
        check=True,
    )
    logger.info("Wrote %s", output)
    logger.info("Open it with: open %s", output)


if __name__ == "__main__":
    main()
