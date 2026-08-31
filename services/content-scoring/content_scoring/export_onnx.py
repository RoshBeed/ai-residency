"""Export each architecture to ONNX, for inspection in Netron.

The four models differ in ways a parameter count only hints at: `late` runs five separate
towers and joins them in a second layer, `block_diagonal` is the same first layer as
`early` with the cross-group weights removed. Both are much easier to see than to read
about, and an exported graph is checkable in a way a hand-drawn diagram is not -- it comes
from the model rather than from someone's memory of it.

    make -C services/content-scoring onnx
    netron services/content-scoring/assets/onnx/late.onnx

Exports the *architecture*, not a trained checkpoint: the shapes are the point, and a
randomly initialised model has the same graph. That also keeps this runnable before
anything has been trained.
"""

import argparse
from pathlib import Path

import torch

from ai_residency_common import get_logger, load_checkpoint, resolve_checkpoint
from config import Config, Fusion
from content_scoring.features import COLUMN_ORDER
from content_scoring.model import build_model

__all__ = ["export"]

logger = get_logger(__name__)

ASSETS = Path(__file__).resolve().parent.parent / "assets" / "onnx"

# Small enough that Netron renders the whole graph legibly. The real vocabulary is 86,438
# rows and the real domain table 1,003; both appear as a single Gather node either way,
# and a 104MB initialiser makes the file slow to open for no extra information.
PREVIEW_VOCAB = 64
PREVIEW_DOMAINS = 8


def export(fusion: Fusion, output: Path, *, real_shapes: bool = False) -> Path:
    """Write one architecture to ONNX.

    Args:
        fusion: Which architecture to export.
        output: Destination path.
        real_shapes: Use the true vocabulary and domain-table sizes. Off by default --
            it adds 104MB of embedding weights that Netron draws as the same one node.

    Returns:
        The path written.
    """
    config = Config(fusion=fusion)
    vocab, domains = (PREVIEW_VOCAB, PREVIEW_DOMAINS)
    if real_shapes:
        path, _ = resolve_checkpoint(config.embeddings_path, config.hf_repo_id)
        if path is not None:
            extra = load_checkpoint(path)["extra"]
            vocab, domains = len(extra["word_to_index"]), 1003

    model = build_model(torch.randn(vocab, 300), domains, config).eval()
    example = (
        torch.randint(0, vocab, (1, config.max_title_tokens)),
        torch.ones(1, config.max_title_tokens),
        torch.zeros(1, len(COLUMN_ORDER)),
        torch.zeros(1, dtype=torch.long),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        example,
        str(output),
        input_names=["title_tokens", "title_mask", "features", "domain_id"],
        output_names=["expected_upvotes"],
        dynamo=False,
    )
    logger.info("%s -> %s", fusion, output)
    return output


def main() -> None:
    """Export every architecture."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-shapes", action="store_true", help="Use the true table sizes.")
    arguments = parser.parse_args()

    for fusion in ("text_only", "features_only", "early", "block_diagonal", "late"):
        export(fusion, ASSETS / f"{fusion}.onnx", real_shapes=arguments.real_shapes)
    logger.info("Open one with: netron %s", ASSETS / "late.onnx")


if __name__ == "__main__":
    main()
