"""Loading checkpoints written before this module existed.

**This module is a bridge, and is meant to be deleted.** The checkpoints currently on
the Hub are end-of-week results that will be replaced by proper training runs; nothing
here is a promise about their format. What it buys is that the repo keeps working
*between* the code being unified and the models being retrained, rather than every
service 503ing until someone finds a GPU. Once each service has a checkpoint written
by this module, delete `compat.py` and its tests.

Every one of the old implementations holds the same
two matrices and differs only in what it calls them:

    this module          in_embed.weight    out_embed.weight
    word-embeddings      in_embed.weight    out_embed.weight     (identical)
    vector-search        emb.weight         ffw.weight

`ffw` is `nn.Linear(embed_size, vocab_size, bias=False)`, whose weight is
`(vocab_size, embed_size)` -- the same shape, and the same role, as an output
embedding matrix. So the difference is naming, and a rename is lossless.

What does *not* transfer is the objective: a checkpoint trained with full softmax was
fitted under a different loss than one trained with negative sampling. The vectors are
still the vectors, which is all anything downstream reads, but do not resume training
from one under the other objective and expect the loss curve to continue.
"""

from collections.abc import Mapping

import torch

__all__ = ["LEGACY_KEY_MAPS", "state_dict_from_legacy"]

LEGACY_KEY_MAPS: dict[str, dict[str, str]] = {
    # services/word-embeddings -- CBOWNS, a PreTrainedModel. Already agrees.
    "word-embeddings": {
        "in_embed.weight": "in_embed.weight",
        "out_embed.weight": "out_embed.weight",
    },
    # services/vector-search-retrieval -- CBOW with a full-softmax projection.
    "vector-search-retrieval": {
        "emb.weight": "in_embed.weight",
        "ffw.weight": "out_embed.weight",
    },
}
"""Old parameter name -> new one, per implementation that wrote checkpoints."""


def state_dict_from_legacy(
    state_dict: Mapping[str, torch.Tensor],
    layout: str,
    *,
    strict: bool = True,
) -> dict[str, torch.Tensor]:
    """Rename a pre-unification state dict so `Word2Vec` can load it.

    Args:
        state_dict: The saved parameters, as written by one of the old
            implementations. A checkpoint wrapping these under `model_state_dict`
            should be unwrapped by the caller first.
        layout: Which implementation wrote it -- a key of `LEGACY_KEY_MAPS`.
        strict: Whether a key the map does not mention is an error. True is right
            for a migration, where an unexpected key means the layout guess was
            wrong. False drops such keys, which is occasionally useful for a
            checkpoint carrying optimizer state alongside the model.

    Returns:
        A state dict keyed the way `Word2Vec` expects.

    Raises:
        KeyError: If `layout` is not a known implementation.
        ValueError: If a mapped key is missing, or -- under `strict` -- an
            unexpected key is present.
    """
    if layout not in LEGACY_KEY_MAPS:
        raise KeyError(f"unknown layout {layout!r}; expected one of {sorted(LEGACY_KEY_MAPS)}")

    key_map = LEGACY_KEY_MAPS[layout]

    missing = [old for old in key_map if old not in state_dict]
    if missing:
        raise ValueError(
            f"checkpoint does not look like {layout!r}: missing {missing}. "
            f"It has {sorted(state_dict)}"
        )

    if strict:
        unexpected = [key for key in state_dict if key not in key_map]
        if unexpected:
            raise ValueError(
                f"unexpected keys for layout {layout!r}: {unexpected}. "
                f"Pass strict=False to drop them."
            )

    return {new: state_dict[old] for old, new in key_map.items()}
