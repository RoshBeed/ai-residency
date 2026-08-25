"""The two axes a word2vec variant is chosen along.

Their own module because `config.py` needs them to type its fields and `model.py`
needs them to branch on, and having either import the other would be circular.
"""

from typing import Literal

__all__ = ["Architecture", "Objective"]

Architecture = Literal["cbow", "skipgram"]
"""How context and centre relate: average the context, or predict from the centre."""

Objective = Literal["negative_sampling", "softmax"]
"""How a score becomes a loss: a few sampled words, or the whole vocabulary."""
