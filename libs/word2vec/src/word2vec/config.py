"""What a word2vec model is built from.

Separate from `model.py` so a service's own `Config` can hold one of these without
importing torch, and so the architecture/objective pair has one place to be
validated rather than being checked at every use site.
"""

from dataclasses import dataclass
from typing import get_args

from word2vec.model_types import Architecture, Objective

__all__ = ["Word2VecConfig"]


@dataclass(frozen=True)
class Word2VecConfig:
    """Dimensions and the two choices that define a word2vec variant.

    Attributes:
        vocab_size: Number of tokens the model can represent.
        embed_size: Width of each word vector.
        architecture: `"cbow"` averages the context to predict the centre word;
            `"skipgram"` uses the centre word to predict each context word.
            Skip-gram is slower per example and generally better on rare words.
        objective: `"negative_sampling"` scores the target against a handful of
            sampled words; `"softmax"` scores it against the entire vocabulary.
            Softmax is exact and O(vocab_size), which is what negative sampling
            exists to avoid -- prefer it only for small vocabularies or for
            checking that a negative-sampling result is not an artefact.
        num_negatives: Negatives drawn per positive. Ignored by the softmax
            objective. Mikolov et al. suggest 5-20 for small corpora, 2-5 for
            large ones.
        window_size: Words of context taken either side of the centre word. Used
            by dataset construction rather than by the model itself, but it belongs
            with the other hyperparameters that define a run.
    """

    vocab_size: int
    embed_size: int = 128
    architecture: Architecture = "cbow"
    objective: Objective = "negative_sampling"
    num_negatives: int = 5
    window_size: int = 4

    def __post_init__(self) -> None:
        """Reject impossible combinations at construction rather than at first batch.

        Raises:
            ValueError: If a field is out of range or names an unknown variant.
        """
        if self.vocab_size < 1:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.embed_size < 1:
            raise ValueError(f"embed_size must be positive, got {self.embed_size}")
        if self.architecture not in get_args(Architecture):
            raise ValueError(
                f"unknown architecture {self.architecture!r}; "
                f"expected one of {get_args(Architecture)}"
            )
        if self.objective not in get_args(Objective):
            raise ValueError(
                f"unknown objective {self.objective!r}; expected one of {get_args(Objective)}"
            )
        if self.objective == "negative_sampling" and self.num_negatives < 1:
            raise ValueError(
                f"negative_sampling needs at least one negative, got {self.num_negatives}"
            )
        if self.window_size < 1:
            raise ValueError(f"window_size must be positive, got {self.window_size}")
