"""Drawing the negative words that make negative sampling work.

Which distribution you draw from matters more than it looks. Uniform sampling wastes
almost every negative on a rare word the model already scores near zero; sampling
straight from the unigram frequencies spends almost all of them on "the". Mikolov et
al. (2013) raise the frequencies to the 3/4 power, which sits between the two: common
words still dominate, but not by the margin their raw counts imply.
"""

import torch

__all__ = ["NegativeSampler", "unigram_distribution"]

SAMPLING_POWER = 0.75
"""The 3/4 exponent from Mikolov et al. (2013), found empirically in that paper."""


def unigram_distribution(counts: torch.Tensor, power: float = SAMPLING_POWER) -> torch.Tensor:
    """Turn word counts into the distribution negatives are drawn from.

    Args:
        counts: Occurrences per word id, `(vocab_size,)`. Zero-count entries stay at
            zero probability, so padding and unused ids are never sampled.
        power: Exponent applied to each count. 1.0 is the raw unigram distribution,
            0.0 is uniform, and the default 0.75 is the usual compromise.

    Returns:
        A `(vocab_size,)` probability vector summing to 1.

    Raises:
        ValueError: If counts are negative, or sum to zero.
    """
    if torch.any(counts < 0):
        raise ValueError("counts cannot be negative")
    weighted = counts.double().pow(power)
    total = weighted.sum()
    if total <= 0:
        raise ValueError("counts sum to zero; nothing to sample from")
    return (weighted / total).float()


class NegativeSampler:
    """Draws negative word ids, in batches, from a fixed distribution.

    Holds the distribution rather than recomputing it: it depends only on corpus
    counts, so building it once and sampling many times is the whole point.

    Attributes:
        probabilities: The `(vocab_size,)` distribution being sampled from.
    """

    def __init__(self, counts: torch.Tensor, power: float = SAMPLING_POWER) -> None:
        """Build a sampler from corpus word counts.

        Args:
            counts: Occurrences per word id, `(vocab_size,)`.
            power: Exponent applied to the counts. See `unigram_distribution`.
        """
        self.probabilities = unigram_distribution(counts, power)

    def __call__(
        self,
        batch_size: int,
        num_negatives: int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Draw negatives for one batch.

        Sampled with replacement, and without excluding the true target. Both are
        deliberate and match the reference implementation: the check costs more than
        the occasional collision, and at any realistic vocabulary size a collision
        is rare enough to be noise in the gradient.

        Args:
            batch_size: Rows to draw for.
            num_negatives: Negatives per row.
            generator: Optional RNG, for reproducible runs.
            device: Where to put the result. Defaults to the distribution's device.

        Returns:
            Word ids of shape `(batch_size, num_negatives)`.

        Raises:
            ValueError: If either dimension is not positive.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if num_negatives < 1:
            raise ValueError(f"num_negatives must be positive, got {num_negatives}")

        drawn = torch.multinomial(
            self.probabilities,
            batch_size * num_negatives,
            replacement=True,
            generator=generator,
        )
        reshaped = drawn.view(batch_size, num_negatives)
        return reshaped.to(device) if device is not None else reshaped
