"""Turning an encoded corpus into training batches.

This replaces a `torch.utils.data.Dataset` that materialised every training example
as a Python tuple in `__init__`. At text8's 16.7M tokens that is 16.7M tuples of an
8-element list, built before the first optimizer step and held for the whole run --
several GB of Python objects, and the reason the Makefile warned that this stage takes
hours. The arithmetic was never the expensive part.

What replaces it is a batcher that keeps the corpus as one `int64` tensor and cuts
batches out of it with tensor operations. `Tensor.unfold` gives every sliding window
as a *view*, so the window matrix costs nothing to build, and a permutation of its
rows is the shuffle.

Two things the old dataset left out, both of which matter more for embedding quality
than any hyperparameter in the sweep:

**Subsampling.** Mikolov et al. (2013b) discard frequent words with probability rising
in their frequency, which both speeds training and improves the vectors for rare
words. Without it most of the gradient goes into `the`. It is re-drawn every epoch, so
a word dropped in one epoch is usually kept in another.

**The unigram^0.75 negative distribution.** The old dataset drew negatives uniformly,
so nearly every negative was a rare word the model already scored near zero and the
update was wasted. `libs/word2vec` already had `NegativeSampler`; this uses it.
"""

from collections.abc import Iterator

import torch

from ai_residency_common import get_logger
from config import Config
from word2vec import NegativeSampler
from word_embeddings.corpus import Corpus

__all__ = ["Batch", "Word2VecBatcher", "keep_probabilities"]

logger = get_logger(__name__)

Batch = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def keep_probabilities(counts: torch.Tensor, threshold: float) -> torch.Tensor:
    """Probability of keeping each word, by Mikolov's frequent-word discard rule.

    `P(keep) = (sqrt(z/t) + 1) * (t/z)`, where `z` is the word's corpus frequency and
    `t` the threshold. Words at or below `t` are always kept; above it the probability
    falls off, so `the` at ~6% frequency survives roughly one occurrence in forty at
    `t=1e-4`.

    Args:
        counts: Occurrences per word id, `(vocab_size,)`.
        threshold: The `t` above. Typically `1e-3` to `1e-5`.

    Returns:
        A `(vocab_size,)` float tensor of probabilities, clamped to at most 1.

    Raises:
        ValueError: If `threshold` is not positive, which would divide by zero and
            silently produce a corpus of NaNs rather than an error.
    """
    if threshold <= 0:
        raise ValueError(f"threshold must be positive, got {threshold}")
    frequency = counts.double() / counts.sum()
    probability = ((frequency / threshold).sqrt() + 1) * (threshold / frequency)
    return probability.clamp(max=1.0).float()


class Word2VecBatcher:
    """Yields `(centre, context, negatives)` batches over a corpus.

    Attributes:
        window_size: Context words taken either side of the centre.
        batch_size: Windows per batch.
        num_negatives: Negatives drawn per row.
    """

    def __init__(
        self,
        corpus: Corpus,
        config: Config,
        *,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> None:
        """Prepare the sampler and subsampling table for a corpus.

        Args:
            corpus: The encoded corpus and its counts.
            config: Supplies window, batch size, negatives and the threshold.
            device: Where batches are delivered.
            generator: Optional RNG, for reproducible runs.
        """
        self._ids = corpus.ids
        self._device = device
        self._generator = generator
        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.num_negatives = config.num_negatives

        self._keep = keep_probabilities(corpus.counts, config.subsample_threshold)
        self._sampler = NegativeSampler(corpus.counts)

        # Negatives are drawn *per positive pair*, which for skip-gram means
        # `num_negatives` for each of the 2*window context words rather than
        # `num_negatives` for the row. The shared loss sums logsigmoid over every
        # positive and every negative, so drawing one row's worth would leave
        # skip-gram with ten positives pulling up against five negatives pushing
        # down -- a ratio CBOW does not have, and one that lets the model raise
        # every score at once. It is not a knob; it is what makes the two
        # architectures comparable at the same `num_negatives`.
        context_size = 2 * config.window_size
        self._draws = self.num_negatives * (
            context_size if config.architecture == "skipgram" else 1
        )

        expected = float(self._keep[self._ids].sum())
        logger.info(
            "Subsampling at t=%g keeps about %.1fM of %.1fM tokens per epoch",
            config.subsample_threshold,
            expected / 1e6,
            len(corpus) / 1e6,
        )
        self._steps = max(1, int(expected - 2 * self.window_size) // self.batch_size)

    def __len__(self) -> int:
        """Approximate batches per epoch.

        Approximate because subsampling is re-drawn each epoch, so the exact count
        varies by a fraction of a percent. Used for progress bars and for projecting
        how long a run will take, neither of which needs it exact.
        """
        return self._steps

    def _subsampled(self) -> torch.Tensor:
        """Draw one epoch's corpus, with frequent words thinned out."""
        draws = torch.rand(self._ids.shape, generator=self._generator)
        return self._ids[draws < self._keep[self._ids]]

    def epoch(self) -> Iterator[Batch]:
        """Yield one epoch of shuffled batches.

        Yields:
            `(centre, context, negatives)` with shapes `(B,)`, `(B, 2*window)` and
            `(B, draws)`, on the batcher's device -- where `draws` is
            `num_negatives` for CBOW and `num_negatives * 2 * window` for skip-gram.
        """
        kept = self._subsampled()
        span = 2 * self.window_size + 1
        if kept.numel() < span:
            return

        # A view, not a copy: `windows[i]` is the slice of `kept` starting at i.
        windows = kept.unfold(0, span, 1)
        order = torch.randperm(windows.shape[0], generator=self._generator)

        for start in range(0, order.numel() - self.batch_size + 1, self.batch_size):
            rows = windows[order[start : start + self.batch_size]]
            centre = rows[:, self.window_size]
            context = torch.cat(
                (rows[:, : self.window_size], rows[:, self.window_size + 1 :]), dim=1
            )
            negatives = self._sampler(rows.shape[0], self._draws, generator=self._generator)
            yield (
                centre.to(self._device),
                context.to(self._device),
                negatives.to(self._device),
            )
