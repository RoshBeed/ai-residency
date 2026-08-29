"""Fetching text8 and turning it into a vocabulary and a token-id array.

text8 is the first 100MB of a 2006 English Wikipedia dump, lowercased and stripped to
`[a-z ]`. It is the standard word2vec benchmark corpus, which is the reason to train
on it: published numbers for WordSim-353 and the Google analogy set are quoted against
this exact corpus, so a score here is comparable to something rather than only to
itself.

Two decisions here matter more than they look.

**Ids are assigned in descending frequency order.** Nothing requires that, but it
makes "the 300 most frequent words" the slice `[:300]`, which is what every plot and
probe in this service actually wants. The previous pipeline assigned ids in
first-appearance order, so figures captioned "top 100 by frequency" were showing the
first 100 words the tokenizer happened to see.

**Words below `min_count` are dropped, not mapped to `<UNK>`.** A single `<UNK>` row
standing in for every rare word becomes one of the most frequent tokens in the corpus,
and it appears in the context of everything, so it trains into a vector near the
centre of the space and drags its neighbours there. The reference implementation drops
them, and so does this.
"""

from collections import Counter

import torch

from ai_residency_common import get_logger, resolve_checkpoint
from config import Config

__all__ = ["Corpus", "build_vocabulary", "encode", "load_corpus", "load_text8"]

logger = get_logger(__name__)


class Corpus(torch.utils.data.Dataset[int]):
    """An encoded corpus and the vocabulary it is indexed by.

    Attributes:
        ids: Token ids, one per surviving corpus word, in corpus order.
        word_to_index: The vocabulary. Ids are dense, `0..len-1`, most frequent first.
        counts: Occurrences per id, `(vocab_size,)`, for the negative sampler and the
            subsampling probabilities.
    """

    def __init__(
        self,
        ids: torch.Tensor,
        word_to_index: dict[str, int],
        counts: torch.Tensor,
    ) -> None:
        self.ids = ids
        self.word_to_index = word_to_index
        self.counts = counts

    @property
    def vocab_size(self) -> int:
        """Number of words in the vocabulary, which is also the embedding row count."""
        return len(self.word_to_index)

    @property
    def index_to_word(self) -> dict[int, str]:
        """The inverse vocabulary. Total, because ids are dense and unique."""
        return {index: word for word, index in self.word_to_index.items()}

    def __len__(self) -> int:
        return int(self.ids.numel())

    def __getitem__(self, index: int) -> int:
        return int(self.ids[index])


def load_text8(config: Config) -> str:
    """Return the text8 corpus, downloading it from the Hub if it is not local.

    Args:
        config: Supplies `text8_file` and `text8_repo_id`.

    Returns:
        The corpus as one whitespace-separated string.

    Raises:
        FileNotFoundError: If the corpus is neither local nor fetchable.
    """
    path, source = resolve_checkpoint(config.text8_file, config.text8_repo_id, repo_type="dataset")
    if path is None:
        raise FileNotFoundError(
            f"text8 is not at {config.text8_file} and could not be fetched from "
            f"{config.text8_repo_id}"
        )
    logger.info("Reading text8 from %s (%s)", path, source)
    return path.read_text(encoding="utf-8")


def build_vocabulary(words: list[str], min_count: int) -> tuple[dict[str, int], torch.Tensor]:
    """Build a frequency-ordered vocabulary over `words`.

    Args:
        words: The corpus, tokenized.
        min_count: Minimum occurrences for a word to be kept.

    Returns:
        The word-to-index mapping, ids assigned most-frequent-first, and a
        `(vocab_size,)` tensor of counts aligned to those ids.

    Raises:
        ValueError: If `min_count` removes every word, which means the threshold and
            the corpus do not belong together rather than that the corpus is empty.
    """
    frequencies = Counter(words)
    kept = [(word, count) for word, count in frequencies.most_common() if count >= min_count]
    if not kept:
        raise ValueError(f"min_count={min_count} removed all {len(frequencies)} distinct words")

    word_to_index = {word: index for index, (word, _) in enumerate(kept)}
    counts = torch.tensor([count for _, count in kept], dtype=torch.long)
    logger.info(
        "Vocabulary holds %d of %d distinct words at min_count=%d (%.1f%% of tokens kept)",
        len(word_to_index),
        len(frequencies),
        min_count,
        100.0 * int(counts.sum()) / max(len(words), 1),
    )
    return word_to_index, counts


def encode(words: list[str], word_to_index: dict[str, int]) -> torch.Tensor:
    """Encode a tokenized corpus, dropping words the vocabulary does not hold.

    Args:
        words: The corpus, tokenized.
        word_to_index: Vocabulary to encode against.

    Returns:
        A `(n_tokens,)` long tensor of ids, shorter than `words` by the number of
        below-threshold words removed.
    """
    ids = [word_to_index[word] for word in words if word in word_to_index]
    logger.info("Encoded %d tokens from %d corpus words", len(ids), len(words))
    return torch.tensor(ids, dtype=torch.long)


def load_corpus(config: Config) -> Corpus:
    """Fetch text8, build its vocabulary and encode it.

    Args:
        config: Supplies the corpus location and `min_count`.

    Returns:
        The encoded corpus with its vocabulary and counts.
    """
    words = load_text8(config).split()
    word_to_index, counts = build_vocabulary(words, config.min_count)
    return Corpus(encode(words, word_to_index), word_to_index, counts)
