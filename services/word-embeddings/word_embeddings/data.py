"""Vocabulary and corpus loading for the CBOW stage.

Split out of the training function, which used to do this inline along with model
construction, the epoch loop, evaluation, checkpointing and the sweep. Reading the
corpus is a separate concern from training on it, and keeping it separate is what
lets the tests exercise a tiny vocabulary without touching the real 100M-token file.
"""

import json
import random

import torch
from torch.utils.data import Dataset

from ai_residency_common import get_logger
from config import Config

__all__ = [
    "UNKNOWN_TOKEN",
    "CBOWNegativeSamplingDataset",
    "encode_corpus",
    "load_corpus",
    "load_vocabulary",
    "sample_vocabulary",
]

logger = get_logger(__name__)

UNKNOWN_TOKEN = "<UNK>"

# A handful of words with obvious relationships, used by the sample vocabulary and by
# the analogy check in `evaluation.py`. Kept here so the two cannot drift apart: the
# evaluation used to assume "king"/"queen"/"man"/"woman" existed in whatever
# vocabulary it was handed, which is only guaranteed for this one.
_SAMPLE_WORDS = [
    "hi",
    "my",
    "name",
    "is",
    "kernel",
    "and",
    "i",
    "like",
    "to",
    "code",
    "python",
    "machine",
    "learning",
    "is",
    "fun",
    "king",
    "queen",
    "man",
    "woman",
    "dog",
    "cat",
]

_SAMPLE_CORPUS = (
    "python machine learning is fun and i like to code python is a great "
    "language for machine learning and coding is fun with python "
    "king queen man woman king queen man woman"
)


def _with_unknown(words: list[str]) -> dict[str, int]:
    """Build a word-to-index mapping with `UNKNOWN_TOKEN` appended last.

    Args:
        words: Vocabulary words, in any order; duplicates are collapsed.

    Returns:
        The mapping, including an entry for `UNKNOWN_TOKEN`.
    """
    unique = sorted(set(words))
    mapping = {word: index for index, word in enumerate(unique)}
    mapping[UNKNOWN_TOKEN] = len(unique)
    return mapping


class CBOWNegativeSamplingDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Windowed context/target pairs with sampled negatives, for one CBOW epoch.

    For every token, the surrounding `window_size` words on each side form the
    context and the token itself the target. `num_negatives` random words are drawn
    per sample as the negative examples the model scores against.

    Args:
        data: The encoded corpus, one token id per word.
        window_size: Number of context words on each side of the target.
        num_negatives: Negative samples per (context, target) pair.
    """

    def __init__(self, data: list[int], window_size: int, num_negatives: int) -> None:
        self.samples = []
        self.vocab_size = len(set(data))
        self.num_negatives = num_negatives
        for i in range(window_size, len(data) - window_size):
            context = data[i - window_size : i] + data[i + 1 : i + window_size + 1]
            target = data[i]
            self.samples.append((context, target))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context, target = self.samples[idx]
        negatives: list[int] = []
        while len(negatives) < self.num_negatives:
            neg = random.randint(0, self.vocab_size - 1)
            if neg != target:
                negatives.append(neg)
        return torch.tensor(context), torch.tensor(target), torch.tensor(negatives)

    def __len__(self) -> int:
        return len(self.samples)


def sample_vocabulary() -> tuple[dict[str, int], dict[str, int], str]:
    """Return a tiny vocabulary and corpus, for smoke runs and tests.

    Returns:
        The word-to-index mapping, the word-to-lemma-index mapping, and the corpus
        text. The two mappings are identical here; on the real corpus they differ.
    """
    mapping = _with_unknown(_SAMPLE_WORDS)
    return mapping, dict(mapping), _SAMPLE_CORPUS


def load_vocabulary(config: Config) -> tuple[dict[str, int], dict[str, int]]:
    """Load the vocabularies the ETL stage produced.

    Args:
        config: Supplies `word_index_file` and `vocab_file`.

    Returns:
        The word-to-index and word-to-lemma-index mappings, each guaranteed to
        contain `UNKNOWN_TOKEN`.

    Raises:
        FileNotFoundError: If either mapping is missing; run the corpus stage first.
    """
    for path in (config.word_index_file, config.vocab_file):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Build the corpus first with `make train-corpus`."
            )

    with config.word_index_file.open(encoding="utf-8") as handle:
        word_to_index: dict[str, int] = json.load(handle)
    with config.vocab_file.open(encoding="utf-8") as handle:
        word_to_lemma_index: dict[str, int] = json.load(handle)

    if UNKNOWN_TOKEN not in word_to_lemma_index:
        unknown = len(word_to_lemma_index)
        word_to_lemma_index[UNKNOWN_TOKEN] = unknown
        word_to_index[UNKNOWN_TOKEN] = unknown

    logger.info("Vocabulary holds %d words", len(word_to_lemma_index))
    return word_to_index, word_to_lemma_index


def encode_corpus(text: str, word_to_index: dict[str, int]) -> list[int]:
    """Turn corpus text into token ids, mapping unseen words to `UNKNOWN_TOKEN`.

    Args:
        text: Raw corpus text.
        word_to_index: Mapping to encode against.

    Returns:
        One token id per whitespace-separated word.
    """
    unknown = word_to_index[UNKNOWN_TOKEN]
    encoded = [word_to_index.get(word, unknown) for word in text.lower().split()]
    logger.info("Encoded %d tokens", len(encoded))
    return encoded


def load_corpus(config: Config) -> str:
    """Read the merged corpus text.

    Args:
        config: Supplies `corpus_file`.

    Returns:
        The corpus as one string.

    Raises:
        FileNotFoundError: If the corpus is missing; run the corpus stage first.
    """
    if not config.corpus_file.exists():
        raise FileNotFoundError(
            f"{config.corpus_file} not found. Build the corpus first with `make train-corpus`."
        )
    return config.corpus_file.read_text(encoding="utf-8")
