"""CBOW training dataset."""

import pickle

import torch
from torch.utils.data import Dataset

from cbow.vocabulary import (
    load_ids_to_words,
    load_words_to_ids,
)
from config import Config

__all__ = ["Example", "Wiki"]

# What this dataset yields: a context window and the centre word it predicts.
Example = tuple[torch.Tensor, torch.Tensor]


class Wiki(Dataset[Example]):
    """The preprocessed `text8` corpus as CBOW (context, target) pairs.

    Attributes:
        words_to_ids: Vocabulary mapping word to token id.
        ids_to_words: Vocabulary mapping token id to word.
        tokens: The whole corpus as a flat list of token ids.
        context_size: Words of context taken from each side of the target.
    """

    def __init__(self, config: Config) -> None:
        """Load the corpus and vocabulary from disk.

        Args:
            config: Supplies `corpus_path`, the vocabulary paths and `context_size`.
        """
        self.words_to_ids = load_words_to_ids(config)
        self.ids_to_words = load_ids_to_words(config)
        with config.corpus_path.open("rb") as handle:
            corpus: list[str] = pickle.load(handle)
        self.tokens = [self.words_to_ids[word] for word in corpus]
        self.context_size = config.context_size

    def __len__(self) -> int:
        """Number of training examples, one per token in the corpus."""
        return len(self.tokens)

    def __getitem__(self, index: int) -> Example:
        """Return the context window around `index` and the word at it.

        Windows at the very start or end of the corpus are left-padded or
        right-padded with token 0, which the vocabulary reserves for `<PAD>`.

        Args:
            index: Position of the target word in the corpus.

        Returns:
            A `(context, target)` pair of tensors.
        """
        size = self.context_size
        target = self.tokens[index]
        previous = self.tokens[max(index - size, 0) : index]
        following = self.tokens[index + 1 : index + 1 + size]
        if len(previous) < size:
            previous = [0] * (size - len(previous)) + previous
        if len(following) < size:
            following = following + [0] * (size - len(following))
        return torch.tensor(previous + following), torch.tensor([target])
