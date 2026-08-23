"""Stage 1: build the corpus and vocabulary from `text8`.

Was `00_train_tkn.py`. A leading-digit file name is not a valid module name, so it
could only ever be run as a loose script from inside its own directory — which is also
why its outputs landed next to the source code instead of in `data/`. Both are fixed
here: this is `python -m cbow.train_tokenizer`
and it writes to `config.data_dir`.
"""

import collections
import pickle

import requests
from ai_residency_common import get_logger

from config import Config

__all__ = ["build_vocabulary", "download_corpus", "main", "preprocess"]

logger = get_logger(__name__)

# Punctuation is replaced with word-like tokens rather than stripped, so the CBOW model
# can learn that a full stop is a boundary instead of silently joining two sentences.
_PUNCTUATION_TOKENS = {
    ".": " <PERIOD> ",
    ",": " <COMMA> ",
    '"': " <QUOTATION_MARK> ",
    ";": " <SEMICOLON> ",
    "!": " <EXCLAMATION_MARK> ",
    "?": " <QUESTION_MARK> ",
    "(": " <LEFT_PAREN> ",
    ")": " <RIGHT_PAREN> ",
    "--": " <HYPHENS> ",
    ":": " <COLON> ",
}


def download_corpus(config: Config) -> str:
    """Download `text8` if it is not already on disk, and return its text.

    Args:
        config: Supplies `text8_url` and `text8_path`.

    Returns:
        The raw corpus as one string.
    """
    if not config.text8_path.exists():
        logger.info("Downloading %s", config.text8_url)
        config.text8_path.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(config.text8_url, timeout=300)
        response.raise_for_status()
        config.text8_path.write_bytes(response.content)
    return config.text8_path.read_text()


def preprocess(text: str, min_word_count: int) -> list[str]:
    """Lowercase, tokenize punctuation, split, and drop rare words.

    Args:
        text: Raw corpus text.
        min_word_count: Words appearing this many times or fewer are dropped. Rare
            words never get enough context to learn a useful vector and each one costs
            a full row of the embedding matrix and the output projection.

    Returns:
        The corpus as a flat list of words.
    """
    text = text.lower()
    for symbol, replacement in _PUNCTUATION_TOKENS.items():
        text = text.replace(symbol, replacement)
    words = text.split()
    counts = collections.Counter(words)
    return [word for word in words if counts[word] > min_word_count]


def build_vocabulary(words: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    """Assign token ids to words, most frequent first.

    Id 0 is reserved for `<PAD>` so that a padded context window is distinguishable
    from a real word.

    Args:
        words: The preprocessed corpus.

    Returns:
        A `(words_to_ids, ids_to_words)` pair.
    """
    counts = collections.Counter(words)
    ordered = sorted(counts, key=lambda word: counts[word], reverse=True)
    ids_to_words = {index + 1: word for index, word in enumerate(ordered)}
    ids_to_words[0] = "<PAD>"
    words_to_ids = {word: index for index, word in ids_to_words.items()}
    return words_to_ids, ids_to_words


def main() -> None:
    """Build and persist the corpus and vocabulary."""
    config = Config()
    config.data_dir.mkdir(parents=True, exist_ok=True)

    corpus = preprocess(download_corpus(config), config.min_word_count)
    logger.info("Corpus: %d words, first few %s", len(corpus), corpus[:7])

    words_to_ids, ids_to_words = build_vocabulary(corpus)
    logger.info("Vocabulary: %d words", len(words_to_ids))

    with config.corpus_path.open("wb") as handle:
        pickle.dump(corpus, handle)
    with config.words_to_ids_path.open("wb") as handle:
        pickle.dump(words_to_ids, handle)
    with config.ids_to_words_path.open("wb") as handle:
        pickle.dump(ids_to_words, handle)
    logger.info("Wrote corpus and vocabulary to %s", config.data_dir)


if __name__ == "__main__":
    main()
