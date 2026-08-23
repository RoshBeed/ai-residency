"""Map the text triples through the CBOW vocabulary.

Words the CBOW corpus never saw become token 0, the padding id. That is a real limit on
this pipeline rather than an implementation detail: MS MARCO passages contain plenty of
vocabulary `text8` does not, and every one of those words arrives at the towers as
padding, indistinguishable from the others.
"""

import json
from typing import Any

from ai_residency_common import get_logger
from tqdm import tqdm

from cbow.vocabulary import load_words_to_ids
from config import Config
from data_prep.generate_triples import SPLITS

__all__ = ["Triple", "main", "tokenize_text", "tokenize_triples"]

logger = get_logger(__name__)

# One tokenized triple: three token id lists plus the text they came from. `Any` because
# the values are heterogeneous (`list[int]` and `str`) and it is read straight from JSON.
Triple = dict[str, Any]


def tokenize_text(text: str, words_to_ids: dict[str, int]) -> list[int]:
    """Map text to token ids, sending unknown words to the padding id.

    Args:
        text: Text to tokenize.
        words_to_ids: Vocabulary mapping word to token id.

    Returns:
        One token id per whitespace-separated word.
    """
    return [words_to_ids.get(word, 0) for word in text.lower().split()]


def tokenize_triples(config: Config) -> dict[str, list[Triple]]:
    """Tokenize every triple, keeping the original text alongside the ids.

    Args:
        config: Supplies the vocabulary and both triple paths.

    Returns:
        Tokenized triples keyed by split.
    """
    words_to_ids = load_words_to_ids(config)
    with config.triples_path.open() as handle:
        triples = json.load(handle)

    tokenized: dict[str, list[Triple]] = {split: [] for split in SPLITS}
    for split in SPLITS:
        for triple in tqdm(triples[split], desc=f"Tokenizing {split}"):
            tokenized[split].append(
                {
                    "query_tokens": tokenize_text(triple["query"], words_to_ids),
                    "positive_document_tokens": tokenize_text(triple["positive_doc"], words_to_ids),
                    "negative_document_tokens": tokenize_text(triple["negative_doc"], words_to_ids),
                    "query": triple["query"],
                    "positive_document": triple["positive_doc"],
                    "negative_document": triple["negative_doc"],
                }
            )
        logger.info("%s: %d tokenized triples", split, len(tokenized[split]))

    config.data_dir.mkdir(parents=True, exist_ok=True)
    with config.tokenized_triples_path.open("w") as handle:
        json.dump(tokenized, handle, indent=2)
    logger.info("Wrote %s", config.tokenized_triples_path)
    return tokenized


def main() -> None:
    """Entry point for the tokenize stage."""
    tokenize_triples(Config())


if __name__ == "__main__":
    main()
