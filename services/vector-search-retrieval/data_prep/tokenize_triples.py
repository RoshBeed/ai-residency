"""Map the text triples through the CBOW vocabulary.

Words the CBOW corpus never saw become token 0, the padding id. That is a real limit on
this pipeline rather than an implementation detail: MS MARCO passages contain plenty of
vocabulary `text8` does not, and every one of those words arrives at the towers as
padding, indistinguishable from the others.
"""

import json
from typing import Any

from tqdm import tqdm

from ai_residency_common import get_logger, push_checkpoint
from cbow.vocabulary import load_words_to_ids, vocabulary_fingerprint
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

    fingerprint = vocabulary_fingerprint(words_to_ids)
    logger.info("Tokenizing against vocabulary %s (%d words)", fingerprint, len(words_to_ids))

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
    # The fingerprint travels with the ids. Without it a stale file is undetectable:
    # these ids were once written against a 7,692-word vocabulary and then read against
    # a 63,642-word one, so every embedding lookup returned a different word's vector.
    # Training still converged and recall@10 still reproduced, because training and
    # evaluation shared the same wrong mapping -- the only thing that broke was serving,
    # which tokenizes query text live and therefore used the *right* mapping.
    payload: dict[str, Any] = dict(tokenized)
    payload["vocabulary_fingerprint"] = fingerprint
    with config.tokenized_triples_path.open("w") as handle:
        json.dump(payload, handle, indent=2)
    logger.info("Wrote %s", config.tokenized_triples_path)

    # Published for the same reason the vocabulary is: these token ids only mean
    # anything against the vocabulary that produced them, and `save_doc_embeddings_to_redis`
    # resolves this file from the Hub. A stale copy there does not fail -- it indexes
    # documents whose ids point into a different vocabulary than the towers' frozen
    # embedding matrix, and reports nothing. That is worse than the 503 a mismatched
    # checkpoint gives, because there is no symptom at all.
    if config.hf_dataset_repo_id is not None:
        push_checkpoint(
            config.hf_dataset_repo_id, config.tokenized_triples_path, repo_type="dataset"
        )

    return tokenized


def main() -> None:
    """Entry point for the tokenize stage."""
    tokenize_triples(Config())


if __name__ == "__main__":
    main()
