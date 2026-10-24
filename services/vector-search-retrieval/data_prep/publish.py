"""Publish the artifact set this service serves from, as one set.

**These files only mean anything together.** The vocabulary maps words to row numbers,
the CBOW matrix has one row per word, each tower carries that matrix frozen inside it,
and `tokenized_triples.json` holds ids that index the same mapping. Publish five of the
six and the Hub serves a set that disagrees with itself.

That is not hypothetical. It has now happened twice. The first time the tokenizer and
tokenize stages wrote their outputs locally and uploaded nothing, so the Hub held a
63,642-row CBOW checkpoint beside a 7,692-word vocabulary and every container answered
503. The second time the vocabulary was extended to 88,114 words and everything was
published except the triples, so `/query` worked -- it tokenizes live -- while the index
job refused to build against ids from the old mapping and the deployed search returned
nothing at all. The guard did its job on both; what was missing was a way to publish the
set without having to remember what is in it.

So: one command, and it checks the set agrees before it sends anything.
"""

import argparse
import json

from ai_residency_common import get_logger, load_checkpoint, push_checkpoint
from cbow.vocabulary import (
    CANONICAL_CBOW_CHECKPOINT,
    latest_cbow_checkpoint,
    load_words_to_ids,
    vocabulary_fingerprint,
)
from config import Config

__all__ = ["check_consistent", "main", "publish"]

logger = get_logger(__name__)


def check_consistent(config: Config) -> int:
    """Raise unless every local artifact indexes the same vocabulary.

    Args:
        config: Supplies every artifact path.

    Returns:
        The vocabulary size the whole set agrees on.

    Raises:
        ValueError: If any artifact disagrees with the vocabulary.
    """
    words_to_ids = load_words_to_ids(config)
    size = len(words_to_ids)
    fingerprint = vocabulary_fingerprint(words_to_ids)
    logger.info("Vocabulary: %d words, fingerprint %s", size, fingerprint)

    cbow_rows = load_checkpoint(latest_cbow_checkpoint(config))["model_state_dict"][
        "in_embed.weight"
    ].shape[0]
    if cbow_rows != size:
        raise ValueError(
            f"CBOW matrix has {cbow_rows} rows against a {size}-word vocabulary. "
            "Rebuild it with `make train-cbow && make extend-vocab`."
        )

    for name, path in (
        ("query", config.query_tower_path),
        ("document", config.document_tower_path),
    ):
        state = load_checkpoint(path)
        state = state.get("model_state_dict", state)
        rows = next(
            (tensor.shape[0] for key, tensor in state.items() if key.endswith("embedding.weight")),
            None,
        )
        if rows is not None and rows != size:
            raise ValueError(
                f"The {name} tower embeds {rows} words against a {size}-word vocabulary. "
                "Retrain it with `make train-encoder`."
            )

    with config.tokenized_triples_path.open() as handle:
        recorded = json.load(handle).get("vocabulary_fingerprint")
    if recorded != fingerprint:
        raise ValueError(
            f"{config.tokenized_triples_path} records vocabulary {recorded}, not {fingerprint}. "
            "Rebuild it with `make train-triples`."
        )

    logger.info("All artifacts agree on the %d-word vocabulary", size)
    return size


def publish(config: Config) -> None:
    """Check the set, then push every part of it.

    Args:
        config: Supplies the repo ids and every artifact path.
    """
    check_consistent(config)

    if config.hf_repo_id is not None:
        push_checkpoint(
            config.hf_repo_id,
            latest_cbow_checkpoint(config),
            path_in_repo=CANONICAL_CBOW_CHECKPOINT,
        )
        push_checkpoint(config.hf_repo_id, config.query_tower_path)
        push_checkpoint(config.hf_repo_id, config.document_tower_path)

    if config.hf_dataset_repo_id is not None:
        for path in (
            config.words_to_ids_path,
            config.ids_to_words_path,
            config.tokenized_triples_path,
        ):
            push_checkpoint(config.hf_dataset_repo_id, path, repo_type="dataset")


def main() -> None:
    """Entry point for `make publish`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="verify the local set agrees and publish nothing",
    )
    args = parser.parse_args()
    config = Config()
    if args.check_only:
        check_consistent(config)
    else:
        publish(config)


if __name__ == "__main__":
    main()
