"""Exploratory dump of MS MARCO queries and their labelled passages.

Not part of the pipeline. It exists to look at the raw data before deciding how to build
triples from it, and to see how few passages per query are actually marked relevant.
"""

import json
from typing import TypedDict

from ai_residency_common import get_logger
from config import Config
from data_prep.generate_triples import SPLITS
from datasets import load_dataset

__all__ = ["ExtractedQuery", "extract_queries_and_documents", "main"]

logger = get_logger(__name__)


class LabelledPassage(TypedDict):
    """One candidate passage and whether MS MARCO marked it relevant."""

    passage: str
    is_relevant: bool


class ExtractedQuery(TypedDict):
    """One query and all of its candidate passages."""

    query: str
    passages_with_relevance: list[LabelledPassage]


def extract_queries_and_documents(config: Config) -> dict[str, list[ExtractedQuery]]:
    """Dump every query with its passages and relevance flags.

    Args:
        config: Supplies `data_dir`.

    Returns:
        The extracted data, keyed by split.
    """
    logger.info("Loading MS MARCO")
    dataset = load_dataset(config.source_dataset_repo_id, config.source_dataset_config)

    extracted: dict[str, list[ExtractedQuery]] = {split: [] for split in SPLITS}
    for split in SPLITS:
        for example in dataset[split]:
            extracted[split].append(
                ExtractedQuery(
                    query=example["query"],
                    passages_with_relevance=[
                        LabelledPassage(passage=passage, is_relevant=bool(is_relevant))
                        for passage, is_relevant in zip(
                            example["passages"]["passage_text"],
                            example["passages"]["is_selected"],
                            strict=True,
                        )
                    ],
                )
            )

        total = sum(len(item["passages_with_relevance"]) for item in extracted[split])
        logger.info("%s: %d queries, %d passages", split, len(extracted[split]), total)

    config.data_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.data_dir / "extracted_data.json"
    with output_path.open("w") as handle:
        json.dump(extracted, handle, indent=2)
    logger.info("Wrote %s", output_path)
    return extracted


def main() -> None:
    """Entry point for the exploratory extract."""
    extract_queries_and_documents(Config())


if __name__ == "__main__":
    main()
