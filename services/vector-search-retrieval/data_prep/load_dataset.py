"""Print MS MARCO split sizes and a couple of examples.

Pure inspection, no outputs. Useful for sanity-checking that the dataset resolves at all
before spending time on the stages that consume it.
"""

from ai_residency_common import get_logger
from config import Config
from data_prep.generate_triples import SPLITS
from datasets import load_dataset

__all__ = ["describe_dataset", "main"]

logger = get_logger(__name__)


def describe_dataset(config: Config) -> None:
    """Log split sizes, average lengths and a sample query per split.

    Args:
        config: Supplies the source dataset to read.
    """
    logger.info("Loading MS MARCO")
    dataset = load_dataset(config.source_dataset_repo_id, config.source_dataset_config)

    for split in SPLITS:
        split_data = dataset[split]
        query_lengths = [len(example["query"].split()) for example in split_data]
        passage_lengths = [
            len(passage.split())
            for example in split_data
            for passage in example["passages"]["passage_text"]
        ]
        logger.info(
            "%s: %d queries, mean query %.2f words, %d passages, mean passage %.2f words",
            split,
            len(split_data),
            sum(query_lengths) / max(len(query_lengths), 1),
            len(passage_lengths),
            sum(passage_lengths) / max(len(passage_lengths), 1),
        )
        for example in list(split_data)[:2]:
            logger.info(
                "  query=%r  passages=%d",
                example["query"],
                len(example["passages"]["passage_text"]),
            )


def main() -> None:
    """Entry point for the dataset inspection script."""
    describe_dataset(Config())


if __name__ == "__main__":
    main()
