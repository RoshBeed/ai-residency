"""Build (query, positive document, negative document) triples from MS MARCO.

MS MARCO gives a query and a handful of candidate passages, one or more flagged as
relevant. Triplet loss needs an irrelevant passage too, and the dataset does not label
one, so a negative is sampled at random from the split's other passages. Random negatives
are weak — a passage drawn at random is usually obviously unrelated, so the towers learn
an easy problem — but harder negative mining is a change to the method, not a cleanup, so
this stays as the team wrote it.
"""

import json
import random

from tqdm import tqdm

from ai_residency_common import get_logger, set_seed
from config import Config
from datasets import load_dataset

__all__ = ["generate_triples", "main"]

logger = get_logger(__name__)

SPLITS = ("train", "validation", "test")


def generate_triples(config: Config) -> dict[str, list[dict[str, str]]]:
    """Build triples for every split.

    Args:
        config: Supplies the seed and where to write the result.

    Returns:
        Triples keyed by split.
    """
    set_seed(config.seed)
    logger.info("Loading MS MARCO")
    dataset = load_dataset("ms_marco", "v1.1")

    triples: dict[str, list[dict[str, str]]] = {split: [] for split in SPLITS}
    for split in SPLITS:
        split_data = dataset[split]

        passages = {
            passage for example in split_data for passage in example["passages"]["passage_text"]
        }
        pool = list(passages)
        logger.info("%s: %d unique passages to sample negatives from", split, len(pool))
        if len(pool) < 2:
            continue

        for example in tqdm(split_data, desc=f"Triples for {split}"):
            relevance = example["passages"]["is_selected"]
            for passage, is_relevant in zip(
                example["passages"]["passage_text"], relevance, strict=True
            ):
                if not is_relevant:
                    continue
                # Redraw on the rare collision with the positive itself, rather than
                # filtering the whole pool once per positive example.
                negative = random.choice(pool)
                while negative == passage:
                    negative = random.choice(pool)
                triples[split].append(
                    {
                        "query": example["query"],
                        "positive_doc": passage,
                        "negative_doc": negative,
                    }
                )

        logger.info("%s: generated %d triples", split, len(triples[split]))

    config.data_dir.mkdir(parents=True, exist_ok=True)
    with config.triples_path.open("w") as handle:
        json.dump(triples, handle, indent=2)
    logger.info("Wrote %s", config.triples_path)
    return triples


def main() -> None:
    """Entry point for `make train-triples`."""
    generate_triples(Config())


if __name__ == "__main__":
    main()
