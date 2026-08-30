"""Fetching the benchmark files this service scores against.

The scoring itself lives in `libs/word2vec` -- two services need it now, since
`content-scoring` extends this vocabulary onto the Hacker News corpus and has to show
it did not damage these vectors doing so. What stays here is the part that is genuinely
this service's: knowing that the three files are mirrored at
`roshbeed/ai-residency-word-embeddings-eval` and pulling them through the Hub helper.

Everything the old module exported is re-exported, so existing call sites are unchanged.
"""

from pathlib import Path

from ai_residency_common import get_logger, resolve_checkpoint
from config import Config
from word2vec import (
    AnalogyQuestion,
    Benchmarks,
    SimilarityPair,
    analogy_accuracy,
    evaluate_embeddings,
    parse_analogies,
    parse_similarity_pairs,
    similarity_score,
)

__all__ = [
    "AnalogyQuestion",
    "Benchmarks",
    "SimilarityPair",
    "analogy_accuracy",
    "evaluate_embeddings",
    "load_benchmarks",
    "similarity_score",
]

logger = get_logger(__name__)

_SIMILARITY_FILES = {"wordsim353": "wordsim353.txt", "simlex999": "simlex999.txt"}
_ANALOGY_FILE = "questions-words.txt"


def _fetch(config: Config, filename: str) -> Path:
    """Resolve one benchmark file, pulling it from the Hub if it is not local.

    Args:
        config: Supplies `eval_dir` and `eval_repo_id`.
        filename: Name of the file within the dataset repo.

    Returns:
        Path to the local file.

    Raises:
        FileNotFoundError: If it is neither local nor fetchable.
    """
    path, _ = resolve_checkpoint(
        config.eval_dir / filename, config.eval_repo_id, repo_type="dataset"
    )
    if path is None:
        raise FileNotFoundError(
            f"{filename} is not in {config.eval_dir} and could not be fetched from "
            f"{config.eval_repo_id}"
        )
    return path


def load_benchmarks(config: Config) -> Benchmarks:
    """Load all three benchmark sets.

    Args:
        config: Supplies where the files live.

    Returns:
        The parsed benchmarks, lowercased to match the corpus.
    """
    similarity = {
        name: parse_similarity_pairs(_fetch(config, filename))
        for name, filename in _SIMILARITY_FILES.items()
    }
    analogies = parse_analogies(_fetch(config, _ANALOGY_FILE))

    logger.info(
        "Benchmarks: %s, %d analogy questions",
        ", ".join(f"{name} {len(pairs)} pairs" for name, pairs in similarity.items()),
        len(analogies),
    )
    return Benchmarks(similarity, analogies)
