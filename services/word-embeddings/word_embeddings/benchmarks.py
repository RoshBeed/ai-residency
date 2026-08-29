"""Intrinsic evaluation against the standard word-vector benchmarks.

Three of them, mirrored at `roshbeed/ai-residency-word-embeddings-eval`:

* **WordSim-353** -- 353 human-scored pairs. Spearman rank correlation. Measures
  *relatedness*: `computer`/`keyboard` scores high here despite not being similar.
* **SimLex-999** -- 999 pairs, scored for *similarity* specifically, and deliberately
  built so that a model which only captures relatedness does badly. Scores are much
  lower than WordSim-353 across the board; ~0.3 is a normal word2vec result, not a
  failure.
* **Google analogies** -- 19,544 `a:b :: c:d` questions in 14 categories, five
  semantic and nine syntactic.

What this replaces was three hand-picked word pairs and one analogy. A Spearman
correlation over three points can only take the values {-1, -0.5, 0.5, 1}, so the
"1.0" the write-up reported carried roughly one bit of information.

Two details decide whether an analogy score is comparable to a published one:

**All three query words are excluded from the answer.** `king - man + woman` lands
nearest to `king` itself far more often than to `queen`, so a scorer that does not
exclude a, b and c measures almost nothing. The previous implementation did not.

**Candidates are restricted to the most frequent words.** The convention is the top
30k; searching the full tail lets a rare word with an under-trained vector win by
accident, and makes the score depend on `min_count` rather than on the embeddings.
"""

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from ai_residency_common import get_logger, resolve_checkpoint
from config import Config

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

SimilarityPair = tuple[str, str, float]
AnalogyQuestion = tuple[str, str, str, str, bool]
"""`(a, b, c, expected, is_semantic)`. The Google set's `gram*` categories are the
syntactic ones; the other five are semantic."""

_SIMILARITY_FILES = {"wordsim353": "wordsim353.txt", "simlex999": "simlex999.txt"}
_ANALOGY_FILE = "questions-words.txt"


class Benchmarks:
    """The loaded benchmark sets.

    Attributes:
        similarity: Benchmark name to its human-scored pairs.
        analogies: The analogy questions, with their semantic/syntactic flag.
    """

    def __init__(
        self,
        similarity: dict[str, list[SimilarityPair]],
        analogies: list[AnalogyQuestion],
    ) -> None:
        self.similarity = similarity
        self.analogies = analogies


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
    similarity: dict[str, list[SimilarityPair]] = {}
    for name, filename in _SIMILARITY_FILES.items():
        pairs: list[SimilarityPair] = []
        for line in _fetch(config, filename).read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) == 3:
                pairs.append((fields[0].lower(), fields[1].lower(), float(fields[2])))
        similarity[name] = pairs

    analogies: list[AnalogyQuestion] = []
    is_semantic = True
    for line in _fetch(config, _ANALOGY_FILE).read_text(encoding="utf-8").splitlines():
        if line.startswith(":"):
            is_semantic = not line.split()[1].startswith("gram")
            continue
        fields = line.lower().split()
        if len(fields) == 4:
            analogies.append((fields[0], fields[1], fields[2], fields[3], is_semantic))

    logger.info(
        "Benchmarks: %s, %d analogy questions",
        ", ".join(f"{name} {len(pairs)} pairs" for name, pairs in similarity.items()),
        len(analogies),
    )
    return Benchmarks(similarity, analogies)


def similarity_score(
    embeddings: torch.Tensor,
    word_to_index: dict[str, int],
    pairs: list[SimilarityPair],
) -> tuple[float, float]:
    """Spearman correlation between cosine similarity and human judgement.

    Args:
        embeddings: `(vocab_size, dim)` embedding matrix.
        word_to_index: Vocabulary the rows are indexed by.
        pairs: `(word1, word2, human_score)` triples.

    Returns:
        `(spearman, coverage)`. Coverage is the fraction of pairs whose words are both
        in the vocabulary, and is returned alongside the score rather than folded into
        it: a correlation over the 60% of pairs a small vocabulary happens to cover is
        not comparable to one over 95%, and reporting only the number invites exactly
        that mistake. `(nan, coverage)` when fewer than two pairs are usable.
    """
    usable = [
        (word_to_index[first], word_to_index[second], human)
        for first, second, human in pairs
        if first in word_to_index and second in word_to_index
    ]
    coverage = len(usable) / max(len(pairs), 1)
    if len(usable) < 2:
        return float("nan"), coverage

    normalised = F.normalize(embeddings, dim=1)
    left = normalised[torch.tensor([pair[0] for pair in usable])]
    right = normalised[torch.tensor([pair[1] for pair in usable])]
    predicted = (left * right).sum(dim=1).cpu().tolist()
    human_scores = [pair[2] for pair in usable]

    correlation, _ = spearmanr(predicted, human_scores)
    return float(correlation), coverage


def analogy_accuracy(
    embeddings: torch.Tensor,
    word_to_index: dict[str, int],
    questions: list[AnalogyQuestion],
    *,
    max_vocab: int = 30_000,
    batch_size: int = 512,
) -> dict[str, float]:
    """Accuracy on `a : b :: c : ?` by 3CosAdd, excluding a, b and c.

    Args:
        embeddings: `(vocab_size, dim)` embedding matrix.
        word_to_index: Vocabulary the rows are indexed by.
        questions: `(a, b, c, expected, is_semantic)` tuples.
        max_vocab: Search answers among this many most-frequent words. Requires the
            vocabulary to be frequency-ordered, which `corpus.build_vocabulary`
            guarantees.
        batch_size: Questions scored per matmul.

    Returns:
        `semantic`, `syntactic` and `total` accuracy, plus `coverage` -- the fraction
        of questions whose four words are all in the vocabulary. Accuracies are over
        the covered questions only, so they are not diluted by words the corpus never
        contained.
    """
    candidates = min(max_vocab, embeddings.shape[0])
    normalised = F.normalize(embeddings[:candidates], dim=1)

    usable = [
        (word_to_index[a], word_to_index[b], word_to_index[c], word_to_index[d], semantic)
        for a, b, c, d, semantic in questions
        if all(word in word_to_index for word in (a, b, c, d))
        and all(word_to_index[word] < candidates for word in (a, b, c, d))
    ]
    coverage = len(usable) / max(len(questions), 1)
    correct = {True: 0, False: 0}
    total = {True: 0, False: 0}

    for start in range(0, len(usable), batch_size):
        chunk = usable[start : start + batch_size]
        rows = torch.tensor([[a, b, c, d] for a, b, c, d, _ in chunk])
        query = F.normalize(
            normalised[rows[:, 1]] - normalised[rows[:, 0]] + normalised[rows[:, 2]], dim=1
        )
        scores = query @ normalised.T

        # a, b and c are excluded, not merely deprioritised: the query vector is a
        # short hop from c, so without this the answer is almost always c itself.
        scores.scatter_(1, rows[:, :3].to(scores.device), float("-inf"))

        predicted = scores.argmax(dim=1).cpu()
        for (_, _, _, expected, semantic), guess in zip(chunk, predicted, strict=True):
            total[semantic] += 1
            correct[semantic] += int(guess) == expected

    answered = total[True] + total[False]
    return {
        "semantic": correct[True] / total[True] if total[True] else float("nan"),
        "syntactic": correct[False] / total[False] if total[False] else float("nan"),
        "total": (correct[True] + correct[False]) / answered if answered else float("nan"),
        "coverage": coverage,
    }


def evaluate_embeddings(
    embeddings: torch.Tensor,
    word_to_index: dict[str, int],
    benchmarks: Benchmarks,
    *,
    max_analogy_vocab: int = 30_000,
) -> dict[str, float]:
    """Score embeddings on all three benchmarks.

    Args:
        embeddings: `(vocab_size, dim)` embedding matrix.
        word_to_index: Vocabulary the rows are indexed by.
        benchmarks: The loaded benchmark sets.
        max_analogy_vocab: Candidate restriction for the analogy search.

    Returns:
        Flat metric names to values, ready to hand to `log_metrics`. Includes a
        `score` composite -- the mean of the two correlations and analogy accuracy --
        so a sweep has one number to rank on without hiding the three behind it.
    """
    embeddings = embeddings.detach()
    metrics: dict[str, float] = {}

    for name, pairs in benchmarks.similarity.items():
        correlation, coverage = similarity_score(embeddings, word_to_index, pairs)
        metrics[f"{name}_spearman"] = correlation
        metrics[f"{name}_coverage"] = coverage

    analogy = analogy_accuracy(
        embeddings, word_to_index, benchmarks.analogies, max_vocab=max_analogy_vocab
    )
    for key, value in analogy.items():
        metrics[f"analogy_{key}"] = value

    # The mean of whichever components could be scored. A benchmark whose words the
    # vocabulary does not contain yields NaN, and a plain mean would propagate that
    # to the composite -- which a sweep ranks on, so one uncovered benchmark would
    # make an otherwise good run unrankable rather than merely less well measured.
    components = [
        metrics["wordsim353_spearman"],
        metrics["simlex999_spearman"],
        metrics["analogy_total"],
    ]
    scored = [value for value in components if not math.isnan(value)]
    metrics["score"] = sum(scored) / len(scored) if scored else float("nan")
    return metrics
