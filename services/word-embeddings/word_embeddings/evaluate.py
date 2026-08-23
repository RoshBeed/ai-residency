"""Intrinsic evaluation of the learned word vectors.

Two standard word2vec checks, both weak on their own and worth stating plainly:

* the analogy test (``king - man + woman ~= queen``),
* a Spearman correlation against a handful of hand-scored similarity pairs.

They are a smoke signal that the vectors learned *structure* during a long run, not
a measure of embedding quality. The real evaluation of these vectors is the
downstream content-scoring head in the sibling
`ai-residency-week1-content-scoring` repo. ``visualise_embeddings`` projects the space
to 2D with t-SNE.

Run ``python -m word_embeddings.evaluate`` to load the saved checkpoint and print both.
"""

import argparse

import matplotlib

# Agg before pyplot: on macOS the default interactive backend blocks on `plt.show()`
# when this runs as a plain script rather than inside a notebook.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from ai_residency_common import get_logger, load_checkpoint, resolve_required
from scipy.stats import spearmanr
from sklearn.manifold import TSNE

from config import Config

__all__ = ["evaluate_embeddings", "main", "visualise_embeddings"]

logger = get_logger(__name__)

# Pairs with a rough human judgement of similarity, for a Spearman correlation.
_SIMILARITY_PAIRS = (("king", "queen"), ("dog", "cat"), ("king", "dog"))
_HUMAN_SCORES = (0.95, 0.85, 0.2)

# The canonical word2vec analogy: king - man + woman ~= queen.
_ANALOGY = ("king", "man", "woman", "queen")


def _nearest_word(
    embeddings: torch.Tensor, vector: torch.Tensor, index_to_word: dict[int, str]
) -> str:
    """Return the vocabulary word whose vector is closest to `vector`."""
    similarities = F.cosine_similarity(embeddings, vector.unsqueeze(0)).cpu()
    return index_to_word[int(similarities.argmax())]


def evaluate_embeddings(
    embeddings: torch.Tensor, word_to_index: dict[str, int]
) -> dict[str, float]:
    """Score the embeddings on the analogy and similarity checks.

    Args:
        embeddings: The input-embedding matrix, one row per vocabulary entry.
        word_to_index: Vocabulary the rows are indexed by.

    Returns:
        `analogy_correct` (1.0 or 0.0) and `similarity_spearman`. Either is omitted
        when the vocabulary lacks the words that check needs — a real corpus is not
        obliged to contain "queen", and silently scoring 0.0 for that would look like
        a training failure rather than a missing word.
    """
    metrics: dict[str, float] = {}
    index_to_word = {index: word for word, index in word_to_index.items()}

    pairs = [pair for pair in _SIMILARITY_PAIRS if all(word in word_to_index for word in pair)]
    if len(pairs) == len(_SIMILARITY_PAIRS):
        model_scores = [
            F.cosine_similarity(
                embeddings[word_to_index[first]], embeddings[word_to_index[second]], dim=0
            ).item()
            for first, second in pairs
        ]
        correlation, _ = spearmanr(model_scores, list(_HUMAN_SCORES))
        metrics["similarity_spearman"] = float(correlation)

    first, second, third, expected = _ANALOGY
    if all(word in word_to_index for word in _ANALOGY):
        predicted = _nearest_word(
            embeddings,
            embeddings[word_to_index[second]]
            - embeddings[word_to_index[first]]
            + embeddings[word_to_index[third]],
            index_to_word,
        )
        metrics["analogy_correct"] = float(predicted == expected)
        logger.debug("Analogy king-man+woman predicted %r, expected %r", predicted, expected)

    return metrics


def visualise_embeddings(config: Config, max_embeddings: int, output: str) -> None:
    """Write a t-SNE scatter plot of the most frequent word vectors."""
    checkpoint = load_checkpoint(config.cbow_checkpoint)
    embeddings = checkpoint["model_state_dict"]["in_embed.weight"].cpu()
    word_to_index: dict[str, int] = checkpoint["extra"]["word_to_lemma_index"]

    words = sorted(word_to_index, key=lambda word: word_to_index[word])[:max_embeddings]
    vectors = embeddings[: len(words)].numpy()
    logger.info("Projecting %d of %d vectors", len(words), embeddings.shape[0])

    perplexity = min(30, max(2, len(words) - 1))
    reduced = TSNE(n_components=2, perplexity=perplexity, random_state=config.seed).fit_transform(
        vectors
    )

    plt.figure(figsize=(10, 10))
    for (x, y), word in zip(reduced, words, strict=True):
        plt.scatter(x, y, s=6)
        plt.annotate(word, (x, y), fontsize=7)
    plt.title(f"CBOW embeddings, t-SNE projection (top {len(words)})")
    plt.grid(True)
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Wrote %s", output)


def main() -> None:
    """Load the saved CBOW checkpoint and report its intrinsic metrics."""
    parser = argparse.ArgumentParser(description="Evaluate the learned word vectors")
    parser.add_argument(
        "--visualise",
        action="store_true",
        help="Also write a t-SNE plot of the embedding space",
    )
    parser.add_argument("--max-embeddings", type=int, default=300)
    parser.add_argument("--output", default="assets/embeddings_tsne.png")
    args = parser.parse_args()

    config = Config()

    cbow_path, cbow_source = resolve_required(config.cbow_checkpoint, config.hf_repo_id)
    cbow = load_checkpoint(cbow_path)
    embeddings = cbow["model_state_dict"]["in_embed.weight"]
    word_to_index: dict[str, int] = cbow["extra"]["word_to_lemma_index"]
    logger.info(
        "CBOW checkpoint from %s (epoch %s, loss %.4f)",
        cbow_source,
        cbow["epoch"],
        cbow["metrics"].get("epoch_loss", float("nan")),
    )
    logger.info("Intrinsic embedding metrics: %s", evaluate_embeddings(embeddings, word_to_index))

    if args.visualise:
        visualise_embeddings(config, args.max_embeddings, args.output)


if __name__ == "__main__":
    main()
