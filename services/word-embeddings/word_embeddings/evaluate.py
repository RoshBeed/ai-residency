"""Score a trained checkpoint and look at what it learned.

    python -m word_embeddings.evaluate              # the benchmark table
    python -m word_embeddings.evaluate --visualise  # and a t-SNE plot

The scoring itself lives in `benchmarks.py`; this module resolves a checkpoint,
prints the numbers next to the baselines that make them mean something, and draws
the embedding space.

`visualise_embeddings` projects to 2D with t-SNE. Its "top N most frequent words"
slice is `[:n]` because `corpus.build_vocabulary` assigns ids in descending frequency
order -- under the previous pipeline ids came out in first-appearance order, so the
figure captioned "top 100 by frequency" was showing the first 100 words the tokenizer
happened to encounter.
"""

import argparse

import matplotlib

# Agg before pyplot: on macOS the default interactive backend blocks on `plt.show()`
# when this runs as a plain script rather than inside a notebook.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from ai_residency_common import get_logger, load_checkpoint, resolve_required, set_seed
from ai_residency_common.figures import save_figure
from config import Config
from word2vec import nearest_words
from word_embeddings.benchmarks import evaluate_embeddings, load_benchmarks

__all__ = ["checkpoint_vocabulary", "main", "report", "visualise_embeddings"]

logger = get_logger(__name__)

# Published word2vec results on text8, for the table to sit next to. A number with no
# baseline beside it is not a result.
PUBLISHED_TEXT8 = {
    "wordsim353_spearman": 0.68,
    "simlex999_spearman": 0.30,
    "analogy_total": 0.38,
}


def checkpoint_vocabulary(extra: dict[str, object]) -> dict[str, int]:
    """Read the vocabulary a checkpoint's embedding rows are indexed by.

    Args:
        extra: The checkpoint's `extra` payload.

    Returns:
        The word-to-index mapping.

    Raises:
        KeyError: If the checkpoint carries no vocabulary at all.

    Prefers `word_to_index` and falls back to `word_to_lemma_index` so checkpoints
    written before the index spaces were untangled still load. They will score badly,
    and that is not a regression: `word_to_lemma_index` maps words to *lemma* ids
    while the matrix was trained on *word* ids, so those checkpoints were always being
    read through the wrong mapping.
    """
    for key in ("word_to_index", "word_to_lemma_index"):
        if key in extra:
            vocabulary: dict[str, int] = extra[key]  # type: ignore[assignment]
            if key != "word_to_index":
                logger.warning(
                    "Checkpoint predates the vocabulary fix; reading %s, whose ids do "
                    "not index these rows. Retrain for meaningful scores.",
                    key,
                )
            return vocabulary
    raise KeyError("checkpoint carries no vocabulary")


def report(config: Config) -> dict[str, float]:
    """Score the configured checkpoint and log the table.

    Args:
        config: Supplies the checkpoint location and the analogy restriction.

    Returns:
        The benchmark metrics.
    """
    path, source = resolve_required(config.embedding_checkpoint, config.hf_repo_id)
    checkpoint = load_checkpoint(path)
    embeddings = checkpoint["model_state_dict"]["in_embed.weight"].cpu()
    word_to_index = checkpoint_vocabulary(checkpoint["extra"])
    logger.info(
        "%s checkpoint from %s: %d rows, %d dims, epoch %s",
        config.architecture,
        source,
        embeddings.shape[0],
        embeddings.shape[1],
        checkpoint["epoch"],
    )

    benchmarks = load_benchmarks(config)
    metrics = evaluate_embeddings(
        embeddings, word_to_index, benchmarks, max_analogy_vocab=config.max_analogy_vocab
    )

    logger.info("%-24s %8s %8s %10s", "benchmark", "score", "published", "coverage")
    for name, coverage_key in (
        ("wordsim353_spearman", "wordsim353_coverage"),
        ("simlex999_spearman", "simlex999_coverage"),
        ("analogy_total", "analogy_coverage"),
    ):
        published = PUBLISHED_TEXT8.get(name)
        logger.info(
            "%-24s %8.4f %8s %9.1f%%",
            name,
            metrics[name],
            f"{published:.2f}" if published else "-",
            100 * metrics[coverage_key],
        )
    logger.info(
        "%-24s %8.4f  (semantic %.4f, syntactic %.4f)",
        "analogy breakdown",
        metrics["analogy_total"],
        metrics["analogy_semantic"],
        metrics["analogy_syntactic"],
    )

    index_to_word = {index: word for word, index in word_to_index.items()}
    for word in config.probe_words:
        if word in word_to_index:
            neighbours = nearest_words(
                embeddings, embeddings[word_to_index[word]], index_to_word, k=5, exclude=word
            )
            logger.info("  %-12s %s", word, ", ".join(other for other, _ in neighbours))
    return metrics


def visualise_embeddings(config: Config, max_embeddings: int, output: str) -> None:
    """Write a t-SNE scatter plot of the most frequent word vectors."""
    set_seed(config.seed)
    path, _ = resolve_required(config.embedding_checkpoint, config.hf_repo_id)
    checkpoint = load_checkpoint(path)
    embeddings = checkpoint["model_state_dict"]["in_embed.weight"].cpu()
    word_to_index = checkpoint_vocabulary(checkpoint["extra"])

    index_to_word = {index: word for word, index in word_to_index.items()}
    count = min(max_embeddings, embeddings.shape[0])
    words = [index_to_word[index] for index in range(count) if index in index_to_word]
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
    plt.title(f"{config.architecture} embeddings, t-SNE projection (top {len(words)})")
    plt.grid(True)
    save_figure(plt.gcf(), output)
    plt.close()
    logger.info("Wrote %s", output)


def main() -> None:
    """Load the saved checkpoint and report its intrinsic metrics."""
    parser = argparse.ArgumentParser(description="Evaluate the learned word vectors")
    parser.add_argument(
        "--visualise", action="store_true", help="Also write a t-SNE plot of the embedding space"
    )
    # 200 rather than 300: past roughly that, t-SNE packs the function words into a
    # centre where the labels overlap into illegibility, and the clusters worth
    # seeing are all at the periphery anyway.
    parser.add_argument("--max-embeddings", type=int, default=200)
    # Matches the filename the README displays. The default used to be
    # `embeddings_tsne.png`, so `make visualise` wrote a file nothing referenced and
    # never regenerated the figure in the write-up.
    parser.add_argument("--output", default="services/word-embeddings/assets/tsne_embeddings.png")
    args = parser.parse_args()

    config = Config()
    report(config)
    if args.visualise:
        visualise_embeddings(config, args.max_embeddings, args.output)


if __name__ == "__main__":
    main()
