"""Score the towers the way a search system is actually judged.

Pairwise accuracy -- is the positive closer than the one sampled negative? -- is what
this service reported before, and it flatters a retriever badly. Chance is 50%, and the
sampled negative is a random passage from the corpus, so a model that has learned only
"documents about the same topic as the query are closer" scores well while being useless
at finding the right passage among thousands.

Recall@K asks the question the service exists to answer: rank the query against a whole
pool of documents and see whether the right one is in the top K. With a 5,000-document
pool chance recall@10 is 10/5000 = 0.0020, so there is no hiding in it.

Both numbers are reported, because the pairwise one is what the earlier checkpoints were
measured with and dropping it would make this run incomparable to them.
"""

import argparse
import json
import pickle
import statistics

import torch
import torch.nn.functional as F

from ai_residency_common import get_logger, load_checkpoint, set_seed
from cbow.vocabulary import build_embedding_layer
from config import Config
from data_prep.tokenize_triples import Triple
from models.towers import GruTower, encode_batch

__all__ = ["DOCUMENT_BATCH", "evaluate", "rank_queries", "recall_at_k"]

logger = get_logger(__name__)

# How many documents are encoded per forward pass when building the pool. Only a memory
# bound: the pool is encoded once, under no_grad.
DOCUMENT_BATCH = 256


def rank_queries(
    query_tower: GruTower,
    document_tower: GruTower,
    triples: list[Triple],
    *,
    pool_size: int,
) -> tuple[list[int], list[str], int]:
    """Rank every held-out query against the document pool and return the ranks.

    Recall@K, recall@1 and the median are all the same ranking, so it is done once here
    and the aggregates are derived from it. Reporting only a single K hides the shape of
    the result: this service's ranks run 893 firsts, 2,005 within ten, and 1,654 past a
    hundred, which is a different story from any one of those numbers alone.

    The pool is built from the held-out positives themselves, deduplicated, so every
    query has exactly one correct answer present and the rest of the pool is other
    queries' answers -- realistic distractors rather than random text.

    Args:
        query_tower: Tower encoding the queries.
        document_tower: Tower encoding the documents.
        triples: Held-out triples supplying queries and their positives.
        pool_size: Maximum documents to rank against.

    Returns:
        The 1-based rank of each query's own document, the query texts in the same
        order, and the pool size actually used.
    """
    document_index: dict[str, int] = {}
    documents: list[list[int]] = []
    pairs: list[tuple[list[int], int, str]] = []

    for triple in triples:
        query_tokens = triple["query_tokens"]
        document_tokens = triple["positive_document_tokens"]
        if not query_tokens or not document_tokens:
            continue
        key = triple["positive_document"]
        if key not in document_index:
            if len(documents) >= pool_size:
                continue
            document_index[key] = len(documents)
            documents.append(document_tokens)
        pairs.append((query_tokens, document_index[key], triple["query"]))

    if not pairs or not documents:
        return [], [], 0

    with torch.no_grad():
        encoded: list[torch.Tensor] = []
        for start in range(0, len(documents), DOCUMENT_BATCH):
            encoded.append(encode_batch(document_tower, documents[start : start + DOCUMENT_BATCH]))
        document_matrix = F.normalize(torch.cat(encoded), dim=1)

        ranks: list[int] = []
        texts: list[str] = []
        for start in range(0, len(pairs), DOCUMENT_BATCH):
            chunk = pairs[start : start + DOCUMENT_BATCH]
            queries = F.normalize(
                encode_batch(query_tower, [tokens for tokens, _, _ in chunk]), dim=1
            )
            similarities = queries @ document_matrix.T
            order = similarities.argsort(dim=1, descending=True)
            answers = torch.tensor([answer for _, answer, _ in chunk]).unsqueeze(1)
            # Where each row's own document landed in that row's ranking, 1-based.
            positions = (order == answers).nonzero()[:, 1] + 1
            ranks.extend(int(position) for position in positions.tolist())
            texts.extend(text for _, _, text in chunk)

    return ranks, texts, len(documents)


def recall_at_k(
    query_tower: GruTower,
    document_tower: GruTower,
    triples: list[Triple],
    *,
    k: int,
    pool_size: int,
) -> tuple[float, int, int]:
    """Fraction of queries whose own document ranks in the top K of a document pool.

    The pool is built from the held-out positives themselves, deduplicated. Every query
    therefore has exactly one correct answer present, and the rest of the pool is other
    queries' answers -- realistic distractors rather than random text.

    Args:
        query_tower: Tower encoding the queries.
        document_tower: Tower encoding the documents.
        triples: Held-out triples supplying queries and their positives.
        k: Rank cutoff.
        pool_size: Maximum documents to rank against.

    Returns:
        Recall@K, the number of queries scored, and the pool size actually used.
    """
    # One entry per distinct document, so a document shared by two queries is not
    # counted as two different correct answers.
    document_index: dict[str, int] = {}
    documents: list[list[int]] = []
    pairs: list[tuple[list[int], int]] = []

    for triple in triples:
        query_tokens = triple["query_tokens"]
        document_tokens = triple["positive_document_tokens"]
        if not query_tokens or not document_tokens:
            continue
        key = triple["positive_document"]
        if key not in document_index:
            if len(documents) >= pool_size:
                continue
            document_index[key] = len(documents)
            documents.append(document_tokens)
        pairs.append((query_tokens, document_index[key]))

    if not pairs or not documents:
        return 0.0, 0, 0

    with torch.no_grad():
        encoded: list[torch.Tensor] = []
        for start in range(0, len(documents), DOCUMENT_BATCH):
            encoded.append(encode_batch(document_tower, documents[start : start + DOCUMENT_BATCH]))
        document_matrix = F.normalize(torch.cat(encoded), dim=1)

        hits = 0
        cutoff = min(k, len(documents))
        for start in range(0, len(pairs), DOCUMENT_BATCH):
            chunk = pairs[start : start + DOCUMENT_BATCH]
            queries = F.normalize(encode_batch(query_tower, [tokens for tokens, _ in chunk]), dim=1)
            ranked = (queries @ document_matrix.T).topk(cutoff, dim=1).indices
            answers = torch.tensor([answer for _, answer in chunk]).unsqueeze(1)
            hits += int((ranked == answers).any(dim=1).sum())

    return hits / len(pairs), len(pairs), len(documents)


def evaluate(config: Config) -> dict[str, float]:
    """Score the saved towers against untrained ones and against chance.

    Args:
        config: Supplies the checkpoints, the held-out split and the pool size.

    Returns:
        Recall@K and pairwise accuracy for the trained and untrained towers, plus the
        chance rate for each.
    """
    from models.training import load_triples, pairwise_accuracy

    # The untrained towers are the control, and they are randomly initialised here rather
    # than loaded, so without a seed this harness reports a slightly different floor every
    # run and the README cannot quote one. The trained numbers do not move either way.
    set_seed(config.seed)

    triples = load_triples(config)
    held_out = triples["validation"]
    embedding_layer = build_embedding_layer(config)

    untrained_query = GruTower(embedding_layer, config.hidden_size).eval()
    untrained_document = GruTower(embedding_layer, config.hidden_size).eval()

    trained_query = GruTower(embedding_layer, config.hidden_size)
    trained_document = GruTower(embedding_layer, config.hidden_size)
    load_checkpoint(config.query_tower_path, trained_query)
    load_checkpoint(config.document_tower_path, trained_document)
    trained_query.eval()
    trained_document.eval()

    results: dict[str, float] = {}
    for name, query_tower, document_tower in (
        ("untrained", untrained_query, untrained_document),
        ("trained", trained_query, trained_document),
    ):
        ranks, texts, pool = rank_queries(
            query_tower,
            document_tower,
            held_out,
            pool_size=config.document_pool_size,
        )
        recall = sum(1 for rank in ranks if rank <= config.recall_at_k) / max(len(ranks), 1)
        accuracy, _ = pairwise_accuracy(query_tower, document_tower, held_out[:1000])
        results[f"{name}_recall_at_{config.recall_at_k}"] = recall
        results[f"{name}_recall_at_1"] = sum(1 for rank in ranks if rank == 1) / max(len(ranks), 1)
        results[f"{name}_median_rank"] = statistics.median(ranks) if ranks else 0.0
        results[f"{name}_pairwise_accuracy"] = accuracy
        logger.info(
            "%s: recall@%d %.4f, recall@1 %.4f, median rank %.0f "
            "over %d queries against %d documents, pairwise %.4f",
            name,
            config.recall_at_k,
            recall,
            results[f"{name}_recall_at_1"],
            results[f"{name}_median_rank"],
            len(ranks),
            pool,
            accuracy,
        )
        results["chance_recall"] = config.recall_at_k / max(pool, 1)

        # The vocabulary split, reported for the trained towers only -- it is a statement
        # about where this checkpoint's ceiling is, and an untrained model has no ceiling
        # to describe. `tokenize_text` maps an unknown word to token 0, which is also the
        # padding id, so an out-of-vocabulary word does not just lose its meaning: it
        # reaches the GRU indistinguishable from padding. The embeddings are trained on
        # text8 and the queries are real search traffic, so this is not a rare edge.
        if name == "trained" and ranks:
            results.update(_vocabulary_split(config, ranks, texts))

    logger.info(
        "chance recall@%d is %.4f and chance pairwise accuracy is 0.5000",
        config.recall_at_k,
        results["chance_recall"],
    )
    return results


def _vocabulary_split(config: Config, ranks: list[int], texts: list[str]) -> dict[str, float]:
    """Split recall by whether every word of the query is in the vocabulary.

    Args:
        config: Supplies the vocabulary path and K.
        ranks: Rank of each query's own document, from `rank_queries`.
        texts: The query texts, in the same order.

    Returns:
        Recall@K and the query count for each side of the split.
    """
    with config.words_to_ids_path.open("rb") as handle:
        vocabulary: dict[str, int] = pickle.load(handle)

    buckets: dict[str, list[int]] = {"known": [], "unknown": []}
    for rank, text in zip(ranks, texts, strict=True):
        words = text.lower().split()
        key = "known" if all(word in vocabulary for word in words) else "unknown"
        buckets[key].append(rank)

    split: dict[str, float] = {}
    for key, bucket in buckets.items():
        if not bucket:
            continue
        recall = sum(1 for rank in bucket if rank <= config.recall_at_k) / len(bucket)
        split[f"vocabulary_{key}_recall_at_{config.recall_at_k}"] = recall
        split[f"vocabulary_{key}_queries"] = float(len(bucket))
        logger.info(
            "queries with %s vocabulary: n=%d recall@%d %.4f median rank %.0f",
            "all words in" if key == "known" else "a word outside the",
            len(bucket),
            config.recall_at_k,
            recall,
            statistics.median(bucket),
        )
    return split


def main() -> None:
    """Entry point for `make evaluate`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print the results as JSON")
    args = parser.parse_args()
    results = evaluate(Config())
    if args.json:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
