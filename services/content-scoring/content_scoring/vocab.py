"""Extending the text8 vocabulary onto the Hacker News title corpus.

The sibling service trains skip-gram vectors on text8: 71,290 words, lowercase and
alphabetic. Hacker News titles are full of words text8 cannot contain -- `gpt-4`,
`kubernetes`, `rust`, `postgres`, `llm`, every product name of the last three years.
Scoring a title through a vocabulary that has none of them throws away most of what the
title says.

This stage adds those words without touching the word-embeddings service:

1. Count tokens over the **training** titles only. Vocabulary is fitted state; building
   it over val or test would leak the evaluation into the model.
2. Keep the pretrained ids exactly where they are, `0..71289`, and append the new words
   above them. Every existing row keeps its meaning, so the extended matrix is still
   readable by anything that understands the original.
3. Continue training skip-gram on the titles, starting from the pretrained weights.

**The negative sampler uses Hacker News counts only.** A word absent from the titles has
count zero, so it is never drawn as a negative and never appears as a centre -- its row
simply does not move. That is the behaviour a freeze mask would be written to produce,
and here it falls out of the counts instead.

Windows do not cross title boundaries. Titles average ten tokens, so a sliding window
over a concatenated stream would spend a large fraction of its context on the previous
headline -- which is why this batches `(centre, context)` pairs built per title rather
than reusing the corpus-stream batcher next door.
"""

import argparse
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import torch
from torch import optim
from tqdm import tqdm

from ai_residency_common import (
    get_device,
    get_logger,
    load_checkpoint,
    push_checkpoint,
    resolve_required,
    save_checkpoint,
    set_seed,
)
from config import Config
from word2vec import NegativeSampler, Word2Vec, Word2VecConfig

__all__ = [
    "ExtendedVocabulary",
    "build_pairs",
    "extend_vocabulary",
    "load_pretrained",
    "oov_rate",
    "read_titles",
]

logger = get_logger(__name__)


class ExtendedVocabulary:
    """The union vocabulary and the matrix it indexes.

    Attributes:
        word_to_index: Word to row, pretrained ids unchanged.
        counts: Hacker News occurrences per row; zero for text8-only words.
        num_pretrained: How many rows came from the pretrained matrix.
    """

    def __init__(
        self,
        word_to_index: dict[str, int],
        counts: torch.Tensor,
        num_pretrained: int,
    ) -> None:
        self.word_to_index = word_to_index
        self.counts = counts
        self.num_pretrained = num_pretrained

    @property
    def size(self) -> int:
        """Rows in the extended matrix."""
        return len(self.word_to_index)

    @property
    def num_added(self) -> int:
        """Words the Hacker News corpus contributed."""
        return self.size - self.num_pretrained


def read_titles(path: Path) -> list[list[str]]:
    """Read the tokenised title corpus written by the ETL.

    Args:
        path: `data/hn_titles.txt`, one whitespace-separated title per line.

    Returns:
        One token list per title.
    """
    with path.open(encoding="utf-8") as handle:
        return [line.split() for line in handle if line.strip()]


def load_pretrained(
    config: Config,
) -> tuple[dict[str, torch.Tensor], dict[str, int], Word2VecConfig]:
    """Load the sibling service's checkpoint.

    Args:
        config: Supplies the checkpoint path and repo.

    Returns:
        `(state_dict, word_to_index, word2vec_config)`.

    Raises:
        KeyError: If the checkpoint carries no vocabulary.
    """
    path, source = resolve_required(
        config.embedding_checkpoint,
        config.embedding_repo_id,
        rebuild_hint="Train it with `make -C services/word-embeddings train`.",
    )
    payload = load_checkpoint(path)
    extra = payload["extra"]
    if "word_to_index" not in extra:
        raise KeyError(
            f"{path} carries no `word_to_index`. Checkpoints written before the "
            "vocabulary fix indexed the matrix through a different id space and "
            "cannot be extended; retrain word-embeddings."
        )
    word2vec_config = Word2VecConfig(**extra["config"])
    logger.info(
        "Pretrained %s from %s: %d words, %d dims",
        word2vec_config.architecture,
        source,
        len(extra["word_to_index"]),
        word2vec_config.embed_size,
    )
    return payload["model_state_dict"], extra["word_to_index"], word2vec_config


def build_extended_vocabulary(
    titles: list[list[str]], pretrained: dict[str, int], min_count: int
) -> ExtendedVocabulary:
    """Union the pretrained vocabulary with the corpus's own, ids preserved.

    Args:
        titles: Tokenised training titles.
        pretrained: The pretrained word-to-row mapping.
        min_count: Corpus words rarer than this are not added.

    Returns:
        The extended vocabulary, with Hacker News counts per row.

    New words are appended in descending corpus frequency, so "the most common N added
    words" stays a contiguous range -- the same property the pretrained vocabulary has,
    and one the analogy benchmark's candidate restriction depends on.
    """
    corpus_counts = Counter(token for title in titles for token in title)
    word_to_index = dict(pretrained)
    num_pretrained = len(word_to_index)

    additions = sorted(
        (
            (word, count)
            for word, count in corpus_counts.items()
            if count >= min_count and word not in word_to_index
        ),
        key=lambda item: (-item[1], item[0]),
    )
    for word, _ in additions:
        word_to_index[word] = len(word_to_index)

    counts = torch.zeros(len(word_to_index), dtype=torch.long)
    for word, count in corpus_counts.items():
        index = word_to_index.get(word)
        if index is not None:
            counts[index] = count

    logger.info(
        "Vocabulary %d -> %d (+%d Hacker News words at min_count=%d); "
        "%d pretrained rows never occur in the corpus and will not move",
        num_pretrained,
        len(word_to_index),
        len(additions),
        min_count,
        int((counts[:num_pretrained] == 0).sum()),
    )
    return ExtendedVocabulary(word_to_index, counts, num_pretrained)


def oov_rate(titles: list[list[str]], word_to_index: dict[str, int]) -> float:
    """Fraction of tokens the vocabulary cannot represent.

    Args:
        titles: Tokenised titles.
        word_to_index: The vocabulary to measure against.

    Returns:
        Out-of-vocabulary tokens over total tokens. Measured per *token*, not per
        title: a title losing one word in ten is a different problem from one losing
        every word, and a per-title rate hides the difference.
    """
    total = sum(len(title) for title in titles)
    if total == 0:
        return 0.0
    known = sum(1 for title in titles for token in title if token in word_to_index)
    return 1.0 - known / total


def build_pairs(
    titles: list[list[str]], word_to_index: dict[str, int], window_size: int
) -> torch.Tensor:
    """Every `(centre, context)` pair, built within title boundaries.

    Args:
        titles: Tokenised training titles.
        word_to_index: The extended vocabulary.
        window_size: Context words either side of the centre.

    Returns:
        An `(num_pairs, 2)` long tensor of `(centre, context)` row ids.

    Out-of-vocabulary tokens are dropped rather than mapped to a shared `<UNK>` row. A
    single row absorbing every rare word becomes a high-frequency attractor that
    distorts its neighbours, which is the opposite of what an embedding is for.

    Built by offset rather than by nested loops over titles. The obvious triple loop is
    ~50 million Python-level appends on this corpus and takes minutes; this flattens the
    titles once, tags each token with the title it came from, and for each offset `d`
    keeps the positions whose partner `d` away carries the same tag. Titles never bleed
    into each other because that tag comparison is what defines a valid pair.
    """
    flat: list[int] = []
    owner: list[int] = []
    for index, title in enumerate(titles):
        ids = [word_to_index[token] for token in title if token in word_to_index]
        flat.extend(ids)
        owner.extend([index] * len(ids))

    tokens = torch.tensor(flat, dtype=torch.long)
    titles_of = torch.tensor(owner, dtype=torch.long)
    blocks: list[torch.Tensor] = []
    for offset in range(1, window_size + 1):
        if offset >= len(tokens):
            break
        left, right = tokens[:-offset], tokens[offset:]
        same_title = titles_of[:-offset] == titles_of[offset:]
        left, right = left[same_title], right[same_title]
        # Both directions: skip-gram predicts context from centre, and every pair is a
        # centre once and a context once.
        blocks.append(torch.stack([left, right], dim=1))
        blocks.append(torch.stack([right, left], dim=1))

    pairs = torch.cat(blocks) if blocks else torch.zeros((0, 2), dtype=torch.long)
    logger.info(
        "Built %d skip-gram pairs from %d titles (%d in-vocabulary tokens)",
        len(pairs),
        len(titles),
        len(tokens),
    )
    return pairs


def _keep_probabilities(counts: torch.Tensor, threshold: float) -> torch.Tensor:
    """Mikolov's frequent-word discard rule, `P(keep) = (sqrt(z/t) + 1) * (t/z)`.

    Args:
        counts: Occurrences per row.
        threshold: The `t` above.

    Returns:
        A `(vocab_size,)` probability tensor clamped to 1. Rows with zero count get
        probability zero, which is what keeps text8-only words out of the corpus.
    """
    total = counts.sum()
    frequency = counts.double() / total
    probability = torch.zeros_like(frequency)
    present = frequency > 0
    probability[present] = ((frequency[present] / threshold).sqrt() + 1) * (
        threshold / frequency[present]
    )
    return probability.clamp(max=1.0).float()


def extend_vocabulary(config: Config) -> dict[str, float]:
    """Extend and continue-train the word vectors on the title corpus.

    Args:
        config: Supplies the corpus paths, the vocabulary threshold and the schedule.

    Returns:
        Metrics: OOV before and after, vocabulary sizes, and the final loss.

    Raises:
        FileNotFoundError: If the ETL has not run.
    """
    if not config.titles_path.exists():
        raise FileNotFoundError(
            f"{config.titles_path} is missing. Run `make -C services/content-scoring etl` first."
        )
    set_seed(config.seed)
    device = get_device()

    titles = read_titles(config.titles_path)
    state_dict, pretrained_vocab, word2vec_config = load_pretrained(config)

    before = oov_rate(titles, pretrained_vocab)
    vocabulary = build_extended_vocabulary(titles, pretrained_vocab, config.vocab_min_count)
    after = oov_rate(titles, vocabulary.word_to_index)
    logger.info("OOV on training titles: %.2f%% -> %.2f%%", before * 100, after * 100)

    # The pretrained run used a window of 8 over continuous prose. Titles average ten
    # tokens, so a window that wide makes every token in a title a context for every
    # other and roughly doubles the pair count for no extra signal. Capped here rather
    # than in the config because it is a property of the corpus, not a knob.
    window_size = min(word2vec_config.window_size, config.extend_window_size)
    extended_config = Word2VecConfig(
        vocab_size=vocabulary.size,
        embed_size=word2vec_config.embed_size,
        architecture="skipgram",
        objective="negative_sampling",
        num_negatives=word2vec_config.num_negatives,
        window_size=window_size,
    )
    model = Word2Vec(extended_config).to(device)
    with torch.no_grad():
        for name in ("in_embed", "out_embed"):
            rows = state_dict[f"{name}.weight"]
            getattr(model, name).weight[: rows.shape[0]] = rows.to(device)

    if config.freeze_pretrained:
        # Zeroing the gradient rows after backward rather than splitting the matrix in
        # two: a single nn.Embedding keeps the id space contiguous, which is what lets
        # the extended checkpoint stay readable by anything that understood the
        # original.
        logger.info("Pretrained rows are frozen; only the %d new rows train", vocabulary.num_added)

    # Pairs stay on the CPU and batches move across. On this corpus there are tens of
    # millions of them -- about a gigabyte -- and parking that on an MPS device alongside
    # two 300-wide embedding matrices is how a run dies at epoch one on a laptop.
    pairs = build_pairs(titles, vocabulary.word_to_index, extended_config.window_size)
    sampler = NegativeSampler(vocabulary.counts)
    keep = _keep_probabilities(vocabulary.counts, 1e-4)
    optimizer = optim.Adam(model.parameters(), lr=config.extend_learning_rate)

    loss_value = float("nan")
    for epoch in range(1, config.extend_epochs + 1):
        # Subsampling is redrawn each epoch, so a frequent word is dropped from some
        # epochs and kept in others rather than being removed from the corpus once.
        mask = torch.rand(len(pairs)) < keep[pairs[:, 0]]
        kept = pairs[mask]
        order = torch.randperm(len(kept))
        total = 0.0
        steps = 0
        progress = tqdm(
            range(0, len(kept), config.extend_batch_size),
            desc=f"extend epoch {epoch}/{config.extend_epochs}",
            leave=False,
        )
        for start in progress:
            batch = kept[order[start : start + config.extend_batch_size]].to(device)
            centre = batch[:, 0]
            context = batch[:, 1].unsqueeze(1)
            negatives = sampler(len(batch), extended_config.num_negatives, device=device)
            loss = model(centre, context, negatives)
            optimizer.zero_grad()
            loss.backward()
            if config.freeze_pretrained:
                for name in ("in_embed", "out_embed"):
                    weight = getattr(model, name).weight
                    if weight.grad is not None:
                        weight.grad[: vocabulary.num_pretrained] = 0
            optimizer.step()
            total += float(loss.detach())
            steps += 1
        loss_value = total / max(steps, 1)
        logger.info("extend epoch %d/%d loss %.4f", epoch, config.extend_epochs, loss_value)

    config.embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        "oov_before": before,
        "oov_after": after,
        "vocab_pretrained": float(vocabulary.num_pretrained),
        "vocab_extended": float(vocabulary.size),
        "vocab_added": float(vocabulary.num_added),
        "extend_loss": loss_value,
    }
    save_checkpoint(
        config.embeddings_path,
        model,
        epoch=config.extend_epochs,
        metrics=metrics,
        extra={
            "word_to_index": vocabulary.word_to_index,
            "counts": vocabulary.counts,
            "num_pretrained": vocabulary.num_pretrained,
            "config": asdict(extended_config),
        },
    )
    if config.hf_repo_id is not None:
        push_checkpoint(config.hf_repo_id, config.embeddings_path, private=False)
    return metrics


def main() -> None:
    """Run the extension stage from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Also score the extended matrix on WordSim-353 and SimLex-999, to show "
        "that adding rows did not damage the text8 vectors.",
    )
    arguments = parser.parse_args()
    config = Config()
    metrics = extend_vocabulary(config)
    for name, value in metrics.items():
        logger.info("%-18s %.4f", name, value)

    if arguments.benchmark:
        from content_scoring.evaluate import score_extended_embeddings

        for name, value in score_extended_embeddings(config).items():
            logger.info("%-18s %.4f", name, value)


if __name__ == "__main__":
    main()
