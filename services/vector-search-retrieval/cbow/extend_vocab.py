"""Stage 1b: extend the text8 vocabulary onto the corpus being searched.

The vocabulary is built from text8, a 2006 Wikipedia dump, and the queries are MS MARCO.
That mismatch is the measured ceiling on this service: **1,506 of 5,034 held-out queries
contain a word text8 never saw, and they score 0.4309 recall@10 against 0.7761 for the
3,528 it covers** -- median rank 20 against 2. Swapping the loss function moved both
columns and closed none of the gap, so it is a property of the representation rather
than of the objective.

`tokenize_text` maps an unknown word to token 0, which is also the padding id, so the
most discriminative word in a query -- `fridge`, `creatinine`, `quizlet`, `iphone` --
arrives at the GRU indistinguishable from padding. Those words are not obscure. They are
simply not in a Wikipedia dump from 2006, and they occur in the hundreds in the corpus
this service actually searches.

The fix is the one `content-scoring` already uses on Hacker News titles, and it works
the same way here:

1. Count tokens over the **training** triples only. Vocabulary is fitted state; counting
   over validation or test would leak the evaluation into the model.
2. Keep the pretrained ids exactly where they are and append the new words above them,
   so every existing row keeps its meaning and the extended matrix is still readable by
   anything that understands the original.
3. Continue training CBOW on this corpus, starting from the pretrained weights.

**A word absent from the new corpus does not move.** CBOW reads `in_embed` for context
words only, so a row is only touched when its word appears in somebody's context window.
Checked rather than assumed: of the 15,420 text8 words MS MARCO never uses, 15,419 come
out of the continuation bit-identical. The one exception is `<PAD>` itself, which moves
because it *is* context -- it fills the window at a document's edges. No freeze mask is
needed to arrange any of this.

Windows do not cross document boundaries. A passage and the passage after it have
nothing to do with each other, so a sliding window over a concatenated stream would
spend part of its context on the wrong document -- which is why this indexes each
document's span and clips to it.
"""

import argparse
import collections
import json
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ai_residency_common import (
    finish_tracking,
    get_device,
    get_logger,
    init_tracking,
    load_checkpoint,
    log_metrics,
    push_checkpoint,
    save_checkpoint,
    set_seed,
)
from cbow.evaluate import nearest_words
from cbow.model import build_cbow
from cbow.vocabulary import (
    CANONICAL_CBOW_CHECKPOINT,
    latest_cbow_checkpoint,
    load_words_to_ids,
    normalise_words,
)
from config import Config

__all__ = ["Corpus", "count_training_tokens", "extend_vocabulary", "main", "train_extension"]

logger = get_logger(__name__)

Example = tuple[torch.Tensor, torch.Tensor]


def count_training_tokens(config: Config) -> tuple[collections.Counter[str], list[list[str]]]:
    """Count words over the training split, and return the documents they came from.

    Both documents of every training triple and the training queries. The negative
    document is included because it is a real passage from the same corpus -- it is only
    "negative" with respect to one query.

    Args:
        config: Supplies `triples_path`.

    Returns:
        The counts, and every training text as a list of words, in corpus order.
    """
    with config.triples_path.open() as handle:
        triples = json.load(handle)

    counts = collections.Counter[str]()
    texts: list[list[str]] = []
    seen: set[str] = set()
    for row in triples["train"]:
        for field in ("query", "positive_doc", "negative_doc"):
            text = row[field]
            if text in seen:
                continue
            seen.add(text)
            # The same normalisation every other path uses, or the vocabulary fills up
            # with `however,` and `cells.` -- tokens that only exist because a word was
            # next to a comma, and that no lookup would ever produce again.
            words = normalise_words(text)
            counts.update(words)
            texts.append(words)
    logger.info(
        "Training corpus: %d distinct texts, %d tokens, %d distinct words",
        len(texts),
        sum(counts.values()),
        len(counts),
    )
    return counts, texts


def extend_vocabulary(
    words_to_ids: dict[str, int], counts: collections.Counter[str], min_count: int
) -> tuple[dict[str, int], dict[int, str], int]:
    """Append the corpus's own frequent words above the pretrained ids.

    The cutoff matters more than it looks. A word added at count 3 gets three context
    windows and ends up with a near-random vector, which is no better than the padding
    row it replaces; the cutoff is `min_word_count`, the same one text8 was filtered
    with, so both halves of the vocabulary clear the same bar.

    Args:
        words_to_ids: The pretrained vocabulary, whose ids are preserved exactly.
        counts: Word counts over the corpus being added.
        min_count: Keep a new word only if it occurs at least this many times.

    Returns:
        The extended `words_to_ids`, its inverse, and how many words were added.
    """
    extended = dict(words_to_ids)
    added = sorted(
        (word for word, count in counts.items() if count >= min_count and word not in extended),
        key=lambda word: (-counts[word], word),
    )
    for word in added:
        extended[word] = len(extended)
    ids_to_words = {index: word for word, index in extended.items()}
    logger.info(
        "Vocabulary %d -> %d words (+%d at count >= %d)",
        len(words_to_ids),
        len(extended),
        len(added),
        min_count,
    )
    if added:
        logger.info("Most frequent additions: %s", ", ".join(added[:12]))
    return extended, ids_to_words, len(added)


class Corpus(Dataset[Example]):
    """Training texts as CBOW `(context, target)` pairs, clipped per document.

    Attributes:
        tokens: Every in-vocabulary token, concatenated in corpus order.
        starts: Where each document begins in `tokens`, for clipping windows.
        context_size: Words of context taken from each side of the target.
    """

    def __init__(self, texts: list[list[str]], words_to_ids: dict[str, int], context_size: int):
        """Flatten the texts, dropping words the extended vocabulary still lacks.

        Args:
            texts: Each training text as a list of words.
            words_to_ids: The extended vocabulary.
            context_size: Words of context on each side.
        """
        tokens: list[int] = []
        starts: list[int] = []
        for words in texts:
            ids = [words_to_ids[word] for word in words if word in words_to_ids]
            if len(ids) < 2:
                continue
            starts.append(len(tokens))
            tokens.extend(ids)
        self.tokens = np.asarray(tokens, dtype=np.int32)
        self.starts = np.asarray(starts, dtype=np.int64)
        self.context_size = context_size
        logger.info("Corpus: %d documents, %d in-vocabulary tokens", len(starts), len(tokens))

    def __len__(self) -> int:
        """One training example per token."""
        return int(self.tokens.shape[0])

    def _span(self, index: int) -> tuple[int, int]:
        """The start and end of the document `index` falls in."""
        which = int(np.searchsorted(self.starts, index, side="right")) - 1
        start = int(self.starts[which])
        end = int(self.starts[which + 1]) if which + 1 < len(self.starts) else len(self.tokens)
        return start, end

    def __getitem__(self, index: int) -> Example:
        """Return the context window around `index`, padded at document edges.

        Args:
            index: Position of the target word in the flattened corpus.

        Returns:
            A `(context, target)` pair of tensors. Padding is token 0, which the
            vocabulary reserves for `<PAD>`, the same convention the text8 stage uses.
        """
        size = self.context_size
        start, end = self._span(index)
        target = int(self.tokens[index])
        previous = [int(t) for t in self.tokens[max(index - size, start) : index]]
        following = [int(t) for t in self.tokens[index + 1 : min(index + 1 + size, end)]]
        previous = [0] * (size - len(previous)) + previous
        following = following + [0] * (size - len(following))
        return torch.tensor(previous + following), torch.tensor([target])


def train_extension(config: Config, epochs: int) -> float:
    """Extend the vocabulary and continue training CBOW on the search corpus.

    Args:
        config: Run configuration.
        epochs: Passes over the corpus being added.

    Returns:
        The mean loss of the final epoch.
    """
    set_seed(config.seed)
    device = get_device()

    pretrained_vocabulary = load_words_to_ids(config)
    counts, texts = count_training_tokens(config)
    words_to_ids, ids_to_words, added = extend_vocabulary(
        pretrained_vocabulary, counts, config.min_word_count
    )

    model = build_cbow(len(words_to_ids), config.embedding_dim)
    checkpoint_path = latest_cbow_checkpoint(config)
    state = load_checkpoint(checkpoint_path)["model_state_dict"]
    with torch.no_grad():
        for name in ("in_embed", "out_embed"):
            pretrained = state[f"{name}.weight"]
            getattr(model, name).weight[: pretrained.shape[0]].copy_(pretrained)
    logger.info(
        "Seeded %d pretrained rows from %s; %d new rows keep their initialisation",
        state["in_embed.weight"].shape[0],
        checkpoint_path,
        added,
    )
    model = model.to(device)

    dataset = Corpus(texts, words_to_ids, config.context_size)
    loader: DataLoader[Example] = DataLoader(dataset, batch_size=config.cbow_batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.cbow_learning_rate)

    init_tracking(
        config.wandb_project,
        config={"stage": "cbow-extend", "added_words": added, **vars(config)},
        run_name="cbow-extend",
    )

    epoch_loss = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        for index, (context, target) in enumerate(
            tqdm(loader, desc=f"Extend epoch {epoch}", leave=False)
        ):
            context, target = context.to(device), target.to(device)
            optimizer.zero_grad()
            loss = model(target.squeeze(1), context)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batches += 1
            if index % config.cbow_eval_every == 0:
                logger.info(
                    "Nearest to %r: %s",
                    config.cbow_eval_word,
                    nearest_words(model, config.cbow_eval_word, words_to_ids, ids_to_words),
                )

        epoch_loss = total_loss / max(batches, 1)
        log_metrics({"cbow_extend_loss": epoch_loss}, step=epoch)
        logger.info("Extend epoch %d/%d  loss=%.4f", epoch, epochs, epoch_loss)

        # The vocabulary and the matrix are one artifact: a reader that has one without
        # the other indexes a matrix through a mapping built for a different one, which
        # is the bug this service already shipped once. They are written together, every
        # epoch, so there is no window where the pair on disk disagrees.
        with config.words_to_ids_path.open("wb") as handle:
            pickle.dump(words_to_ids, handle)
        with config.ids_to_words_path.open("wb") as handle:
            pickle.dump(ids_to_words, handle)
        save_checkpoint(
            config.cbow_checkpoint_dir / CANONICAL_CBOW_CHECKPOINT,
            model,
            optimizer=optimizer,
            epoch=epoch,
            metrics={"cbow_extend_loss": epoch_loss},
            extra={
                "vocab_size": len(words_to_ids),
                "embedding_dim": config.embedding_dim,
                "pretrained_rows": int(state["in_embed.weight"].shape[0]),
                "added_words": added,
            },
        )

    if config.hf_repo_id is not None:
        push_checkpoint(
            config.hf_repo_id,
            config.cbow_checkpoint_dir / CANONICAL_CBOW_CHECKPOINT,
            path_in_repo=CANONICAL_CBOW_CHECKPOINT,
        )
    if config.hf_dataset_repo_id is not None:
        for path in (config.words_to_ids_path, config.ids_to_words_path):
            push_checkpoint(config.hf_dataset_repo_id, path, repo_type="dataset")

    finish_tracking()
    return epoch_loss


def main() -> None:
    """Entry point for `make extend-vocab`."""
    parser = argparse.ArgumentParser(description="Extend the vocabulary onto the search corpus")
    parser.add_argument("--epochs", type=int, default=2, help="passes over the corpus being added")
    args = parser.parse_args()
    train_extension(Config(), args.epochs)


if __name__ == "__main__":
    main()
