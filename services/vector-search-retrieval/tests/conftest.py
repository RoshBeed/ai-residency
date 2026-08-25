"""Shared fixtures.

The tests build their own tiny corpus, vocabulary and CBOW checkpoint rather than reading
the real gitignored artifacts, so the suite runs on a fresh clone with nothing trained.
"""

import json
import pickle
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

from ai_residency_common import save_checkpoint
from cbow.model import build_cbow
from config import Config

VOCABULARY = ["<PAD>", "reagan", "democrat", "was", "a", "party", "rust", "nails", "shower"]


@pytest.fixture
def config(tmp_path: Path) -> Config:
    """A configuration pointing at a throwaway directory, small enough to train in a test."""
    return Config(
        data_dir=tmp_path / "data",
        checkpoint_dir=tmp_path / "checkpoints",
        embedding_dim=8,
        hidden_size=8,
        num_epochs=2,
        cbow_num_epochs=1,
        cbow_batch_size=4,
        # Both Hub repos off: these tests assert local behaviour, and a loader that
        # silently reaches the network would make them slow, flaky and dependent on
        # what happens to be published.
        hf_repo_id=None,
        hf_dataset_repo_id=None,
        wandb_project=None,
    )


@pytest.fixture
def vocabulary(config: Config) -> dict[str, int]:
    """Write a tiny vocabulary and corpus to `config.data_dir`."""
    words_to_ids = {word: index for index, word in enumerate(VOCABULARY)}
    ids_to_words = {index: word for word, index in words_to_ids.items()}

    config.data_dir.mkdir(parents=True, exist_ok=True)
    with config.corpus_path.open("wb") as handle:
        pickle.dump(VOCABULARY[1:] * 4, handle)
    with config.words_to_ids_path.open("wb") as handle:
        pickle.dump(words_to_ids, handle)
    with config.ids_to_words_path.open("wb") as handle:
        pickle.dump(ids_to_words, handle)
    return words_to_ids


@pytest.fixture
def cbow_checkpoint(config: Config, vocabulary: dict[str, int]) -> Path:
    """Write a CBOW checkpoint in the shared format, as `train_word2vec` would."""
    torch.manual_seed(0)
    model = build_cbow(len(vocabulary), config.embedding_dim)

    # Stand in for *trained* vectors, not freshly initialised ones. word2vec starts
    # in_embed uniform in +/- 0.5/embed_size -- deliberately tiny, so the first updates
    # are not fighting noise -- and a frozen matrix of near-zero vectors gives the
    # towers downstream almost nothing to separate. Real CBOW output has spread, so the
    # fixture gives it spread.
    with torch.no_grad():
        model.in_embed.weight.normal_(0.0, 1.0)

    path = config.cbow_checkpoint_dir / "2026_01_01__00_00_00.1.cbow.pth"
    save_checkpoint(path, model, epoch=1, metrics={"cbow_loss": 1.0})
    return path


def _triple(query: list[int], positive: list[int], negative: list[int]) -> dict[str, object]:
    """Build one tokenized triple in the shape `tokenize_triples` writes."""
    return {
        "query_tokens": query,
        "positive_document_tokens": positive,
        "negative_document_tokens": negative,
        "query": "was reagan a democrat",
        "positive_document": "reagan was a democrat before switching party",
        "negative_document": "rust forms on nails left in a wet shower",
    }


@pytest.fixture
def triples(config: Config, vocabulary: dict[str, int]) -> Iterator[Path]:
    """Write a small set of tokenized triples across all three splits."""
    positive = [vocabulary[word] for word in ("reagan", "was", "a", "democrat", "party")]
    negative = [vocabulary[word] for word in ("rust", "nails", "shower")]
    query = [vocabulary[word] for word in ("was", "reagan", "a", "democrat")]

    data = {split: [_triple(query, positive, negative)] * 6 for split in ("train", "validation")}
    data["test"] = [_triple(query, negative, positive)]

    config.data_dir.mkdir(parents=True, exist_ok=True)
    with config.tokenized_triples_path.open("w") as handle:
        json.dump(data, handle)
    yield config.tokenized_triples_path
