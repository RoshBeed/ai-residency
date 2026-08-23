"""Tests for the word-embeddings stage.

Focused on the things that actually broke here: the text-normalization table being
dead code, the CBOW model being constructed with the wrong arguments, and the
embeddings carrying their vocabulary with them.
"""

from typing import Any

import pytest
from fastapi.testclient import TestClient

from config import Config
from serving.api import app
from word_embeddings.etl import TextProcessor
from word_embeddings.model import CBOWNS, CBOWConfig


class TestTextNormalization:
    """The replacement table used to be unreachable: an ASCII strip ran first and
    deleted every character it matches, so nothing in it could ever fire."""

    def test_currency_and_dashes_are_translated_not_dropped(self) -> None:
        assert TextProcessor().normalize_text("cost: €20 – £15") == "cost: EUR20 - GBP15"

    def test_math_symbols_are_translated(self) -> None:
        processor = TextProcessor()
        assert processor.normalize_text("5 × 3") == "5 x 3"
        assert processor.normalize_text("a ≤ b") == "a <= b"

    def test_unmapped_non_ascii_is_still_dropped(self) -> None:
        assert "漢" not in TextProcessor().normalize_text("hello 漢 world")


class TestCbowModel:
    """CBOWNS takes a CBOWConfig. Calling it with loose ints raised TypeError at
    runtime, which is how the prediction path was broken."""

    def test_constructs_from_a_config(self) -> None:
        model = CBOWNS(CBOWConfig(vocab_size=64, embed_size=8))
        assert model.in_embed.weight.shape == (64, 8)

    def test_rejects_positional_ints(self) -> None:
        # Typed as Any so this stays a runtime contract test rather than something
        # mypy rejects at check time — no suppression needed.
        model_class: Any = CBOWNS
        with pytest.raises(TypeError):
            model_class(64, 8)


class TestApi:
    def test_health_reports_whether_the_model_loaded(self) -> None:
        response = TestClient(app).get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_root_describes_the_service(self) -> None:
        response = TestClient(app).get("/")
        assert response.status_code == 200
        assert "/nearest" in response.json()["endpoint"]


class TestConfig:
    def test_seed_is_fixed(self) -> None:
        assert Config().seed == 42

    def test_cbow_checkpoint_dir_derives_from_the_checkpoint(self) -> None:
        config = Config()
        assert config.cbow_checkpoint_dir == config.cbow_checkpoint.parent


def test_config_carries_the_cbow_hyperparameters() -> None:
    """They lived in a separate YAML read out of wandb.config until Pass 3."""
    config = Config()
    assert config.cbow_embed_size > 0
    assert config.cbow_window_size > 0
    assert config.cbow_num_negatives > 0
    assert config.cbow_checkpoint_dir == config.cbow_checkpoint.parent


def test_cbow_hyperparameters_are_overridable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A sweep sets these as environment variables; nothing else plumbs them."""
    monkeypatch.setenv("AI_RESIDENCY_CBOW_EMBED_SIZE", "64")
    monkeypatch.setenv("AI_RESIDENCY_CBOW_LEARNING_RATE", "0.005")
    config = Config()
    assert config.cbow_embed_size == 64
    assert config.cbow_learning_rate == 0.005


def test_sample_vocabulary_is_self_consistent() -> None:
    from word_embeddings.data import (
        UNKNOWN_TOKEN,
        encode_corpus,
        sample_vocabulary,
    )

    word_to_index, word_to_lemma_index, text = sample_vocabulary()
    assert UNKNOWN_TOKEN in word_to_index
    assert word_to_index == word_to_lemma_index

    encoded = encode_corpus(text, word_to_index)
    assert encoded, "the sample corpus must encode to something"
    assert max(encoded) < len(word_to_index)


def test_encode_corpus_maps_unseen_words_to_unknown() -> None:
    from word_embeddings.data import (
        UNKNOWN_TOKEN,
        encode_corpus,
    )

    vocabulary = {"python": 0, UNKNOWN_TOKEN: 1}
    assert encode_corpus("python haskell", vocabulary) == [0, 1]


def test_evaluation_skips_checks_the_vocabulary_cannot_support() -> None:
    """A real corpus need not contain "queen"; scoring 0.0 would read as failure."""
    import torch

    from word_embeddings.evaluate import evaluate_embeddings

    metrics = evaluate_embeddings(torch.randn(2, 8), {"python": 0, "rust": 1})
    assert metrics == {}


def test_evaluation_scores_a_vocabulary_that_supports_it() -> None:
    import torch

    from word_embeddings.data import sample_vocabulary
    from word_embeddings.evaluate import evaluate_embeddings

    word_to_index, _, _ = sample_vocabulary()
    metrics = evaluate_embeddings(torch.randn(len(word_to_index), 8), word_to_index)
    assert "analogy_correct" in metrics
    assert "similarity_spearman" in metrics


def test_cbow_training_runs_and_is_reproducible(tmp_path: Any) -> None:
    """The whole stage, end to end, on the sample corpus. Twice, identically."""
    from word_embeddings.train import train_cbow

    def run() -> float:
        config = Config(
            cbow_checkpoint=tmp_path / "cbow.pt",
            cbow_num_epochs=1,
            cbow_batch_size=8,
            hf_repo_id=None,
            wandb_project=None,
        )
        return train_cbow(config, sample=True)

    first = run()
    assert (tmp_path / "cbow.pt").exists()
    assert first == run(), "seeded training must reproduce"


def test_cbow_checkpoint_carries_its_vocabulary(tmp_path: Any) -> None:
    """Embeddings are meaningless without the mapping they were trained against."""
    from ai_residency_common import load_checkpoint

    from word_embeddings.train import train_cbow

    config = Config(
        cbow_checkpoint=tmp_path / "cbow.pt",
        cbow_num_epochs=1,
        cbow_batch_size=8,
        hf_repo_id=None,
        wandb_project=None,
    )
    train_cbow(config, sample=True)

    payload = load_checkpoint(config.cbow_checkpoint)
    vocabulary = payload["extra"]["word_to_lemma_index"]
    assert vocabulary
    assert payload["model_state_dict"]["in_embed.weight"].shape[0] == len(vocabulary)
