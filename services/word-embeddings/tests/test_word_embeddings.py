"""Tests for the word-embeddings stage.

Focused on the things that actually broke here: the text-normalization table being
dead code, the model being constructed with the wrong arguments, the embeddings
carrying their vocabulary with them -- and, most of the above, the two index spaces
that used to be confused for one another.
"""

import math
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
from fastapi.testclient import TestClient

from config import Config
from serving.api import app
from word_embeddings.etl import TextProcessor
from word_embeddings.model import build_word2vec
from word_embeddings.train import train


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
    """The model comes from libs/word2vec now; this checks the service's own view of
    it -- the shape everything downstream reads, and that a single-example batch works.
    """

    def test_builds_the_matrix_everything_downstream_reads(self) -> None:
        model = build_word2vec(64, Config(embed_size=8))
        assert model.in_embed.weight.shape == (64, 8)
        assert model.embeddings.shape == (64, 8)

    def test_a_batch_of_one_works(self) -> None:
        """The previous implementation raised IndexError here.

        It scored negatives with `torch.bmm(...).squeeze()` -- no dim -- which
        collapses the batch axis along with the one it meant to remove.
        """
        model = build_word2vec(64, Config(embed_size=8, num_negatives=5))
        loss = model(
            torch.randint(0, 64, (1,)),
            torch.randint(0, 64, (1, 4)),
            torch.randint(0, 64, (1, 5)),
        )
        assert loss.shape == ()


@pytest.fixture
def loaded_state() -> Iterator[Callable[[torch.Tensor, dict[str, int]], None]]:
    """Load a vocabulary into the module-level state, and put it back afterwards.

    `serving.api.state` is a module global the app reads on every request, so a test
    that loads into it has to restore it or the next test sees the leftovers.
    """
    from serving.api import state

    saved = (
        state.embeddings,
        state.unit_embeddings,
        state.word_to_index,
        state.index_to_word,
        state.valid_rows,
        state.source,
    )

    def load(embeddings: torch.Tensor, vocabulary: dict[str, int]) -> None:
        state.load(embeddings, vocabulary, "test")

    yield load

    (
        state.embeddings,
        state.unit_embeddings,
        state.word_to_index,
        state.index_to_word,
        state.valid_rows,
        state.source,
    ) = saved


class TestApi:
    def test_health_reports_whether_the_model_loaded(self) -> None:
        response = TestClient(app).get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_root_describes_the_service(self) -> None:
        response = TestClient(app).get("/")
        assert response.status_code == 200
        assert "/nearest" in response.json()["endpoint"]

    def test_nearest_never_names_a_row_the_vocabulary_lacks(
        self, loaded_state: Callable[[torch.Tensor, dict[str, int]], None]
    ) -> None:
        """`/nearest` returned 500 in production on a vocabulary with holes.

        The checkpoint this service shipped for months had 258,196 embedding rows
        and only 230,701 of them mapped to a word. `topk` searched all of them, so a
        nameless row could win and the lookup that followed raised `KeyError`. Row 1
        below stands in for that: it ties with the query for closest, and has no word.
        """
        loaded_state(torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), {"king": 0, "apple": 2})

        response = TestClient(app).post("/nearest", json={"word": "king", "k": 2})

        assert response.status_code == 200, response.text
        assert [word for word, _ in response.json()["nearest"]] == ["apple"]

    def test_analogy_scores_the_same_way_the_benchmark_does(
        self, loaded_state: Callable[[torch.Tensor, dict[str, int]], None]
    ) -> None:
        """`/analogy` and `benchmarks.analogy_accuracy` must define 3CosAdd alike.

        The endpoint used to sum raw embeddings while the benchmark summed normalised
        ones, so the accuracy the write-up reports was measured on a slightly
        different computation from the one the service performs. Norms grow with word
        frequency, which tilts a raw sum towards whichever of the three words is
        commonest rather than towards the relation being asked about.

        `a` and `b` point the same way and differ only in length, so on unit vectors
        they cancel and the answer is `c`'s direction -- row `d`. On raw vectors `a`
        dominates the sum by being ten times longer, and the answer swings to `e`.
        """
        embeddings = torch.tensor(
            [
                [10.0, 0.0],  # a -- same direction as b, ten times the norm
                [1.0, 0.0],  # b
                [0.0, 1.0],  # c
                [0.05, 1.0],  # d -- wins on normalised vectors
                [1.0, 0.05],  # e -- wins on raw ones
            ]
        )
        loaded_state(embeddings, {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4})

        response = TestClient(app).post("/analogy", json={"a": "a", "b": "b", "c": "c", "k": 1})

        assert response.status_code == 200, response.text
        assert response.json()["nearest"][0][0] == "d"

    def test_analogy_returns_words_rather_than_the_route_function(
        self, loaded_state: Callable[[torch.Tensor, dict[str, int]], None]
    ) -> None:
        """The response field and the route handler share the name `nearest`."""
        loaded_state(torch.eye(3), {"a": 0, "b": 1, "c": 2})

        response = TestClient(app).post("/analogy", json={"a": "a", "b": "b", "c": "c", "k": 2})

        assert response.status_code == 200, response.text
        for word, score in response.json()["nearest"]:
            assert isinstance(word, str) and isinstance(score, float)


class TestConfig:
    def test_seed_is_fixed(self) -> None:
        assert Config().seed == 42

    def test_checkpoint_dir_derives_from_the_checkpoint(self) -> None:
        config = Config()
        assert config.embedding_checkpoint_dir == config.embedding_checkpoint.parent


def test_config_carries_the_hyperparameters() -> None:
    """They lived in a separate YAML read out of wandb.config until Pass 3."""
    config = Config()
    assert config.embed_size > 0
    assert config.window_size > 0
    assert config.num_negatives > 0
    assert config.min_count > 0
    assert config.embedding_checkpoint_dir == config.embedding_checkpoint.parent


def test_hyperparameters_are_overridable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A sweep sets these as environment variables; nothing else plumbs them."""
    monkeypatch.setenv("AI_RESIDENCY_EMBED_SIZE", "64")
    monkeypatch.setenv("AI_RESIDENCY_LEARNING_RATE", "0.005")
    config = Config()
    assert config.embed_size == 64
    assert config.learning_rate == 0.005


def test_architecture_is_overridable_and_validated(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Literal field could not be set from the environment until `_coerce` learned
    about Literal, which silently excluded `architecture` from every sweep."""
    monkeypatch.setenv("AI_RESIDENCY_ARCHITECTURE", "skipgram")
    assert Config().architecture == "skipgram"

    monkeypatch.setenv("AI_RESIDENCY_ARCHITECTURE", "cbowe")
    with pytest.raises(ValueError, match="not one of"):
        Config()


class TestVocabulary:
    """`corpus.build_vocabulary`, which decides the index space everything else uses."""

    def test_ids_are_ordered_by_descending_frequency(self) -> None:
        """Every figure and probe in this service slices `[:n]` for the n most
        frequent words. The previous pipeline assigned ids in first-appearance order,
        so a plot captioned "top 100 by frequency" showed the first 100 words seen."""
        from word_embeddings.corpus import build_vocabulary

        word_to_index, counts = build_vocabulary(["rare"] + ["common"] * 5 + ["mid"] * 3, 1)

        assert word_to_index == {"common": 0, "mid": 1, "rare": 2}
        assert counts.tolist() == [5, 3, 1]

    def test_min_count_drops_words_from_vocabulary_and_corpus(self) -> None:
        from word_embeddings.corpus import build_vocabulary, encode

        words = ["keep"] * 5 + ["drop"] * 2
        word_to_index, _ = build_vocabulary(words, 5)

        assert word_to_index == {"keep": 0}
        # Dropped, not mapped to <UNK>: one row standing in for every rare word
        # becomes a high-frequency token that appears beside everything and trains
        # into the centre of the space.
        assert encode(words, word_to_index).tolist() == [0] * 5

    def test_an_impossible_threshold_raises_rather_than_returning_nothing(self) -> None:
        from word_embeddings.corpus import build_vocabulary

        with pytest.raises(ValueError, match="removed all"):
            build_vocabulary(["a", "b"], 10)


class TestBatcher:
    """The batcher replaced a Dataset that materialised 16.7M Python tuples up front."""

    def test_keep_probabilities_match_the_published_formula(self) -> None:
        """P(keep) = (sqrt(z/t) + 1) * (t/z), Mikolov et al. (2013b)."""
        from word_embeddings.data import keep_probabilities

        counts = torch.tensor([600, 300, 100])
        probabilities = keep_probabilities(counts, 0.1)

        assert probabilities[0] == pytest.approx((math.sqrt(6.0) + 1) * (1 / 6), rel=1e-5)
        assert probabilities[1] == pytest.approx((math.sqrt(3.0) + 1) * (1 / 3), rel=1e-5)
        # At or below the threshold a word is always kept, never kept 1.4 times.
        assert probabilities[2] == pytest.approx(1.0)

    def test_a_zero_threshold_raises_instead_of_producing_nans(self) -> None:
        from word_embeddings.data import keep_probabilities

        with pytest.raises(ValueError, match="must be positive"):
            keep_probabilities(torch.tensor([1, 2]), 0.0)

    def test_context_surrounds_the_centre_and_excludes_it(self) -> None:
        from word_embeddings.corpus import Corpus
        from word_embeddings.data import Word2VecBatcher

        ids = torch.arange(1000) % 50
        corpus = Corpus(ids, {f"w{i}": i for i in range(50)}, torch.ones(50, dtype=torch.long))
        # Architecture pinned rather than inherited from the default: skip-gram draws
        # num_negatives per positive pair, so this shape depends on it.
        config = Config(
            architecture="cbow",
            window_size=2,
            batch_size=4,
            num_negatives=3,
            subsample_threshold=1.0,
        )

        centre, context, negatives = next(
            Word2VecBatcher(corpus, config, device=torch.device("cpu")).epoch()
        )

        assert centre.shape == (4,) and context.shape == (4, 4) and negatives.shape == (4, 3)
        for row in range(4):
            middle = int(centre[row])
            expected = [(middle + offset) % 50 for offset in (-2, -1, 1, 2)]
            assert context[row].tolist() == expected

    def test_skipgram_draws_negatives_per_positive_pair(self) -> None:
        """Skip-gram scores 2*window positives against the same negative set, so
        drawing one row's worth would leave ten positives pulling up against five
        negatives pushing down -- an imbalance CBOW does not have."""
        from word_embeddings.corpus import Corpus
        from word_embeddings.data import Word2VecBatcher

        ids = torch.arange(1000) % 50
        corpus = Corpus(ids, {f"w{i}": i for i in range(50)}, torch.ones(50, dtype=torch.long))
        shapes = {}
        for architecture in ("cbow", "skipgram"):
            config = Config(
                architecture=architecture,
                window_size=3,
                batch_size=4,
                num_negatives=5,
                subsample_threshold=1.0,
            )
            _, _, negatives = next(
                Word2VecBatcher(corpus, config, device=torch.device("cpu")).epoch()
            )
            shapes[architecture] = negatives.shape[1]

        assert shapes["cbow"] == 5
        assert shapes["skipgram"] == 5 * 6


class TestBenchmarks:
    def test_similarity_reports_coverage_alongside_the_score(self) -> None:
        """A correlation over the pairs a small vocabulary happens to cover is not
        comparable to one over nearly all of them, so the two travel together."""
        from word_embeddings.benchmarks import similarity_score

        embeddings = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
        vocabulary = {"a": 0, "b": 1, "c": 2}
        pairs = [("a", "b", 9.0), ("a", "c", 1.0), ("a", "missing", 5.0)]

        correlation, coverage = similarity_score(embeddings, vocabulary, pairs)

        assert correlation == pytest.approx(1.0)
        assert coverage == pytest.approx(2 / 3)

    def test_analogy_excludes_all_three_query_words(self) -> None:
        """`king - man + woman` lands nearest to `king` itself far more often than to
        `queen`. The previous scorer excluded none of them, so the metric it reported
        was very nearly a constant zero."""
        from word_embeddings.benchmarks import analogy_accuracy

        # `b` is the nearest row to the query vector b - a + c; `d` is the answer.
        # Without exclusion the argmax is `b` and the question scores wrong.
        embeddings = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 0.05], [0.02, 1.0]], dtype=torch.float32
        )
        vocabulary = {"a": 0, "b": 1, "c": 2, "d": 3}

        result = analogy_accuracy(embeddings, vocabulary, [("a", "b", "c", "d", True)])

        assert result["total"] == pytest.approx(1.0)
        assert result["coverage"] == pytest.approx(1.0)

    def test_untrained_embeddings_score_about_zero(self) -> None:
        """The baseline every reported number has to be read against."""
        from word_embeddings.benchmarks import similarity_score

        torch.manual_seed(0)
        vocabulary = {f"w{i}": i for i in range(200)}
        pairs = [(f"w{i}", f"w{i + 1}", float(i)) for i in range(0, 100, 2)]

        correlation, _ = similarity_score(torch.randn(200, 32), vocabulary, pairs)

        assert abs(correlation) < 0.4


class TestCheckpoint:
    def test_training_runs_and_is_reproducible(self, tmp_path: Any) -> None:
        from word_embeddings.train import train

        def once() -> float:
            return train(_tiny_config(tmp_path), corpus=_tiny_corpus())["score"]

        assert once() == pytest.approx(once())

    def test_checkpoint_carries_the_vocabulary_its_rows_are_indexed_by(self, tmp_path: Any) -> None:
        """The bug this rewrite exists to fix: the corpus was encoded with
        `word_to_index` while the checkpoint saved `word_to_lemma_index`, a different
        index space, so every neighbour served was read through the wrong mapping."""
        from ai_residency_common import load_checkpoint

        config = _tiny_config(tmp_path)
        corpus = _tiny_corpus()
        train(config, corpus=corpus)

        payload = load_checkpoint(config.embedding_checkpoint)
        vocabulary = payload["extra"]["word_to_index"]

        assert vocabulary == corpus.word_to_index
        assert payload["model_state_dict"]["in_embed.weight"].shape[0] == len(vocabulary)
        # Dense and total: every row can be named, so nearest-neighbour search has no
        # holes to fall into.
        assert sorted(vocabulary.values()) == list(range(len(vocabulary)))

    def test_reader_prefers_the_new_key_and_falls_back_to_the_old(self) -> None:
        from word_embeddings.evaluate import checkpoint_vocabulary

        assert checkpoint_vocabulary({"word_to_index": {"a": 0}}) == {"a": 0}
        assert checkpoint_vocabulary({"word_to_lemma_index": {"b": 0}}) == {"b": 0}
        with pytest.raises(KeyError):
            checkpoint_vocabulary({})


def _tiny_corpus() -> Any:
    from word_embeddings.corpus import Corpus

    # Real benchmark words, so the tiny run exercises the scoring path rather
    # than short-circuiting on an empty intersection.
    words = ("king queen man woman car cat dog tiger book computer " * 40).split()
    from word_embeddings.corpus import build_vocabulary, encode

    word_to_index, counts = build_vocabulary(words, 1)
    return Corpus(encode(words, word_to_index), word_to_index, counts)


def _tiny_config(tmp_path: Any) -> Config:
    return Config(
        embed_size=8,
        window_size=2,
        num_negatives=2,
        batch_size=8,
        num_epochs=1,
        eval_every=4,
        subsample_threshold=1.0,
        checkpoint_dir=tmp_path,
        eval_dir=Path("data/eval"),
        hf_repo_id=None,
        wandb_project=None,
    )
