"""Tests for the Hacker News score predictor.

These assert that the code is correct. How *good* the model is comes from
`make evaluate`, which is a different question and a different command.

The tests that matter most here are the ones about leakage and about determinism. A
score predictor that quietly sees the future scores beautifully and is worthless, and
the failure leaves no trace in a loss curve -- so the causal author feature and the
train-only fitting each get a test that would fail loudly if someone simplified them.
"""

import numpy as np
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient

from config import Config, Fusion
from content_scoring.etl import (
    clean_frame,
    normalise_text,
    registered_domain,
    split_by_time,
    tokenise,
)
from content_scoring.evaluate import _ndcg_at_k, _spearman, evaluate_predictions
from content_scoring.features import (
    COLUMN_ORDER,
    FEATURE_GROUPS,
    GROUP_COLUMNS,
    GROUP_SLICES,
    author_history,
    build_features,
    fit_feature_spec,
)
from content_scoring.model import (
    LateFusionScorer,
    build_model,
    load_predictor_weights,
    predictor_state_dict,
)
from content_scoring.vocab import build_extended_vocabulary, build_pairs, oov_rate
from serving.api import app


def _frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    """A cleaned frame from literal rows, with the columns the feature code needs."""
    frame = pd.DataFrame(rows)
    frame["tokens"] = frame["title"].map(lambda title: tokenise(normalise_text(str(title))))
    frame["domain"] = frame["url"].map(registered_domain)
    frame["posted_at"] = pd.to_datetime(frame["time"], unit="s", utc=True)
    frame["comments"] = 0
    return frame.sort_values(["time", "id"]).reset_index(drop=True)


class TestTextNormalization:
    """The replacement table used to be unreachable: an ASCII strip ran first and
    deleted every character it matches, so nothing in it could ever fire."""

    def test_currency_and_dashes_are_translated_not_dropped(self) -> None:
        assert normalise_text("cost: €20 – £15") == "cost: EUR20 - GBP15"

    def test_math_symbols_are_translated(self) -> None:
        assert normalise_text("5 × 3") == "5 x 3"
        assert normalise_text("a ≤ b") == "a <= b"

    def test_unmapped_non_ascii_is_still_dropped(self) -> None:
        assert "漢" not in normalise_text("hello 漢 world")

    def test_html_entities_are_unescaped_first(self) -> None:
        """Hacker News serves `&#x27;` and `&amp;` inside titles."""
        assert normalise_text("Ben &amp; Jerry&#x27;s") == "Ben & Jerry's"

    def test_smart_quotes_do_not_weld_words_together(self) -> None:
        """Dropping the apostrophe rather than mapping it turns `don't` into `dont`."""
        assert "don't" in normalise_text("why you don’t need Kubernetes")


class TestTokenisation:
    """text8 is lowercase alphabetic, so these are exactly the tokens the extension
    stage exists to add. Splitting on every non-letter would discard them."""

    @pytest.mark.parametrize(
        ("title", "expected"),
        [
            ("GPT-4 is out", ["gpt-4", "is", "out"]),
            ("Writing C++ in 2026", ["writing", "c++", "in", "2026"]),
            ("Node.js 24", ["node.js", "24"]),
            ("Deploying to K8s", ["deploying", "to", "k8s"]),
        ],
    )
    def test_technical_tokens_survive_whole(self, title: str, expected: list[str]) -> None:
        assert tokenise(normalise_text(title)) == expected


class TestDomains:
    def test_www_is_stripped(self) -> None:
        assert registered_domain("https://www.github.com/x") == "github.com"

    def test_a_text_post_is_a_category_not_a_gap(self) -> None:
        assert registered_domain("") == "<none>"
        assert registered_domain(None) == "<none>"


class TestEtl:
    def test_rows_are_deduplicated_on_id_keeping_the_later_score(self) -> None:
        frame = pd.DataFrame(
            {
                "id": [1, 1, 2],
                "title": ["A post", "A post", "Another"],
                "url": ["https://a.com", "https://a.com", ""],
                "score": [3, 40, 5],
                "time": [100, 200, 150],
                "author": ["x", "x", "y"],
                "comments": [0, 0, 0],
            }
        )
        cleaned = clean_frame(frame)
        assert len(cleaned) == 2
        assert int(cleaned.loc[cleaned["id"] == 1, "score"].iloc[0]) == 40

    def test_rows_are_ordered_by_time_then_id(self) -> None:
        """The id tie-break is what makes the author feature reproducible: posts sharing
        a timestamp would otherwise be ordered arbitrarily between machines."""
        frame = pd.DataFrame(
            {
                "id": [9, 3, 7],
                "title": ["c", "a", "b"],
                "url": ["", "", ""],
                "score": [1, 1, 1],
                "time": [100, 100, 100],
                "author": ["x", "y", "z"],
                "comments": [0, 0, 0],
            }
        )
        assert clean_frame(frame)["id"].tolist() == [3, 7, 9]

    def test_the_split_is_chronological_and_disjoint(self) -> None:
        frame = _frame(
            [
                {"id": i, "title": f"post {i}", "url": "", "score": 1, "time": t, "author": "a"}
                for i, t in enumerate(
                    [
                        int(pd.Timestamp(day, tz="UTC").timestamp())
                        for day in ("2026-01-01", "2026-04-01", "2026-07-01")
                    ]
                )
            ]
        )
        splits = split_by_time(frame, "2026-03-01", "2026-06-01")
        assert [len(split) for split in splits.values()] == [1, 1, 1]
        assert splits["train"]["posted_at"].max() < splits["val"]["posted_at"].min()
        assert splits["val"]["posted_at"].max() < splits["test"]["posted_at"].min()
        ids = [set(split["id"]) for split in splits.values()]
        assert not (ids[0] & ids[1] or ids[1] & ids[2] or ids[0] & ids[2])


class TestAuthorLeakage:
    """The single most important test in this file.

    An author's mean score is the strongest signal on Hacker News and the easiest way to
    leak the target. These assert the window is strictly backwards -- a later post
    cannot influence an earlier row's feature, and a row cannot influence its own.
    """

    def test_a_row_does_not_see_its_own_score(self) -> None:
        frame = _frame(
            [
                {"id": 1, "title": "first", "url": "", "score": 1, "time": 100, "author": "a"},
                {"id": 2, "title": "second", "url": "", "score": 500, "time": 200, "author": "a"},
            ]
        )
        history = author_history(frame)
        assert np.isnan(history["author_mean"].iloc[0])
        assert history["author_mean"].iloc[1] == pytest.approx(np.log1p(1))

    def test_a_later_post_cannot_change_an_earlier_rows_feature(self) -> None:
        """The regression test for the leak. If the expanding window ever loses its
        `shift(1)`, or is computed over the whole frame rather than forward in time,
        appending a future post changes the past and this fails."""
        rows: list[dict[str, object]] = [
            {"id": 1, "title": "first", "url": "", "score": 2, "time": 100, "author": "a"},
            {"id": 2, "title": "second", "url": "", "score": 3, "time": 200, "author": "a"},
        ]
        before = author_history(_frame(rows))["author_mean"].to_numpy(dtype=float)
        rows.append(
            {"id": 3, "title": "viral", "url": "", "score": 5000, "time": 300, "author": "a"}
        )
        after = author_history(_frame(rows))["author_mean"].to_numpy(dtype=float)
        # `equal_nan`: the first row legitimately has no history, and NaN != NaN.
        assert np.array_equal(after[:2], before, equal_nan=True)

    def test_a_first_post_is_flagged_rather_than_silently_zero(self) -> None:
        frame = _frame(
            [{"id": 1, "title": "hello", "url": "", "score": 9, "time": 100, "author": "new"}]
        )
        history = author_history(frame)
        assert history["author_is_first"].iloc[0] == 1.0
        assert history["author_count"].iloc[0] == 0.0

    def test_authors_do_not_see_each_others_history(self) -> None:
        frame = _frame(
            [
                {"id": 1, "title": "one", "url": "", "score": 900, "time": 100, "author": "a"},
                {"id": 2, "title": "two", "url": "", "score": 1, "time": 200, "author": "b"},
            ]
        )
        assert np.isnan(author_history(frame)["author_mean"].iloc[1])


class TestFeatureLayout:
    """`GROUP_SLICES` is what each late-fusion branch reads. If it disagrees with the
    columns actually emitted, every branch silently reads its neighbour's features."""

    def test_the_groups_partition_the_columns_exactly_once(self) -> None:
        flattened = [column for group in FEATURE_GROUPS for column in GROUP_COLUMNS[group]]
        assert flattened == list(COLUMN_ORDER)
        assert len(set(flattened)) == len(flattened)

    def test_the_slices_cover_the_block_without_overlapping(self) -> None:
        covered: list[int] = []
        for group in FEATURE_GROUPS:
            covered.extend(range(*GROUP_SLICES[group].indices(len(COLUMN_ORDER))))
        assert covered == list(range(len(COLUMN_ORDER)))


class TestFeatureFitting:
    def _split(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        train = _frame(
            [
                {
                    "id": i,
                    "title": f"Show HN: thing {i}",
                    "url": "https://github.com/x" if i % 2 else "",
                    "score": i + 1,
                    "time": 1_700_000_000 + i * 3600,
                    "author": f"user{i % 3}",
                }
                for i in range(120)
            ]
        )
        test = _frame(
            [
                {
                    "id": 900,
                    "title": "A brand new thing",
                    "url": "https://never-seen-before.example/x",
                    "score": 7,
                    "time": 1_800_000_000,
                    "author": "stranger",
                }
            ]
        )
        return train, test

    def test_scaling_uses_training_statistics_only(self) -> None:
        """The test row is scaled by the training mean and std, so its scaled values are
        not centred on itself -- which is what would happen if the spec were refitted."""
        train, test = self._split()
        spec = fit_feature_spec(train, Config(min_domain_count=5))
        numeric, _ = build_features(test, spec)
        assert numeric.shape == (1, len(COLUMN_ORDER))
        assert not np.allclose(numeric, 0.0)

    def test_an_unseen_domain_falls_into_the_rare_bucket(self) -> None:
        train, test = self._split()
        spec = fit_feature_spec(train, Config(min_domain_count=5))
        _, domains = build_features(test, spec)
        assert domains.tolist() == [0]

    def test_a_constant_column_cannot_divide_by_zero(self) -> None:
        train, _ = self._split()
        spec = fit_feature_spec(train, Config(min_domain_count=5))
        assert (spec.stds > 0).all()
        numeric, _ = build_features(train, spec)
        assert np.isfinite(numeric).all()


class TestVocabularyExtension:
    def test_pretrained_ids_are_preserved_exactly(self) -> None:
        """Every downstream reader indexes the matrix by these ids. If the extension
        renumbered them, every pretrained vector would silently mean a different word."""
        pretrained = {"the": 0, "quick": 1, "brown": 2}
        titles = [["gpt-4", "the"], ["gpt-4", "kubernetes"], ["gpt-4", "kubernetes"]]
        vocabulary = build_extended_vocabulary(titles, pretrained, min_count=2)
        for word, index in pretrained.items():
            assert vocabulary.word_to_index[word] == index
        assert vocabulary.num_pretrained == 3

    def test_new_words_are_appended_in_descending_frequency(self) -> None:
        pretrained = {"the": 0}
        titles = [["rare", "common"], ["common"], ["common"]]
        vocabulary = build_extended_vocabulary(titles, pretrained, min_count=1)
        assert vocabulary.word_to_index["common"] < vocabulary.word_to_index["rare"]

    def test_words_below_min_count_are_not_added(self) -> None:
        vocabulary = build_extended_vocabulary([["hapax"]], {"the": 0}, min_count=2)
        assert "hapax" not in vocabulary.word_to_index

    def test_counts_are_corpus_counts_so_absent_rows_never_train(self) -> None:
        """A text8 word absent from the titles gets count zero, so the negative sampler
        never draws it and its row does not move. That is the freeze, for free."""
        vocabulary = build_extended_vocabulary([["gpt-4"]], {"the": 0, "aardvark": 1}, min_count=1)
        assert int(vocabulary.counts[vocabulary.word_to_index["aardvark"]]) == 0
        assert int(vocabulary.counts[vocabulary.word_to_index["gpt-4"]]) == 1

    def test_extending_lowers_the_out_of_vocabulary_rate(self) -> None:
        titles = [["gpt-4", "the"], ["gpt-4", "the"]]
        pretrained = {"the": 0}
        vocabulary = build_extended_vocabulary(titles, pretrained, min_count=1)
        assert oov_rate(titles, pretrained) == pytest.approx(0.5)
        assert oov_rate(titles, vocabulary.word_to_index) == pytest.approx(0.0)

    def test_pairs_do_not_cross_title_boundaries(self) -> None:
        """Titles average ten tokens. A sliding window over a concatenated stream would
        spend much of its context on the previous, unrelated, headline."""
        word_to_index = {"a": 0, "b": 1, "c": 2, "d": 3}
        pairs = build_pairs([["a", "b"], ["c", "d"]], word_to_index, window_size=5)
        seen = {(int(centre), int(context)) for centre, context in pairs}
        assert seen == {(0, 1), (1, 0), (2, 3), (3, 2)}

    def test_out_of_vocabulary_tokens_are_dropped_not_pooled(self) -> None:
        pairs = build_pairs([["a", "zzz", "b"]], {"a": 0, "b": 1}, window_size=5)
        assert {(int(c), int(x)) for c, x in pairs} == {(0, 1), (1, 0)}


class TestMetrics:
    def test_a_perfect_ranker_scores_one(self) -> None:
        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert _spearman(actual, actual) == pytest.approx(1.0)

    def test_a_reversed_ranker_scores_minus_one(self) -> None:
        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert _spearman(-actual, actual) == pytest.approx(-1.0)

    def test_a_constant_prediction_is_not_a_correlation(self) -> None:
        """A model that predicts the mean has no ranking at all. Reporting 0.0 would be
        a claim; NaN is the truth."""
        assert np.isnan(_spearman(np.ones(10), np.arange(10.0)))

    def test_ndcg_is_one_when_the_top_k_is_the_true_top_k(self) -> None:
        actual = np.arange(20.0)
        assert _ndcg_at_k(actual, actual, 10) == pytest.approx(1.0)

    def test_ndcg_falls_when_the_ranking_is_inverted(self) -> None:
        """The floor is not zero: NDCG still credits picking the least-bad items, so
        the assertion is against the perfect ranker rather than an absolute bound."""
        actual = np.arange(20.0)
        assert _ndcg_at_k(-actual, actual, 10) < 0.25 * _ndcg_at_k(actual, actual, 10)

    def test_the_metric_block_is_complete(self) -> None:
        frame = _frame(
            [
                {
                    "id": i,
                    "title": f"post {i}",
                    "url": "",
                    "score": i + 1,
                    "time": 1_700_000_000 + i * 60,
                    "author": "a",
                }
                for i in range(40)
            ]
        )
        actual = np.log1p(frame["score"].to_numpy(dtype=np.float64))
        metrics = evaluate_predictions(frame, actual, actual)
        for key in ("spearman", "rmse", "mae", "r2", "top_decile_lift", "ndcg@10", "precision@10"):
            assert key in metrics
        assert metrics["rmse"] == pytest.approx(0.0)
        assert metrics["spearman"] == pytest.approx(1.0)


class TestArchitectures:
    """Every architecture takes the same four inputs and returns one number per post.
    The ablations get the tensors they ignore, so a caller never has to know which is
    which."""

    def _inputs(self, batch: int = 4) -> tuple[torch.Tensor, ...]:
        return (
            torch.randint(0, 50, (batch, 8)),
            torch.ones(batch, 8),
            torch.randn(batch, len(COLUMN_ORDER)),
            torch.randint(0, 3, (batch,)),
        )

    @pytest.mark.parametrize("fusion", ["text_only", "features_only", "early", "late"])
    def test_every_architecture_returns_one_score_per_post(self, fusion: Fusion) -> None:
        config = Config(fusion=fusion, hidden_dim=8, num_layers=1)
        model = build_model(torch.randn(50, 6), num_domains=3, config=config)
        assert model(*self._inputs()).shape == (4,)

    def test_build_model_rejects_an_unknown_architecture(self) -> None:
        config = Config(hidden_dim=8)
        object.__setattr__(config, "fusion", "sideways")
        with pytest.raises(ValueError, match="unknown fusion"):
            build_model(torch.randn(50, 6), num_domains=3, config=config)

    def test_a_batch_of_one_works(self) -> None:
        config = Config(fusion="late", hidden_dim=8, num_layers=1)
        model = build_model(torch.randn(50, 6), num_domains=3, config=config)
        assert model(*self._inputs(batch=1)).shape == (1,)

    def test_a_title_with_no_known_words_pools_to_zero_not_nan(self) -> None:
        config = Config(fusion="text_only", hidden_dim=8, num_layers=1)
        model = build_model(torch.randn(50, 6), num_domains=3, config=config)
        tokens = torch.zeros(2, 8, dtype=torch.long)
        mask = torch.zeros(2, 8)
        output = model(
            tokens, mask, torch.randn(2, len(COLUMN_ORDER)), torch.zeros(2, dtype=torch.long)
        )
        assert torch.isfinite(output).all()

    def test_the_embedding_table_is_not_saved_with_the_model(self) -> None:
        """It is 85MB shared by all four; saving it four times would put 340MB on the
        Hub to say the same thing."""
        config = Config(fusion="late", hidden_dim=8, num_layers=1)
        model = build_model(torch.randn(50, 6), num_domains=3, config=config)
        assert "embeddings.weight" not in predictor_state_dict(model)
        assert any(key.startswith("branches.") for key in predictor_state_dict(model))

    def test_a_saved_predictor_round_trips(self) -> None:
        config = Config(fusion="late", hidden_dim=8, num_layers=1)
        matrix = torch.randn(50, 6)
        saved = build_model(matrix, num_domains=3, config=config)
        restored = build_model(matrix, num_domains=3, config=config)
        load_predictor_weights(restored, predictor_state_dict(saved))
        saved.eval()
        restored.eval()
        inputs = self._inputs()
        with torch.no_grad():
            assert torch.allclose(saved(*inputs), restored(*inputs))

    def test_a_checkpoint_from_another_architecture_is_refused(self) -> None:
        """`strict=False` is needed because the embedding table is stored separately, and
        on its own it would let a mismatched checkpoint load nothing and serve a random
        head -- silently, with the API answering 200 the whole time."""
        matrix = torch.randn(50, 6)
        early = build_model(matrix, 3, Config(fusion="early", hidden_dim=8, num_layers=1))
        late = build_model(matrix, 3, Config(fusion="late", hidden_dim=8, num_layers=1))
        with pytest.raises(RuntimeError, match="does not fit this architecture"):
            load_predictor_weights(late, predictor_state_dict(early))

    def test_freezing_the_embeddings_stops_their_gradient(self) -> None:
        frozen = build_model(torch.randn(50, 6), 3, Config(hidden_dim=8, freeze_embeddings=True))
        thawed = build_model(torch.randn(50, 6), 3, Config(hidden_dim=8, freeze_embeddings=False))
        assert not frozen.embeddings.weight.requires_grad
        assert thawed.embeddings.weight.requires_grad

    def test_ablating_a_branch_changes_the_output(self) -> None:
        """The leave-one-group-out ablation zeroes a branch's input rather than
        retraining, so it has to actually reach the head."""
        config = Config(fusion="late", hidden_dim=8, num_layers=2)
        model = build_model(torch.randn(50, 6), num_domains=3, config=config)
        assert isinstance(model, LateFusionScorer)
        model.eval()
        inputs = self._inputs()
        with torch.no_grad():
            full = model(*inputs)
            model.disabled = {"author"}
            ablated = model(*inputs)
        assert not torch.allclose(full, ablated)


class TestApi:
    def test_health_reports_whether_the_models_loaded(self) -> None:
        with TestClient(app) as client:
            body = client.get("/health").json()
        assert "checkpoint_loaded" in body
        assert "loaded_architectures" in body

    def test_a_title_is_the_only_required_field(self) -> None:
        """The older client sent `{"title": ...}` and must keep working."""
        with TestClient(app) as client:
            response = client.post("/predict", json={"title": "Show HN: a thing"})
        assert response.status_code in (200, 503)

    def test_an_empty_title_is_rejected_before_any_work(self) -> None:
        with TestClient(app) as client:
            assert client.post("/predict", json={"title": ""}).status_code == 422

    def test_an_overlong_title_is_rejected(self) -> None:
        with TestClient(app) as client:
            response = client.post("/predict", json={"title": "x" * 10_000})
        assert response.status_code == 422

    def test_an_unknown_architecture_is_rejected(self) -> None:
        with TestClient(app) as client:
            response = client.post("/predict", json={"title": "hi", "fusion": "sideways"})
        assert response.status_code == 422


class TestConfig:
    def test_each_architecture_gets_its_own_checkpoint_path(self) -> None:
        """Four architectures writing one path would mean the comparison could only ever
        score whichever trained last."""
        fusions: tuple[Fusion, ...] = ("text_only", "features_only", "early", "late")
        paths = {Config(fusion=fusion).predictor_checkpoint for fusion in fusions}
        assert len(paths) == 4

    def test_the_embedding_matrix_is_shared_across_architectures(self) -> None:
        assert Config(fusion="early").embeddings_path == Config(fusion="late").embeddings_path

    def test_the_word_vectors_come_from_the_sibling_service(self) -> None:
        assert Config().embedding_repo_id == "roshbeed/ai-residency-word-embeddings"

    def test_the_corpus_mirror_is_this_projects_own(self) -> None:
        """Nothing depends on someone else's account: the upstream is recorded, but what
        a run reads is our copy of it."""
        config = Config()
        assert config.hn_repo_id == "roshbeed/ai-residency-hn-posts"
        assert config.upstream_repo_id != config.hn_repo_id

    def test_seed_is_fixed(self) -> None:
        assert Config().seed == 42
