"""Tests for the dual-encoder retrieval pipeline.

Weighted towards the things that actually broke here: artifact paths drifting away from
where the artifacts live, three copies of a tower definition, a search-result parser that
has to handle two different redis-py response shapes, and a training run whose reported
number has to describe the checkpoint on disk.
"""

import json
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from ai_residency_common import load_checkpoint
from cbow.vocabulary import (
    build_embedding_layer,
    build_embedding_placeholder,
    latest_cbow_checkpoint,
    load_words_to_ids,
)
from config import Config
from models.towers import GruTower, encode
from models.training import (
    build_towers,
    compute_baseline,
    load_triples,
    pairwise_accuracy,
    train,
)
from retrieval.query_engine import (
    _parse_search_results,
)

# --- Configuration and artifact paths ---------------------------------------------
#
# Every trained artifact path used to be a string literal in whichever module needed it,
# and they drifted out of sync with where the files actually live. These assert that all
# of them now derive from `data_dir` and `checkpoint_dir`.


def test_artifact_paths_derive_from_the_configured_directories(tmp_path: Path) -> None:
    config = Config(data_dir=tmp_path / "d", checkpoint_dir=tmp_path / "c")
    assert config.corpus_path.parent == config.data_dir
    assert config.words_to_ids_path.parent == config.data_dir
    assert config.tokenized_triples_path.parent == config.data_dir
    assert config.query_tower_path.parent == config.checkpoint_dir
    assert config.document_tower_path.parent == config.checkpoint_dir
    assert config.cbow_checkpoint_dir.parent == config.checkpoint_dir


def test_missing_vocabulary_names_the_command_that_creates_it(config: Config) -> None:
    with pytest.raises(FileNotFoundError, match="make train-cbow"):
        load_words_to_ids(config)


def test_missing_cbow_checkpoint_names_the_command_that_creates_it(
    config: Config, vocabulary: dict[str, int]
) -> None:
    with pytest.raises(FileNotFoundError, match="make train-cbow"):
        latest_cbow_checkpoint(config)


def test_missing_triples_name_the_command_that_creates_them(config: Config) -> None:
    with pytest.raises(FileNotFoundError, match="make train-triples"):
        load_triples(config)


# --- Towers ------------------------------------------------------------------------


def test_embeddings_load_frozen(config: Config, cbow_checkpoint: Path) -> None:
    """The towers train on top of the CBOW vectors without updating them."""
    embedding_layer = build_embedding_layer(config)
    assert embedding_layer.weight.requires_grad is False
    assert embedding_layer.embedding_dim == config.embedding_dim


def test_serving_builds_a_tower_without_a_cbow_checkpoint(
    config: Config, vocabulary: dict[str, int]
) -> None:
    """A trained tower carries its own embedding weights, so serving must not require
    the CBOW checkpoint they came from -- a served host has the towers, not the
    training artifacts."""
    tower = GruTower(build_embedding_placeholder(config), config.hidden_size)
    assert tower.embedding.num_embeddings == len(vocabulary)
    assert tower.embedding.embedding_dim == config.embedding_dim
    assert "embedding.weight" in tower.state_dict()


def test_query_and_document_towers_have_separate_weights(
    config: Config, cbow_checkpoint: Path
) -> None:
    """A query must not be encoded with the document tower's weights."""
    query_tower, document_tower = build_towers(config)
    assert query_tower is not document_tower
    query_weights = query_tower.rnn.state_dict()["weight_ih_l0"]
    document_weights = document_tower.rnn.state_dict()["weight_ih_l0"]
    assert not torch.equal(query_weights, document_weights)


def test_towers_share_one_embedding_matrix(config: Config, cbow_checkpoint: Path) -> None:
    """Both sides must read the same word vectors for a cosine similarity to mean anything."""
    query_tower, document_tower = build_towers(config)
    assert query_tower.embedding is document_tower.embedding


def test_encoding_empty_tokens_returns_none(config: Config, cbow_checkpoint: Path) -> None:
    """Text with no known words has nothing to encode, and a GRU cannot take a zero-length
    sequence. Callers skip those triples rather than inventing a vector."""
    tower = GruTower(build_embedding_layer(config), config.hidden_size)
    assert encode(tower, []) is None
    assert encode(tower, [1, 2]) is not None


def test_encoding_width_matches_the_index_dimension(config: Config, cbow_checkpoint: Path) -> None:
    """The Redis index is created with DIM = hidden_size, so the vectors must match it."""
    tower = GruTower(build_embedding_layer(config), config.hidden_size)
    encoding = encode(tower, [1, 2, 3])
    assert encoding is not None
    assert encoding.shape == (config.hidden_size,)


# --- Training ----------------------------------------------------------------------


def test_untrained_baseline_is_reported_and_reproducible(
    config: Config, cbow_checkpoint: Path, triples: Path
) -> None:
    """The baseline exists to answer 'compared to what?', so it has to be stable."""
    data = load_triples(config)
    torch.manual_seed(config.seed)
    first, count = compute_baseline(config, data["validation"])
    torch.manual_seed(config.seed)
    second, _ = compute_baseline(config, data["validation"])
    assert first == second
    assert 0.0 <= first <= 1.0
    assert count == len(data["validation"])


def test_training_is_reproducible(config: Config, cbow_checkpoint: Path, triples: Path) -> None:
    """Same seed, same numbers. Unseeded runs in this portfolio were hiding real bugs."""
    first = train(config)
    second = train(config)
    assert first == second


def test_training_persists_both_towers(
    config: Config, cbow_checkpoint: Path, triples: Path
) -> None:
    """The original trained in memory and saved nothing, so retrieval silently used random
    weights. Both checkpoints must exist and carry the architecture that produced them."""
    train(config)
    assert config.query_tower_path.exists()
    assert config.document_tower_path.exists()
    payload = load_checkpoint(config.query_tower_path)
    assert payload["extra"]["hidden_size"] == config.hidden_size
    assert payload["extra"]["embedding_dim"] == config.embedding_dim


def test_reported_accuracy_describes_the_saved_checkpoint(
    config: Config, cbow_checkpoint: Path, triples: Path
) -> None:
    """Reloading the persisted towers must reproduce the accuracy the run reported, or the
    number describes weights nobody can get back."""
    result = train(config)

    query_tower, document_tower = build_towers(config)
    load_checkpoint(config.query_tower_path, query_tower)
    load_checkpoint(config.document_tower_path, document_tower)
    query_tower.eval()
    document_tower.eval()

    accuracy, _ = pairwise_accuracy(query_tower, document_tower, load_triples(config)["validation"])
    assert accuracy == result.trained_accuracy


def test_training_reduces_the_triplet_loss(
    config: Config, cbow_checkpoint: Path, triples: Path
) -> None:
    result = train(config)
    assert result.best_loss < config.margin


# --- Search-result parsing ---------------------------------------------------------
#
# redis-py 5 changed the FT.SEARCH response shape. Both are parsed, and both are covered
# here because the old one is otherwise only reachable by pinning an old client.


def test_parses_the_modern_redis_response_shape() -> None:
    response = {
        b"results": [
            {b"extra_attributes": {b"score": b"0.25", b"text": b"reagan was a democrat"}},
            {b"extra_attributes": {b"score": b"0.75", b"text": b"rust forms on nails"}},
        ]
    }
    assert _parse_search_results(response) == [
        (0.25, "reagan was a democrat", None),
        (0.75, "rust forms on nails", None),
    ]


def test_parses_the_legacy_flat_redis_response_shape() -> None:
    response = [
        2,
        b"doc:0",
        [b"text", b"reagan was a democrat", b"score", b"0.25"],
        b"doc:1",
        [b"text", b"rust forms on nails", b"score", b"0.75"],
    ]
    assert _parse_search_results(response) == [
        (0.25, "reagan was a democrat", None),
        (0.75, "rust forms on nails", None),
    ]


def test_an_empty_index_parses_to_no_results() -> None:
    assert _parse_search_results({b"results": []}) == []
    assert _parse_search_results([0]) == []


def test_a_document_without_a_score_is_dropped() -> None:
    response = {b"results": [{b"extra_attributes": {b"text": b"no score here"}}]}
    assert _parse_search_results(response) == []


def test_a_document_without_text_still_ranks() -> None:
    response = {b"results": [{b"extra_attributes": {b"score": b"0.5"}}]}
    assert _parse_search_results(response) == [(0.5, "[No text found]", None)]


# --- Serving -----------------------------------------------------------------------


def test_health_endpoint_reports_ok() -> None:
    from serving.api import app

    with TestClient(app) as client:
        response = client.get("/")
    assert response.status_code == 200

    payload = response.json()
    assert payload["status"] == "ok"
    # Reported so a service running on an untrained tower is visible from outside.
    assert payload["checkpoint_source"] in {"local", "hub", "none"}
    assert payload["checkpoint_loaded"] is (payload["checkpoint_source"] != "none")


def test_query_reports_503_when_the_model_is_untrained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An untrained repo must answer cleanly rather than crash the service."""
    from serving import api

    def missing(*_: object, **__: object) -> None:
        raise FileNotFoundError("qry_tower.pth not found")

    monkeypatch.setattr(api, "run_query", missing)
    with TestClient(api.app) as client:
        response = client.post("/query", json={"text": "was reagan a democrat", "top_k": 3})
    assert response.status_code == 503
    assert "not ready" in response.json()["detail"].lower()


def test_query_returns_ranked_results(monkeypatch: pytest.MonkeyPatch) -> None:
    from serving import api

    monkeypatch.setattr(api, "load_vocabulary", lambda: {"was": 0, "reagan": 1})
    monkeypatch.setattr(api, "embed_query", lambda text: [0.0] * 8)
    monkeypatch.setattr(
        api,
        "run_query",
        lambda text, top_k: [(0.25, "closest", None), (0.75, "further", None)],
    )
    with TestClient(api.app) as client:
        response = client.post("/query", json={"text": "was reagan a democrat", "top_k": 2})

    assert response.status_code == 200
    body = response.json()
    assert [result["rank"] for result in body["results"]] == [1, 2]
    assert body["results"][0]["text"] == "closest"


# --- Data prep ---------------------------------------------------------------------


def test_tokenizing_maps_unknown_words_to_padding(
    config: Config, vocabulary: dict[str, int]
) -> None:
    """Words the CBOW corpus never saw arrive at the towers as padding. That is a real limit
    of this pipeline, so it is pinned rather than left implicit."""
    from data_prep.tokenize_triples import (
        tokenize_text,
    )

    assert tokenize_text("reagan quokka", vocabulary) == [vocabulary["reagan"], 0]


def test_documents_are_deduplicated_by_token_sequence(
    config: Config, cbow_checkpoint: Path, triples: Path
) -> None:
    """MS MARCO repeats passages across queries, and it is the token sequence that decides
    the embedding, so that is what deduplication keys on."""
    from retrieval.save_doc_embeddings_to_redis import (
        unique_documents,
    )

    with config.tokenized_triples_path.open() as handle:
        raw = json.load(handle)
    total = sum(len(split) for split in raw.values())

    documents = unique_documents(config)
    assert len(documents) == 2
    assert len(documents) < total


# --- CBOW stage ---------------------------------------------------------------------


def test_cbow_dataset_pads_the_window_at_the_corpus_edges(
    config: Config, vocabulary: dict[str, int]
) -> None:
    """The first token has no left context. It must be padded with the reserved `<PAD>` id,
    not filled by a negative slice wrapping around to the end of the corpus."""
    from cbow.dataset import Wiki

    dataset = Wiki(config)
    context, target = dataset[0]
    assert context.tolist()[: config.context_size] == [0] * config.context_size
    assert target.tolist() == [dataset.tokens[0]]
    assert len(context) == config.context_size * 2


def test_cbow_training_writes_a_loadable_checkpoint(
    config: Config, vocabulary: dict[str, int]
) -> None:
    """The tower stage reads whichever CBOW checkpoint is newest, so the stage has to write
    one in the shared format with the architecture recorded alongside it."""
    from cbow.train_word2vec import train_word2vec

    loss = train_word2vec(config)
    assert loss > 0

    checkpoint = latest_cbow_checkpoint(config)
    payload = load_checkpoint(checkpoint)
    assert payload["extra"]["vocab_size"] == len(vocabulary)
    assert payload["extra"]["embedding_dim"] == config.embedding_dim

    embedding_layer = build_embedding_layer(config)
    assert embedding_layer.num_embeddings == len(vocabulary)


def test_nearest_words_skips_a_word_outside_the_vocabulary(
    config: Config, vocabulary: dict[str, int]
) -> None:
    """Runs inside a training loop, so an unknown probe word must not raise."""
    from cbow.evaluate import nearest_words
    from cbow.model import build_cbow

    ids_to_words = {index: word for word, index in vocabulary.items()}
    model = build_cbow(len(vocabulary), config.embedding_dim)
    assert nearest_words(model, "quokka", vocabulary, ids_to_words) == []

    neighbours = nearest_words(model, "reagan", vocabulary, ids_to_words, top_k=3)
    assert len(neighbours) == 3
    assert all(word != "reagan" for word, _ in neighbours)
