"""Tests for the content-scoring stage.

Focused on the things that actually broke here: the scorer having to accept both
batched and unbatched input, and the API having to answer without a trained
checkpoint present.
"""

from fastapi.testclient import TestClient

from config import Config
from content_scoring.model import ScorePredictor
from serving.api import app


class TestScorer:
    def test_accepts_batched_and_unbatched_input(self) -> None:
        import torch

        model = ScorePredictor(input_dim=16)
        assert model(torch.zeros(16)).shape == ()
        assert model(torch.zeros(4, 16)).shape == (4,)


class TestApi:
    def test_health_reports_whether_the_model_loaded(self) -> None:
        response = TestClient(app).get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_root_describes_the_service(self) -> None:
        response = TestClient(app).get("/")
        assert response.status_code == 200
        assert "/predict" in response.json()["endpoint"]


class TestConfig:
    def test_paths_derive_from_the_checkpoint_directory(self) -> None:
        config = Config()
        assert config.checkpoint_path == config.checkpoint_dir / config.checkpoint_name
        assert config.info_path == config.checkpoint_dir / config.info_name

    def test_seed_is_fixed(self) -> None:
        assert Config().seed == 42

    def test_cbow_checkpoint_is_pulled_from_the_word_embeddings_repo(self) -> None:
        assert Config().cbow_repo_id == "roshbeed/ai-residency-word-embeddings"
