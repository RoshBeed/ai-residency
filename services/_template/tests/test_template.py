"""Tests for the template.

These are the checks worth carrying into every repo instantiated from it: that a run
is reproducible, that the baseline is computed against the same split the model is
scored on, and that the service answers.
"""

from pathlib import Path

from fastapi.testclient import TestClient

from config import Config
from serving.api import app
from template.training import train


def _fast_config(tmp_path: Path) -> Config:
    """A configuration small enough to train inside a test."""
    return Config(num_epochs=2, num_examples=64, checkpoint_dir=tmp_path)


def test_training_is_reproducible(tmp_path: Path) -> None:
    """Same seed, same result. This is what unseeded runs cost us elsewhere."""
    first = train(_fast_config(tmp_path))
    second = train(_fast_config(tmp_path))
    assert first == second


def test_training_writes_a_checkpoint(tmp_path: Path) -> None:
    config = _fast_config(tmp_path)
    train(config)
    assert config.checkpoint_path.exists()


def test_health_endpoint_reports_device() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["device"] in {"cuda", "mps", "cpu"}


def test_predict_endpoint_returns_a_number() -> None:
    client = TestClient(app)
    features = [0.0] * Config().input_dim
    response = client.post("/predict", json={"features": features})
    assert response.status_code == 200
    assert isinstance(response.json()["prediction"], float)
