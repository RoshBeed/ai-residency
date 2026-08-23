"""Tests for the single-example Whisper fine-tuning demo.

The teacher-forcing test is the one that matters: the decoder is trained to predict
token n+1 from tokens up to n, and getting that shift wrong produces a model that
learns to echo its input instead of continuing it -- which trains cleanly and fails
only at inference.
"""

import shutil
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from config import Config
from serving.api import app
from whisper_fine_tuning.demo import run_demo
from whisper_fine_tuning.models import create_tokens, get_tokenizer
from whisper_fine_tuning.training import shift_for_teacher_forcing, train

# The demo clip is gitignored and openai-whisper shells out to the ffmpeg binary,
# so these two are skipped rather than failed on a host without either.
needs_audio = pytest.mark.skipif(
    not Config().audio_path.exists() or shutil.which("ffmpeg") is None,
    reason="needs the bundled demo clip and a system ffmpeg binary",
)


def test_create_tokens_has_whisper_prefix_and_terminator() -> None:
    tokenizer = get_tokenizer()
    ids = create_tokens(tokenizer, "hello")
    assert ids[:3] == [tokenizer.sot, tokenizer.language_token, tokenizer.transcribe]
    assert ids[-1] == tokenizer.eot


def test_teacher_forcing_shift_predicts_the_next_token() -> None:
    """Position t of the predictions must line up with token t+1 of the input.

    Written against explicit values rather than a trained model, because an off-by-one
    here is invisible in the loss curve — it only shows up as a model that echoes its
    input at inference time.
    """
    tokens = torch.tensor([[3, 5, 7, 9]])
    logits = torch.nn.functional.one_hot(tokens, num_classes=16).float()

    predictions, targets = shift_for_teacher_forcing(logits, tokens)

    assert targets.tolist() == [[5, 7, 9]]
    assert predictions.shape == (1, 3, 16)
    # Prediction slot 0 still carries token 3 while its target is now token 5, which
    # is what "predict the next one" means. Without the shift they would match.
    assert int(predictions[0, 0].argmax()) == 3
    assert int(targets[0, 0]) == 5


@needs_audio
def test_demo_is_reproducible() -> None:
    """Same seed, same transcripts. Verified by hand in an earlier pass; now guarded."""
    config = Config()
    assert run_demo(config) == run_demo(config)


@needs_audio
def test_train_writes_a_checkpoint_and_reports_a_finite_loss(tmp_path: Path) -> None:
    config = Config(checkpoint_dir=tmp_path, num_epochs=2)
    best_loss = train(config)
    assert config.checkpoint_path.exists()
    assert 0.0 <= best_loss < float("inf")


def test_health_endpoint_reports_device() -> None:
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["device"] in {"cuda", "mps", "cpu"}


def test_transcribe_rejects_invalid_base64() -> None:
    client = TestClient(app)
    response = client.post("/transcribe", json={"audio_data": "not base64!!"})
    assert response.status_code == 400


def test_transcribe_rejects_empty_audio() -> None:
    client = TestClient(app)
    response = client.post("/transcribe", json={"audio_data": ""})
    assert response.status_code == 400
