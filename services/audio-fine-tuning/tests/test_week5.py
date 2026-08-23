"""Tests for Week 5.

Two things are worth guarding here. The first is reproducibility: an earlier pass
verified the demo by running it twice and diffing the output by hand, which is exactly
the kind of check that should be a test rather than a note in a TODO file. The second
is the teacher-forcing shift in `compute_loss` — getting that off by one produces a
model that learns to echo its input, a bug this portfolio has hit for real in another
week, and it is invisible until inference.

Anything touching real audio needs the bundled clip and the system `ffmpeg` binary, so
those tests skip rather than fail when either is missing.
"""

import shutil
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from audio_fine_tuning.demo import run_demo
from audio_fine_tuning.models import create_tokens, get_tokenizer
from audio_fine_tuning.training import shift_for_teacher_forcing, train
from config import DATASET_SPECS, Config
from data_prep.prepare_dataset import emotion_for_file
from serving.api import app

needs_audio = pytest.mark.skipif(
    not Config().audio_path.exists() or shutil.which("ffmpeg") is None,
    reason="needs the bundled demo clip and a system ffmpeg binary",
)


def test_config_resolves_dataset_paths() -> None:
    config = Config()
    emodb = DATASET_SPECS["emodb"]
    assert config.source_dir(emodb) == Path("datasets/emodb/wav")
    assert config.split_dir(emodb) == Path("datasets/emodb_organized_train_val")
    assert config.manifest_path(emodb, "train") == Path("data/emodb_train.csv")


def test_dataset_keys_all_have_specs() -> None:
    """A key with no spec would fail at pipeline time rather than config time."""
    for key in Config().dataset_keys:
        assert key in DATASET_SPECS


def test_emotion_code_parsing_per_corpus() -> None:
    """Each corpus encodes the emotion differently; both schemes must still work."""
    crema_d = DATASET_SPECS["crema_d"]
    emodb = DATASET_SPECS["emodb"]

    assert emotion_for_file(crema_d, Path("1091_IEO_SAD_HI.wav")) == "sad"
    assert emotion_for_file(emodb, Path("08a01Fd.wav")) == "fearful"

    # Unknown codes and short stems are skipped rather than crashing the run.
    assert emotion_for_file(crema_d, Path("1091_IEO_XXX_HI.wav")) is None
    assert emotion_for_file(emodb, Path("short.wav")) is None


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
