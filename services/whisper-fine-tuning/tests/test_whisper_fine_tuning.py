"""Tests for the single-example Whisper fine-tuning demo.

The teacher-forcing test is the one that matters: the decoder is trained to predict
token n+1 from tokens up to n, and getting that shift wrong produces a model that
learns to echo its input instead of continuing it -- which trains cleanly and fails
only at inference.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from config import Config
from serving.api import app
from speech import encode_target
from whisper_fine_tuning.demo import run_demo
from whisper_fine_tuning.models import load_whisper_model
from whisper_fine_tuning.training import train

# The demo needs its clip, which is gitignored and resolved from the Hub at run time
# like every other artifact here. This used to gate on a system ffmpeg binary instead,
# because openai-whisper shelled out to it; audio now goes through libsndfile, so the
# only thing that can actually be missing is the clip.
needs_audio = pytest.mark.skipif(
    not Config().audio_path.exists(),
    reason="demo clip not present; run `make demo` or let the Hub resolve it",
)


def test_labels_omit_the_token_the_model_prepends() -> None:
    """The shift this service used to do by hand is now the model's own.

    `WhisperForConditionalGeneration` builds its decoder input by shifting labels
    right and prepending `decoder_start_token_id`, which is `<|startoftranscript|>`.
    Leaving the tokenizer's copy in makes the decoder input `[SOT, SOT, ...]`, so the
    model trains one position out of step with what it is asked at inference -- an
    off-by-one that is invisible in the loss curve.

    The prefix itself is no longer assembled from special-token ids either: the
    tokenizer emits it, including the `<|notimestamps|>` the hand-built version
    omitted.
    """
    # Deliberately not through prepare_training_data: that needs the demo clip, which
    # is gitignored and resolved from the Hub at run time, and this rule is about the
    # label sequence rather than about any particular audio.
    whisper = load_whisper_model(Config())
    labels = encode_target(whisper.processor, "hello")

    tokens = whisper.processor.tokenizer.convert_ids_to_tokens(labels.tolist())
    assert tokens[0] != "<|startoftranscript|>"
    assert tokens[-1] == "<|endoftext|>"
    # And the prefix the old hand-built version omitted is present.
    assert "<|notimestamps|>" in tokens


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
    # 422, not 400: the request is well-formed, one field's content is unprocessable.
    # Shared with every other backend via ai_residency_common.decode_audio.
    assert response.status_code == 422


def test_transcribe_rejects_empty_audio() -> None:
    """422, not 400: the repo settled on 422 for an undecodable payload."""
    client = TestClient(app)
    response = client.post("/transcribe", json={"audio_data": ""})
    assert response.status_code == 422


def test_undecodable_audio_is_422_not_500() -> None:
    """Valid base64 that is not audio is a malformed request, not a server error.

    The guard used to catch `RuntimeError`. The decoder actually raises
    `audioread.exceptions.NoBackendError`, which is a `DecodeError` and unrelated to
    `RuntimeError`, so it escaped as a 500 -- and the streaming handler had no guard
    on that call at all. Both now answer 422, matching what `decode_image` does at
    the same boundary in `libs/common`.
    """
    import base64

    payloads = {
        "not base64": "not base64!!",
        "empty": "",
        "valid base64, not audio": base64.b64encode(b"\x00\x01\x02 not media").decode(),
    }
    client = TestClient(app)
    for name, payload in payloads.items():
        response = client.post("/transcribe", json={"audio_data": payload})
        assert response.status_code == 422, f"{name} gave {response.status_code}"
