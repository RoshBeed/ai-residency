"""Tests for the emotion corpora, the manifests and the trained model.

The per-corpus label parsing is worth guarding: EmoDB and CREMA-D both encode the
emotion inside the filename, in different formats at different positions, and getting
one wrong silently mislabels an entire corpus rather than raising.
"""

import csv
from pathlib import Path
from typing import cast

import pytest
import torch
from fastapi.testclient import TestClient

from config import DATASET_SPECS, Config
from data_prep.prepare_dataset import emotion_for_file
from serving.api import app
from speech import IGNORE_INDEX, collate, encode_target
from speech_emotion_recognition.data import EmotionDataset
from speech_emotion_recognition.models import load_processor
from speech_emotion_recognition.training import compute_baseline


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


def _row(path: str, label: str) -> dict[str, str]:
    """One manifest row in the shape `create_manifest` writes."""
    return {"audio_path": path, "label": label}


def _write_manifest(config: Config, split: str, rows: list[dict[str, str]]) -> None:
    """Write a manifest for the first configured corpus."""
    path = config.manifest_path(config.datasets[0], split)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["audio_path", "label"])
        writer.writeheader()
        writer.writerows(rows)


def test_emotion_tokens_and_labels_stay_in_step() -> None:
    """`emotions` and `emotion_tokens` are two views of one list; a mismatch would
    label every prediction with the wrong emotion while looking entirely healthy."""
    config = Config()
    assert len(config.emotions) == len(config.emotion_tokens)
    for emotion, token in zip(config.emotions, config.emotion_tokens, strict=True):
        assert config.token_for(emotion) == token


def test_baseline_is_the_majority_emotion(tmp_path: Path) -> None:
    """A model that cannot beat 'always answer the most common emotion' has learned
    nothing about the audio, however good its loss curve looks."""
    from speech_emotion_recognition.data import load_manifest_rows

    config = Config(data_dir=tmp_path / "data", hf_repo_id=None, hf_dataset_repo_id=None)
    _write_manifest(
        config,
        "train",
        [_row("a.wav", "happy"), _row("b.wav", "happy"), _row("c.wav", "sad")],
    )
    rows = load_manifest_rows(config, "train")
    assert len(rows) == 3

    class _Stub:
        """Just enough of EmotionDataset for compute_baseline: the label column."""

        def __init__(self) -> None:
            self.rows = [(Path("a"), "happy"), (Path("b"), "happy"), (Path("c"), "sad")]

    emotion, accuracy = compute_baseline(cast(EmotionDataset, _Stub()))
    assert emotion == "happy"
    assert accuracy == pytest.approx(2 / 3)


def test_collate_pads_labels_with_the_ignored_index() -> None:
    """Padded label positions must not teach the decoder to predict padding."""
    features = torch.zeros(80, 3000)
    batch = [(features, torch.tensor([1, 2, 3])), (features, torch.tensor([4]))]
    stacked_features, labels = collate(batch)
    assert stacked_features.shape[0] == 2
    assert labels.shape == (2, 3)
    assert labels[1, 1].item() == IGNORE_INDEX
    assert labels[1, 2].item() == IGNORE_INDEX


def test_missing_manifests_name_the_command_that_builds_them(tmp_path: Path) -> None:
    """A missing manifest is a normal state on a fresh host, so the error has to say
    which command produces one rather than surfacing a bare FileNotFoundError."""
    from speech_emotion_recognition.data import EmotionDataset

    config = Config(data_dir=tmp_path / "empty", hf_repo_id=None, hf_dataset_repo_id=None)
    with pytest.raises(FileNotFoundError, match="prepare-data"):
        EmotionDataset(config, processor=None, split="train")


def test_health_reports_the_emotion_vocabulary() -> None:
    """The frontend renders one bar per emotion, so /health has to say which."""
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert set(body["emotions"]) == set(Config().emotions)
    assert body["checkpoint_source"] in {"local", "hub", "none"}


def test_predict_answers_503_without_a_checkpoint() -> None:
    """No trained checkpoint is a fact to report, not a reason to refuse to start."""
    with TestClient(app) as client:
        if client.get("/health").json()["checkpoint_loaded"]:
            pytest.skip("a checkpoint is present, so /predict should answer normally")
        response = client.post("/predict", json={"audio_data": "AAAA"})
    assert response.status_code == 503


def test_the_label_is_what_the_first_decoder_step_predicts() -> None:
    """Training and serving have to agree about which position holds the answer.

    `serving/api.py` feeds the decoder a single [SOT] and reads `logits[0, -1]`, so
    the emotion must be what the model emits at step one. Whisper builds its decoder
    input by shifting labels right and prepending SOT, so labels of
    `[<|emotion|>, <|endoftext|>]` give a decoder input of `[SOT, <|emotion|>]` --
    step one predicts the emotion, which is exactly what serving reads.

    This used to be the tokenizer's full output: [SOT, <|en|>, <|transcribe|>,
    <|notimestamps|>, <|emotion|>, EOT]. That put a duplicate SOT at the front after
    the shift and pushed the emotion four steps in, so serving read a position trained
    to emit <|startoftranscript|>. Nothing errored -- it simply answered from the
    wrong slot.
    """
    config = Config()
    processor = load_processor(config)

    labels = encode_target(processor, config.token_for(config.emotions[0]), prefix=False)
    tokens = processor.tokenizer.convert_ids_to_tokens(labels.tolist())

    assert tokens[0] == config.token_for(config.emotions[0])
    assert tokens[-1] == "<|endoftext|>"
    assert len(tokens) == 2

    start = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    assert int(labels[0]) != start, "a leading SOT would be doubled by the model's own shift"
