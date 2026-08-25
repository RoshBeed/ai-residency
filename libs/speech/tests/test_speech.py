"""Tests for the shared Whisper layer.

The load tests pull `openai/whisper-tiny` (~150MB) and are skipped when it is not
already cached, so a clean checkout without network still runs the rest. The audio and
label tests need no weights and always run.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import WhisperProcessor

from speech import (
    IGNORE_INDEX,
    SAMPLE_RATE,
    collate,
    encode_target,
    load_waveform,
    load_whisper,
    to_features,
    transcribe,
)

MODEL = "openai/whisper-tiny"


def _cached() -> bool:
    """Whether the model is already on disk, so a test can skip rather than download."""
    try:
        WhisperProcessor.from_pretrained(MODEL, local_files_only=True)
    except Exception:
        return False
    return True


needs_model = pytest.mark.skipif(not _cached(), reason=f"{MODEL} is not cached locally")


@pytest.fixture(scope="module")
def processor() -> WhisperProcessor:
    return WhisperProcessor.from_pretrained(MODEL, local_files_only=True)


def tone(seconds: float = 2.0, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """A deterministic waveform. Contents do not matter; shape and dtype do."""
    t = np.linspace(0, seconds, int(seconds * sample_rate), endpoint=False)
    return (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------


@needs_model
def test_features_are_whispers_fixed_window(processor: WhisperProcessor) -> None:
    """80 mel bands x 3000 frames, whatever the clip length -- a property of the
    architecture, not a preprocessing choice."""
    assert to_features(processor, tone(2.0)).shape == (80, 3000)


@needs_model
@pytest.mark.parametrize("seconds", [0.5, 2.0, 45.0])
def test_short_and_long_clips_both_reach_the_same_shape(
    processor: WhisperProcessor, seconds: float
) -> None:
    """Shorter is padded, longer is truncated. Both are Whisper's doing, not ours."""
    assert to_features(processor, tone(seconds)).shape == (80, 3000)


@needs_model
def test_features_match_openai_whispers_own_spectrogram(processor: WhisperProcessor) -> None:
    """The claim that justified moving both services onto transformers.

    If these two disagreed, migrating whisper-fine-tuning off `openai-whisper` would
    change its inputs and therefore its results. They do not disagree -- the two
    implementations produce the same tensor, so the move is lossless.
    """
    whisper_pkg = pytest.importorskip("whisper", reason="openai-whisper not installed")

    audio = tone(5.0)
    theirs = whisper_pkg.log_mel_spectrogram(whisper_pkg.pad_or_trim(audio))
    ours = to_features(processor, audio)

    assert ours.shape == theirs.shape
    assert torch.allclose(ours, theirs, atol=1e-4)


def test_load_waveform_resamples_and_returns_float32(tmp_path: Path) -> None:
    soundfile = pytest.importorskip("soundfile")
    path = tmp_path / "clip.wav"
    soundfile.write(path, tone(1.0, 8000), 8000)

    waveform = load_waveform(path)

    assert waveform.dtype == np.float32
    assert waveform.ndim == 1
    # 1s at 8kHz, resampled to Whisper's 16kHz.
    assert abs(len(waveform) - SAMPLE_RATE) < SAMPLE_RATE * 0.05


def test_load_waveform_honours_max_duration(tmp_path: Path) -> None:
    soundfile = pytest.importorskip("soundfile")
    path = tmp_path / "clip.wav"
    soundfile.write(path, tone(4.0), SAMPLE_RATE)

    waveform = load_waveform(path, max_duration=1.0)

    assert abs(len(waveform) - SAMPLE_RATE) < SAMPLE_RATE * 0.05


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


@needs_model
def test_encode_target_drops_the_token_the_model_prepends(processor: WhisperProcessor) -> None:
    """The whole reason encode_target exists.

    WhisperForConditionalGeneration builds its decoder input by shifting labels right
    and prepending decoder_start_token_id, which *is* <|startoftranscript|>. Leave the
    tokenizer's own copy in and the decoder input starts [SOT, SOT, ...].
    """
    raw = processor.tokenizer("My name is Rosh", add_special_tokens=True).input_ids
    labels = encode_target(processor, "My name is Rosh")

    assert processor.tokenizer.convert_ids_to_tokens(raw[:1]) == ["<|startoftranscript|>"]
    assert labels.tolist() == raw[1:]
    assert processor.tokenizer.convert_ids_to_tokens(labels.tolist())[-1] == "<|endoftext|>"


@needs_model
def test_the_doubled_start_token_would_change_the_loss(processor: WhisperProcessor) -> None:
    """Not a style point -- it silently changes what the model is trained to do.

    With the duplicate the decoder is one position out of step, so it learns to emit
    <|startoftranscript|> exactly where a caller will later read its answer. Nothing
    errors; the loss simply describes a different task.
    """
    from transformers import WhisperForConditionalGeneration

    model = WhisperForConditionalGeneration.from_pretrained(MODEL, local_files_only=True)
    model.eval()
    features = to_features(processor, tone(3.0)).unsqueeze(0)

    correct = encode_target(processor, "My name is Rosh").unsqueeze(0)
    doubled = torch.cat([correct[:, :1] * 0 + model.config.decoder_start_token_id, correct], dim=1)

    with torch.no_grad():
        clean = model(input_features=features, labels=correct).loss.item()
        broken = model(input_features=features, labels=doubled).loss.item()

    assert clean != pytest.approx(broken, abs=0.01)
    assert clean < broken


@needs_model
def test_an_added_token_encodes_as_exactly_one_id(processor: WhisperProcessor) -> None:
    """An emotion label has to be a single token, or the decoder is being asked to
    spell it rather than to pick it."""
    processor.tokenizer.add_special_tokens({"additional_special_tokens": ["<|happy|>"]})

    ids = processor.tokenizer("<|happy|>", add_special_tokens=False).input_ids

    assert len(ids) == 1


def test_collate_pads_labels_with_the_ignored_index() -> None:
    """Padding with anything else trains the decoder to emit padding."""
    batch = [
        (torch.zeros(80, 3000), torch.tensor([1, 2, 3])),
        (torch.zeros(80, 3000), torch.tensor([4])),
    ]

    features, labels = collate(batch)

    assert features.shape == (2, 80, 3000)
    assert labels.shape == (2, 3)
    assert labels[1].tolist() == [4, IGNORE_INDEX, IGNORE_INDEX]


def test_ignore_index_is_what_cross_entropy_skips() -> None:
    """-100 is not arbitrary; it is torch's own default ignore_index."""
    assert IGNORE_INDEX == -100
    logits = torch.randn(2, 5)
    targets = torch.tensor([1, IGNORE_INDEX])
    loss = torch.nn.functional.cross_entropy(logits, targets)
    only_first = torch.nn.functional.cross_entropy(logits[:1], targets[:1])
    assert loss.item() == pytest.approx(only_first.item())


# ---------------------------------------------------------------------------
# Loading, and the invariant it exists to hold
# ---------------------------------------------------------------------------


@needs_model
def test_loading_puts_model_and_processor_on_one_device() -> None:
    whisper = load_whisper(MODEL, device=torch.device("cpu"))

    assert next(whisper.model.parameters()).device.type == "cpu"
    assert whisper.device.type == "cpu"


@needs_model
def test_extra_tokens_resize_the_embedding_table_to_match() -> None:
    """The invariant this module exists to hold. Add tokens without resizing and the
    new ids index past the end of the table -- a crash if you are lucky, and silently
    wrong if they happen to land inside it."""
    base = load_whisper(MODEL, device=torch.device("cpu"))
    extended = load_whisper(
        MODEL, device=torch.device("cpu"), extra_tokens=["<|happy|>", "<|sad|>"]
    )

    assert extended.vocab_size == base.vocab_size + 2
    assert extended.vocab_size >= len(extended.processor.tokenizer)

    # And the new ids are addressable rather than out of range.
    new_id = extended.processor.tokenizer.convert_tokens_to_ids("<|happy|>")
    assert extended.model.get_input_embeddings()(torch.tensor([new_id])).shape == (1, 384)


@needs_model
def test_a_checkpoint_from_a_different_vocabulary_is_refused(tmp_path: Path) -> None:
    """Loading emotion-trained weights into a base-vocabulary model would either
    crash deep in the decoder or quietly mismatch every added token."""
    extended = load_whisper(MODEL, device=torch.device("cpu"), extra_tokens=["<|happy|>"])
    checkpoint = tmp_path / "emotion.pt"
    torch.save(extended.model.state_dict(), checkpoint)

    with pytest.raises(ValueError, match="vocabulary"):
        load_whisper(MODEL, device=torch.device("cpu"), checkpoint=checkpoint)


@needs_model
def test_a_checkpoint_with_the_matching_vocabulary_loads(tmp_path: Path) -> None:
    """The other half: the same tokens it was trained with, and it restores."""
    extended = load_whisper(MODEL, device=torch.device("cpu"), extra_tokens=["<|happy|>"])
    checkpoint = tmp_path / "emotion.pt"
    torch.save(extended.model.state_dict(), checkpoint)

    restored = load_whisper(
        MODEL,
        device=torch.device("cpu"),
        extra_tokens=["<|happy|>"],
        checkpoint=checkpoint,
    )

    assert restored.vocab_size == extended.vocab_size


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


@needs_model
def test_transcribe_returns_text_and_accepts_an_unbatched_clip() -> None:
    whisper = load_whisper(MODEL, device=torch.device("cpu"))
    features = to_features(whisper.processor, tone(2.0))

    text = transcribe(whisper, features, max_new_tokens=4)

    assert isinstance(text, str)


@needs_model
def test_transcribe_leaves_training_mode_alone() -> None:
    """It is called *between* gradient steps by the fine-tuning demo. A helper that
    flipped the model to eval and left it there would silently disable dropout for
    the rest of training."""
    whisper = load_whisper(MODEL, device=torch.device("cpu"))
    whisper.model.train()
    features = to_features(whisper.processor, tone(1.0))

    transcribe(whisper, features, max_new_tokens=2)

    assert whisper.model.training is True
