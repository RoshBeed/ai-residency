"""Audio in, log-mel spectrogram out.

Whisper's encoder takes a fixed 30-second window of 80 mel bands, whatever you give
it: shorter clips are padded, longer ones truncated. That is a property of the
architecture rather than a preprocessing choice, which is why every caller here goes
through the same two steps and none of them get to pick a shape.

The output is bit-identical to what `openai-whisper`'s `log_mel_spectrogram` produces
for the same waveform -- checked, not assumed, before the services were moved off it.
"""

from pathlib import Path

import librosa
import numpy as np
import torch
from transformers import WhisperProcessor

__all__ = ["SAMPLE_RATE", "features_from_file", "load_waveform", "to_features"]

SAMPLE_RATE = 16_000
"""What Whisper was trained at. Resampling to it is not optional."""


def load_waveform(
    path: str | Path,
    *,
    sample_rate: int = SAMPLE_RATE,
    max_duration: float | None = None,
) -> np.ndarray:
    """Read an audio file as a mono waveform at Whisper's sample rate.

    librosa rather than a shell-out to `ffmpeg`, which is what `openai-whisper` does
    and what used to make the system binary a deployment requirement for one of these
    services.

    Args:
        path: Any file librosa can read.
        sample_rate: Target rate. Leave at the default unless you know why.
        max_duration: Seconds to read at most. None reads the file.

    Returns:
        A 1-D float32 waveform.
    """
    waveform, _ = librosa.load(str(path), sr=sample_rate, duration=max_duration)
    return np.asarray(waveform, dtype=np.float32)


def to_features(
    processor: WhisperProcessor,
    waveform: np.ndarray,
    *,
    sample_rate: int = SAMPLE_RATE,
) -> torch.Tensor:
    """Turn a waveform into the encoder's input.

    Args:
        processor: Supplies the feature extractor.
        waveform: 1-D audio samples.
        sample_rate: The rate `waveform` is at.

    Returns:
        A `(80, 3000)` log-mel spectrogram -- 80 mel bands across Whisper's fixed
        30-second window, with no batch dimension. Add one before calling the model.
    """
    extracted = processor.feature_extractor(
        np.asarray(waveform, dtype=np.float32),
        sampling_rate=sample_rate,
        return_tensors="pt",
    )
    features: torch.Tensor = extracted.input_features[0]
    return features


def features_from_file(
    processor: WhisperProcessor,
    path: str | Path,
    *,
    sample_rate: int = SAMPLE_RATE,
    max_duration: float | None = None,
) -> torch.Tensor:
    """Read a file straight to encoder features. The two steps above, together.

    Args:
        processor: Supplies the feature extractor.
        path: Audio file to read.
        sample_rate: Target rate.
        max_duration: Seconds to read at most.

    Returns:
        A `(80, 3000)` log-mel spectrogram.
    """
    return to_features(
        processor,
        load_waveform(path, sample_rate=sample_rate, max_duration=max_duration),
        sample_rate=sample_rate,
    )
