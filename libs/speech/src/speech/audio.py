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

__all__ = [
    "SAMPLE_RATE",
    "TARGET_RMS",
    "add_noise",
    "features_from_file",
    "load_waveform",
    "rms_normalise",
    "to_features",
]

SAMPLE_RATE = 16_000

# The level every clip is scaled to before featurisation. Chosen as roughly the median
# RMS of the CREMA-D corpus, so the corpus itself barely moves and everything else is
# brought to meet it.
TARGET_RMS = 0.05


def rms_normalise(waveform: np.ndarray, target_rms: float = TARGET_RMS) -> np.ndarray:
    """Scale a waveform to a fixed RMS level.

    **Recording level is a constant in a corpus and a free variable in an upload.**
    CREMA-D is studio audio at a consistent level; a phone held at arm's length is not.
    Whisper's feature extractor does normalise its log-mel, but against each clip's own
    maximum, which does not rescue a quiet recording whose detail has sunk toward the
    floor. Measured on 150 held-out CREMA-D clips with the shipped emotion checkpoint:
    at a quarter of the recorded volume accuracy fell from 82.0% to 58.7%, and at a
    sixteenth to **25.3%, which is near the 16.7% chance line for six emotions**. With
    this, both read 83.3% -- and the unperturbed clips go up slightly too.

    It is here, in the one function both services and all four of their training and
    serving paths go through, rather than in either service: a normalisation applied on
    one side and not the other is the bug this repo keeps finding.

    This does not help with noise, and is not meant to -- scaling a noisy clip scales
    its noise. That needs augmentation during training.

    Args:
        waveform: 1-D audio samples.
        target_rms: Level to scale to.

    Returns:
        The waveform at `target_rms`, or unchanged if it is silent.
    """
    rms = float(np.sqrt(np.mean(np.asarray(waveform, dtype=np.float64) ** 2)))
    if rms < 1e-8:
        return np.asarray(waveform, dtype=np.float32)
    return (np.asarray(waveform, dtype=np.float32) * (target_rms / rms)).astype(np.float32)


"""What Whisper was trained at. Resampling to it is not optional."""


def add_noise(waveform: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add white noise at a given signal-to-noise ratio.

    A corpus recorded in a booth has a clean floor; a room does not. Scaling the clip
    cannot recover from this -- see `rms_normalise` -- so a model that has to survive an
    upload has to have met noise during training.

    Args:
        waveform: 1-D audio samples.
        snr_db: Signal-to-noise ratio in decibels. Lower is noisier.
        rng: Source of randomness, passed in so a run is reproducible.

    Returns:
        The waveform with noise added, at the same length and dtype.
    """
    signal = np.asarray(waveform, dtype=np.float32)
    power = float(np.mean(signal.astype(np.float64) ** 2))
    if power < 1e-12:
        return signal
    noise_power = power / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), len(signal)).astype(np.float32)
    return signal + noise


def load_waveform(
    path: str | Path,
    *,
    sample_rate: int = SAMPLE_RATE,
    max_duration: float | None = None,
    normalise: bool = True,
) -> np.ndarray:
    """Read an audio file as a mono waveform at Whisper's sample rate.

    librosa rather than a shell-out to `ffmpeg`, which is what `openai-whisper` does
    and what used to make the system binary a deployment requirement for one of these
    services.

    Args:
        path: Any file librosa can read.
        sample_rate: Target rate. Leave at the default unless you know why.
        max_duration: Seconds to read at most. None reads the file.
        normalise: Scale to `TARGET_RMS`. On by default -- see `rms_normalise` for
            what an unnormalised upload costs.

    Returns:
        A 1-D float32 waveform.
    """
    waveform, _ = librosa.load(str(path), sr=sample_rate, duration=max_duration)
    if normalise:
        waveform = rms_normalise(waveform)
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
