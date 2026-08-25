"""Whisper loading and decoding for the fine-tuning demo.

Thin over `libs/speech`, which both audio services now share. This used to wrap the
`openai-whisper` package instead, and moving off it removed three things rather than
adding any:

* the system `ffmpeg` binary, which `openai-whisper` shells out to in `load_audio`
  and which was therefore a deployment requirement for this service alone;
* a device workaround -- `whisper.load_model` chooses between CUDA and CPU with no
  MPS branch, so on Apple Silicon it ran on CPU with the GPU idle;
* the hand-built decoder prefix. `[sot, language, transcribe]` assembled from
  special-token ids is what the tokenizer emits by itself, and the hand-built version
  was missing `<|notimestamps|>`, so it trained against a slightly different sequence
  than Whisper's own format specifies.

The log-mel features are bit-identical between the two libraries, checked in
`libs/speech`'s tests, which is what made the move safe to make.
"""

import torch

from ai_residency_common import get_logger
from config import Config
from speech import Whisper, encode_target, features_from_file, load_whisper, transcribe

__all__ = [
    "decode_audio",
    "load_whisper_model",
    "prepare_training_data",
]

logger = get_logger(__name__)


def load_whisper_model(config: Config) -> Whisper:
    """Load Whisper-tiny onto the best available device.

    Args:
        config: Supplies `model_name`.

    Returns:
        The model, its processor and the device.
    """
    return load_whisper(config.model_name)


def decode_audio(whisper: Whisper, audio_path: str) -> str:
    """Transcribe an audio file.

    Args:
        whisper: A loaded model and processor.
        audio_path: File to transcribe.

    Returns:
        The decoded transcript.
    """
    return transcribe(whisper, features_from_file(whisper.processor, audio_path))


def prepare_training_data(
    whisper: Whisper, audio_path: str, text: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a single-example training batch.

    Args:
        whisper: Supplies the processor and the device.
        audio_path: Audio to encode.
        text: Ground-truth transcript for that audio.

    Returns:
        A `(features, labels)` pair, each with a leading batch dimension of 1.
    """
    features = features_from_file(whisper.processor, audio_path).unsqueeze(0)
    labels = encode_target(whisper.processor, text).unsqueeze(0)
    return features.to(whisper.device), labels.to(whisper.device)
