"""Loading Whisper with the emotion vocabulary attached.

Thin over `speech.load_whisper`, which holds the invariant this used to hold on its
own: the tokenizer and the model's embedding table must agree about how many tokens
exist. `whisper-fine-tuning` needs the same thing, so it lives in the library both
services share rather than in one of them.
"""

from pathlib import Path

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from ai_residency_common import get_logger
from config import Config
from speech import Whisper, load_whisper

__all__ = ["load_emotion_whisper", "load_model_and_processor", "load_processor"]

logger = get_logger(__name__)


def load_emotion_whisper(config: Config, checkpoint_path: Path | None = None) -> Whisper:
    """Load Whisper with one added token per emotion, ready to fine-tune or serve.

    Args:
        config: Supplies `model_name` and `emotion_tokens`.
        checkpoint_path: Fine-tuned weights to restore. None starts from the
            pretrained model, which is what training does.

    Returns:
        The model, its processor and the device, with the embedding table already
        resized to cover the emotion tokens.
    """
    return load_whisper(
        config.model_name,
        extra_tokens=config.emotion_tokens,
        checkpoint=checkpoint_path,
    )


def load_processor(config: Config) -> WhisperProcessor:
    """Return just the processor, with the emotion tokens attached.

    Kept for the data pipeline, which needs to tokenize labels without paying to load
    a model.

    Args:
        config: Supplies `model_name` and `emotion_tokens`.

    Returns:
        A processor whose tokenizer knows the emotion tokens.
    """
    processor: WhisperProcessor = WhisperProcessor.from_pretrained(config.model_name)
    processor.tokenizer.add_special_tokens(
        {"additional_special_tokens": list(config.emotion_tokens)}
    )
    return processor


def load_model_and_processor(
    config: Config, checkpoint_path: Path | None = None
) -> tuple[WhisperForConditionalGeneration, WhisperProcessor, torch.device]:
    """Load model, processor and device, in eval mode.

    The older three-tuple shape, kept because the serving path unpacks it that way.

    Args:
        config: Run configuration.
        checkpoint_path: Fine-tuned weights to restore.

    Returns:
        The model in eval mode, the processor, and the device.
    """
    whisper = load_emotion_whisper(config, checkpoint_path)
    whisper.model.eval()
    return whisper.model, whisper.processor, whisper.device
