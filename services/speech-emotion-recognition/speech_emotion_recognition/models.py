"""Loading Whisper with the emotion vocabulary attached.

The one thing that has to stay true here: the tokenizer and the model's embedding table
must agree about how many tokens exist. `add_emotion_tokens` extends the vocabulary by
one token per emotion, so a model loaded against the base vocabulary has no rows for
them -- and indexing an embedding table out of range is a crash at best and, if the ids
happen to land inside the table, silently wrong at worst. `resize_token_embeddings` is
what keeps the two in step, so it happens in the same function that loads either one.
"""

from pathlib import Path

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from ai_residency_common import get_device, get_logger
from config import Config

__all__ = ["load_model_and_processor", "load_processor"]

logger = get_logger(__name__)


def load_processor(config: Config) -> WhisperProcessor:
    """Load the processor, preferring the emotion-extended tokenizer.

    Args:
        config: Supplies `tokenizer_dir` and `model_name`.

    Returns:
        A processor whose tokenizer knows the emotion tokens, if the extended tokenizer
        has been written; otherwise the base one, which is enough for feature
        extraction but will not encode an emotion label.
    """
    processor: WhisperProcessor = WhisperProcessor.from_pretrained(config.model_name)
    if config.tokenizer_dir.exists():
        processor.tokenizer.add_special_tokens(
            {"additional_special_tokens": list(config.emotion_tokens)}
        )
        logger.info("Extended the tokenizer with %d emotion tokens", len(config.emotion_tokens))
    else:
        logger.warning(
            "No extended tokenizer at %s; run `python -m speech_emotion_recognition."
            "add_emotion_tokens` first",
            config.tokenizer_dir,
        )
    return processor


def load_model_and_processor(
    config: Config, checkpoint_path: Path | None = None
) -> tuple[WhisperForConditionalGeneration, WhisperProcessor, torch.device]:
    """Load Whisper, resize it to the extended vocabulary, and optionally restore weights.

    Args:
        config: Run configuration.
        checkpoint_path: Fine-tuned weights to load. None starts from the pretrained
            model, which is what training does.

    Returns:
        The model in eval mode on the selected device, the processor, and the device.
    """
    processor = load_processor(config)
    model = WhisperForConditionalGeneration.from_pretrained(config.model_name)
    model.resize_token_embeddings(len(processor.tokenizer))

    if checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state.get("model_state_dict", state))
        logger.info("Loaded fine-tuned weights from %s", checkpoint_path)

    device = get_device()
    model = model.to(device)
    model.eval()
    return model, processor, device
