"""Extend the Whisper tokenizer with one special token per emotion.

This is the piece that makes emotion a *prediction* rather than a separate classifier
head. Whisper's decoder already emits special tokens for language and task; adding
`<emotion_happy>` and friends to the same vocabulary lets the model emit an emotion
label through the same mechanism it already uses, so the fine-tuning target becomes
"transcript plus emotion" rather than "transcript" alone.

The extended tokenizer is written under `checkpoints/` because it is a trained
artifact of this pipeline, not source.
"""

from ai_residency_common import get_logger
from transformers import WhisperTokenizer

from config import Config

__all__ = ["add_emotion_tokens", "main"]

logger = get_logger(__name__)


def add_emotion_tokens(config: Config) -> int:
    """Add the emotion tokens to the base tokenizer and save the result.

    Args:
        config: Supplies `tokenizer_model_name`, `emotion_tokens` and `tokenizer_dir`.

    Returns:
        How many tokens were actually new to the vocabulary.
    """
    tokenizer = WhisperTokenizer.from_pretrained(config.tokenizer_model_name)
    num_added: int = tokenizer.add_special_tokens(
        {"additional_special_tokens": list(config.emotion_tokens)}
    )
    logger.info("Added %d special tokens: %s", num_added, ", ".join(config.emotion_tokens))

    config.tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(config.tokenizer_dir))
    logger.info("Saved updated tokenizer to %s", config.tokenizer_dir)
    return num_added


def main() -> None:
    """Entry point for the `add-tokens` pipeline stage."""
    add_emotion_tokens(Config())


if __name__ == "__main__":
    main()
