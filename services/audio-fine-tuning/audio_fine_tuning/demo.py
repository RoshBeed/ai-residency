"""CLI demo: what one gradient step actually changes.

Transcribes the bundled clip with released Whisper-tiny weights, takes a single
fine-tuning step toward the correct transcript, then transcribes the same clip again.
The two lines it prints are the clearest artifact this repo produces — the base model
mishears the name, and one step is enough to fix it on this example.

This is the same comparison the FastAPI service and the Streamlit UI produce, sharing
the same functions rather than reimplementing them.
"""

from ai_residency_common import get_logger, set_seed

from audio_fine_tuning.models import decode_audio, get_tokenizer, load_whisper_model
from audio_fine_tuning.training import fine_tune
from config import Config

__all__ = ["main", "run_demo"]

logger = get_logger(__name__)


def run_demo(config: Config) -> tuple[str, str]:
    """Transcribe the demo clip before and after one fine-tuning step.

    Args:
        config: Supplies the model name, the clip path, the target text and the
            learning rate.

    Returns:
        The base transcript and the fine-tuned transcript.
    """
    set_seed(config.seed)
    audio_path = str(config.audio_path)

    model, device = load_whisper_model(config)
    tokenizer = get_tokenizer()

    base_output = decode_audio(model, device, audio_path)
    model = fine_tune(
        model,
        tokenizer,
        device,
        audio_path,
        config.target_text,
        learning_rate=config.learning_rate,
    )
    fine_tuned_output = decode_audio(model, device, audio_path)

    return base_output, fine_tuned_output


def main() -> None:
    """Entry point for `make demo`."""
    config = Config()
    base_output, fine_tuned_output = run_demo(config)
    logger.info("Base model output: %s", base_output)
    logger.info("Fine-tuned model output: %s", fine_tuned_output)


if __name__ == "__main__":
    main()
