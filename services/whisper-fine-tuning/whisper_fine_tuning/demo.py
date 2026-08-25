"""CLI demo: what one gradient step actually changes.

Transcribes the bundled clip with released Whisper-tiny weights, takes a single
fine-tuning step toward the correct transcript, then transcribes the same clip again.
The two lines it prints are the clearest artifact this repo produces — the base model
mishears the name, and one step is enough to fix it on this example.

This is the same comparison the FastAPI service and the Streamlit UI produce, sharing
the same functions rather than reimplementing them.
"""

from ai_residency_common import get_logger, resolve_required, set_seed
from config import Config
from whisper_fine_tuning.models import decode_audio, load_whisper_model
from whisper_fine_tuning.training import fine_tune

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
    # The clip is not committed: the runtime is ephemeral, so it comes from the
    # Hub like every other artifact, downloaded into config.audio_path when absent.
    resolved, source = resolve_required(
        config.audio_path,
        config.hf_dataset_repo_id,
        repo_type="dataset",
        rebuild_hint="Supply your own clip at data/audio.mp3, or record one.",
    )
    logger.info("Demo clip from %s", source)
    audio_path = str(resolved)

    whisper = load_whisper_model(config)

    base_output = decode_audio(whisper, audio_path)
    whisper = fine_tune(
        whisper,
        audio_path,
        config.target_text,
        learning_rate=config.learning_rate,
    )
    fine_tuned_output = decode_audio(whisper, audio_path)

    return base_output, fine_tuned_output


def main() -> None:
    """Entry point for `make demo`."""
    config = Config()
    base_output, fine_tuned_output = run_demo(config)
    logger.info("Base model output: %s", base_output)
    logger.info("Fine-tuned model output: %s", fine_tuned_output)


if __name__ == "__main__":
    main()
