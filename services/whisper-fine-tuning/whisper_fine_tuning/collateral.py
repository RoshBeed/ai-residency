"""What the single-example fine-tune costs on speech it never saw.

This service deliberately overfits one clip, so the demo on its own cannot say whether
the model is still a usable transcriber afterwards. This is the counterweight: transcribe
other speech before and after the step and report word error rate against the baseline.

CREMA-D is the corpus for it because every clip carries its own ground truth. The
corpus is twelve fixed sentences read by 91 actors, and the sentence is encoded in the
filename, so a known reference comes free with no annotation. The clips come from the
sibling speech-emotion service's `make download-data`, which is why `clip_dir` is an
argument rather than something this service pretends to own.

What it measures: one step is free (WER unchanged, no sentence altered), five steps are
not (0.1642 to 0.2687, four of twelve sentences altered). The shipped `num_epochs` is 1.
"""

import re
import sys
from pathlib import Path

from ai_residency_common import get_logger, resolve_required, set_seed
from config import Config
from speech import Whisper
from whisper_fine_tuning.models import decode_audio, load_whisper_model
from whisper_fine_tuning.training import fine_tune

__all__ = ["SENTENCES", "main", "pick_clips", "score", "word_errors"]

logger = get_logger(__name__)

# CREMA-D's twelve sentences, keyed by the code that appears in every filename.
SENTENCES = {
    "IEO": "It's eleven o'clock",
    "TIE": "That is exactly what happened",
    "IOM": "I'm on my way to the meeting",
    "IWW": "I wonder what this is about",
    "TAI": "The airplane is almost full",
    "MTI": "Maybe tomorrow it will be cold",
    "IWL": "I would like a new alarm clock",
    "ITH": "I think I have a doctor's appointment",
    "DFA": "Don't forget a jacket",
    "ITS": "I think I've seen this before",
    "TSI": "The surface is slick",
    "WSI": "We'll stop in a couple of minutes",
}

DEFAULT_CLIP_DIR = Path("../speech-emotion-recognition/datasets/crema_d/AudioWAV")


def normalise(text: str) -> list[str]:
    """Lowercase, drop punctuation, split into words.

    Punctuation is dropped on purpose. The fine-tuning target has no final period and
    the model stops emitting one after the step, which changes every transcript's raw
    text while changing no word. Counting that as an error would report damage where
    there is none.
    """
    return re.sub(r"[^a-z0-9' ]", " ", text.lower()).split()


def word_errors(reference: str, hypothesis: str) -> tuple[int, int]:
    """Word-level Levenshtein distance, and the reference length to divide it by.

    Args:
        reference: The known transcript.
        hypothesis: What the model produced.

    Returns:
        A `(distance, reference_words)` pair, so a caller can pool several
        sentences into one corpus WER rather than averaging per-sentence rates.
    """
    reference_words, hypothesis_words = normalise(reference), normalise(hypothesis)
    previous = list(range(len(hypothesis_words) + 1))
    for i, want in enumerate(reference_words, start=1):
        current = [i]
        for j, got in enumerate(hypothesis_words, start=1):
            current.append(
                previous[j - 1]
                if want == got
                else 1 + min(previous[j - 1], previous[j], current[j - 1])
            )
        previous = current
    return previous[-1], len(reference_words)


def pick_clips(clip_dir: Path) -> list[tuple[str, Path]]:
    """One neutral-emotion clip per sentence, lowest actor id first.

    Args:
        clip_dir: Directory of CREMA-D `.wav` files.

    Returns:
        A `(sentence_code, path)` pair per sentence found. Sorted and deterministic, so
        the reported WER is the same set of clips every run.
    """
    chosen = []
    for code in SENTENCES:
        matches = sorted(clip_dir.glob(f"*_{code}_NEU_XX.wav"))
        if matches:
            chosen.append((code, matches[0]))
    return chosen


def score(whisper: Whisper, clips: list[tuple[str, Path]]) -> tuple[float, list[tuple[str, str]]]:
    """Transcribe every clip and pool the errors into one WER.

    Args:
        whisper: A loaded Whisper model.
        clips: `(sentence_code, path)` pairs from `pick_clips`.

    Returns:
        The corpus WER and the transcripts behind it.
    """
    errors = total = 0
    transcripts = []
    for code, path in clips:
        hypothesis = decode_audio(whisper, str(path))
        distance, length = word_errors(SENTENCES[code], hypothesis)
        errors += distance
        total += length
        transcripts.append((code, hypothesis.strip()))
    return errors / total, transcripts


def main() -> None:
    """Entry point for `make collateral`. Optional args: step count, clip directory."""
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    clip_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_CLIP_DIR

    config = Config()
    set_seed(config.seed)

    clips = pick_clips(clip_dir)
    if not clips:
        logger.error(
            "No CREMA-D clips under %s. Run `make -C ../speech-emotion-recognition "
            "download-data`, or pass a directory of *_<CODE>_NEU_XX.wav files.",
            clip_dir,
        )
        raise SystemExit(1)

    resolved, source = resolve_required(
        config.audio_path,
        config.hf_dataset_repo_id,
        repo_type="dataset",
        rebuild_hint="Supply your own clip at data/audio.mp3, or record one.",
    )
    logger.info("Demo clip from %s; %d held-out sentences, %d step(s)", source, len(clips), steps)

    whisper = load_whisper_model(config)
    demo_clip = str(resolved)

    base_demo = decode_audio(whisper, demo_clip)
    base_wer, base_transcripts = score(whisper, clips)

    whisper = fine_tune(
        whisper,
        demo_clip,
        config.target_text,
        learning_rate=config.learning_rate,
        num_steps=steps,
    )

    tuned_demo = decode_audio(whisper, demo_clip)
    tuned_wer, tuned_transcripts = score(whisper, clips)

    print(f"\ndemo clip  base:  {base_demo.strip()!r}")
    print(f"demo clip  tuned: {tuned_demo.strip()!r}")
    print(f"target:           {config.target_text!r}\n")

    altered = 0
    print(f"{'code':5} {'base':40} after {steps} step(s)")
    for (code, before), (_, after) in zip(base_transcripts, tuned_transcripts, strict=True):
        flag = "" if normalise(before) == normalise(after) else "  <- words changed"
        altered += bool(flag)
        print(f"{code:5} {before[:38]:40} {after[:38]:40}{flag}")

    print(f"\nWER over {len(clips)} held-out CREMA-D sentences:")
    print(f"  base                {base_wer:.4f}")
    print(f"  after {steps} step(s)      {tuned_wer:.4f}")
    print(f"  delta               {tuned_wer - base_wer:+.4f}")
    print(f"  sentences altered   {altered}/{len(clips)}")


if __name__ == "__main__":
    main()
