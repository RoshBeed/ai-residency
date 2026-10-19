"""Whisper loading, features, labels and decoding, shared by the speech services.

Both `whisper-fine-tuning` and `speech-emotion-recognition` fine-tune Whisper. They
had entirely separate implementations -- one on `openai-whisper`, one on
`transformers` -- because different teams built them in different weeks, not because
the tasks needed different tools.

    from speech import features_from_file, load_whisper, transcribe

    whisper = load_whisper("openai/whisper-tiny")
    features = features_from_file(whisper.processor, "clip.wav")
    print(transcribe(whisper, features))

Adding tokens the model was never trained on -- an emotion label the decoder should
learn to emit -- is the same call plus one argument, and the embedding table is
resized to match so the two cannot disagree:

    whisper = load_whisper("openai/whisper-tiny", extra_tokens=["<|happy|>", "<|sad|>"])

Training loops stay in the services. They genuinely differ: one takes five gradient
steps on a single clip and streams a decode after each, the other runs epochs over a
corpus with evaluation. What they share is everything above.
"""

from speech.audio import (
    SAMPLE_RATE,
    add_noise,
    features_from_file,
    load_waveform,
    rms_normalise,
    to_features,
)
from speech.inference import transcribe
from speech.loading import Whisper, load_whisper
from speech.targets import IGNORE_INDEX, collate, encode_target

__all__ = [
    "IGNORE_INDEX",
    "SAMPLE_RATE",
    "Whisper",
    "add_noise",
    "collate",
    "encode_target",
    "features_from_file",
    "load_waveform",
    "load_whisper",
    "rms_normalise",
    "to_features",
    "transcribe",
]
