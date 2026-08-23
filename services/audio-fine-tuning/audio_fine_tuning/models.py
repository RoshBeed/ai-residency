"""Whisper model loading, audio preprocessing and decoding.

Everything here wraps the `openai-whisper` package rather than defining an
architecture, because Part 1 of this week fine-tunes the released Whisper weights
rather than training a model from scratch.

One wrapper is doing real work rather than just renaming. `whisper.load_model` picks
its own device and only ever chooses between `"cuda"` and `"cpu"` — its source reads
`device = "cuda" if torch.cuda.is_available() else "cpu"`, with no MPS branch — so on
Apple Silicon it silently runs on CPU with an MPS GPU sitting idle. Passing a device
explicitly is what makes this repo use MPS at all. `get_device()` from `ai_residency_common` is
now the single source of truth for that choice across the portfolio.

Measured on an M1 Max at this repo's whisper-tiny scale (Pass 2c): decode ~1.0s on CPU
vs ~0.2s on MPS, one training step ~0.9s vs ~0.3s. Both the mel spectrogram
(`torch.stft`-based) and a full forward/backward/optimizer step run correctly on MPS,
so there is no reason to force CPU here.
"""

import torch
import whisper
from whisper.model import Whisper
from whisper.tokenizer import Tokenizer

from ai_residency_common import get_device, get_logger
from config import Config

__all__ = [
    "create_tokens",
    "decode_audio",
    "decode_tokens",
    "forward",
    "get_tokenizer",
    "load_whisper_model",
    "prepare_training_data",
    "preprocess_audio",
]

logger = get_logger(__name__)


def load_whisper_model(config: Config) -> tuple[Whisper, torch.device]:
    """Load the Whisper model onto the best available device.

    Args:
        config: Supplies `whisper_model_name`.

    Returns:
        The loaded model and the device it was placed on.
    """
    device = get_device()
    logger.info("Loading whisper-%s on %s", config.whisper_model_name, device)
    model: Whisper = whisper.load_model(config.whisper_model_name, device=device)
    return model, device


def get_tokenizer() -> Tokenizer:
    """Return the multilingual Whisper tokenizer."""
    tokenizer: Tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True)
    return tokenizer


def preprocess_audio(audio_path: str) -> torch.Tensor:
    """Turn an audio file into the log-mel spectrogram Whisper expects.

    Args:
        audio_path: Any file `ffmpeg` can decode. Whisper shells out to the system
            `ffmpeg` binary, so it has to be installed.

    Returns:
        A log-mel spectrogram, padded or trimmed to Whisper's fixed 30-second window.
    """
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel: torch.Tensor = whisper.log_mel_spectrogram(audio)
    return mel


def decode_audio(model: Whisper, device: torch.device, audio_path: str) -> str:
    """Transcribe an audio file.

    Args:
        model: A loaded Whisper model.
        device: Device the model is on.
        audio_path: File to transcribe.

    Returns:
        The decoded transcript.
    """
    mel = preprocess_audio(audio_path).to(device)
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)
    text: str = result.text
    return text


def create_tokens(tokenizer: Tokenizer, text: str) -> list[int]:
    """Build the token sequence Whisper's decoder is trained against.

    The prefix mirrors Whisper's own multitask format: start-of-transcript, then the
    language token, then the task token, then the text and end-of-transcript.

    Args:
        tokenizer: Multilingual tokenizer from `get_tokenizer`.
        text: Ground-truth transcript.

    Returns:
        Token ids including the special prefix and terminator.
    """
    ids: list[int] = [tokenizer.sot, tokenizer.language_token, tokenizer.transcribe]
    ids += tokenizer.encode(text)
    ids.append(tokenizer.eot)
    return ids


def prepare_training_data(
    tokenizer: Tokenizer, device: torch.device, audio_path: str, text: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a single-example training batch.

    Args:
        tokenizer: Multilingual tokenizer.
        device: Device to place the tensors on.
        audio_path: Audio to encode.
        text: Ground-truth transcript for that audio.

    Returns:
        A `(tokens, mel)` pair, each with a leading batch dimension of 1.
    """
    mel = preprocess_audio(audio_path).unsqueeze(0).to(device)
    tokens = torch.tensor(create_tokens(tokenizer, text)).unsqueeze(0).to(device)
    return tokens, mel


def forward(model: Whisper, tokens: torch.Tensor, mel: torch.Tensor) -> torch.Tensor:
    """Run one forward pass.

    Args:
        model: A loaded Whisper model.
        tokens: Decoder input token ids, shape `(batch, sequence)`.
        mel: Encoder input spectrogram, shape `(batch, n_mels, frames)`.

    Returns:
        Logits of shape `(batch, sequence, vocab)`.
    """
    logits: torch.Tensor = model(tokens=tokens, mel=mel)
    return logits


def decode_tokens(tokenizer: Tokenizer, token_ids: list[int]) -> str:
    """Decode token ids back to text.

    Args:
        tokenizer: Multilingual tokenizer.
        token_ids: Ids to decode.

    Returns:
        The decoded string.
    """
    text: str = tokenizer.decode(token_ids)
    return text
