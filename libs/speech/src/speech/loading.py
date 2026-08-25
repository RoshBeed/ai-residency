"""Loading Whisper, optionally with tokens it was never trained on.

Both speech services fine-tune Whisper. They arrived at it from different weeks and
different teams, so one reached for `openai-whisper` and the other for
`transformers`, and they ended up sharing nothing. There was never a technical
reason for the split: `transformers` does everything `openai-whisper` does here --
its feature extractor produces a *bit-identical* log-mel spectrogram -- and one
thing it cannot, which is extend the vocabulary.

That last point decides the direction. Adding tokens to `openai-whisper`'s
tiktoken-based tokenizer and resizing the model to match is not something it
supports; `transformers` does it in two calls. So this is built on `transformers`,
and dropping `openai-whisper` also drops a system `ffmpeg` requirement -- it shells
out to the binary for `load_audio`, where librosa does not.

The one invariant worth centralising is here rather than in either service: **the
tokenizer and the model's embedding table must agree about how many tokens exist.**
Add tokens to the tokenizer without resizing the model and the new ids index past
the end of the embedding table -- a crash if you are lucky, and silently wrong if
the ids happen to land inside it.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from ai_residency_common import get_device, get_logger

__all__ = ["Whisper", "load_whisper"]

logger = get_logger(__name__)


@dataclass
class Whisper:
    """A loaded Whisper, its processor, and the device they are on.

    Kept together because they have to stay consistent: the processor's tokenizer
    decides how many tokens exist, and the model's embedding table has to have a row
    for each of them.

    Attributes:
        model: The conditional-generation model, on `device`.
        processor: Feature extractor plus tokenizer.
        device: Where the model lives.
    """

    model: WhisperForConditionalGeneration
    processor: WhisperProcessor
    device: torch.device

    @property
    def vocab_size(self) -> int:
        """Rows in the model's input embedding table."""
        return int(self.model.get_input_embeddings().weight.shape[0])


def load_whisper(
    model_name: str,
    *,
    device: torch.device | None = None,
    extra_tokens: Sequence[str] = (),
    checkpoint: Path | None = None,
) -> Whisper:
    """Load Whisper, extend its vocabulary if asked, and put it on a device.

    Args:
        model_name: A Hugging Face id, e.g. `"openai/whisper-tiny"`.
        device: Where to put the model. Defaults to `get_device()`, which is the
            portfolio's single answer to that question -- and the reason this does
            not use `openai-whisper`, whose loader picks between CUDA and CPU only
            and silently ignores MPS.
        extra_tokens: Tokens to add as special tokens before loading weights, e.g.
            emotion labels the decoder should learn to emit. The model's embedding
            table is resized to match, which is the whole point of doing both here.
        checkpoint: Fine-tuned weights to restore. Loaded *after* any resize, so a
            checkpoint that already includes the extra tokens fits.

    Returns:
        The model, its processor and the device, as one object.

    Raises:
        ValueError: If a checkpoint's vocabulary does not match the tokenizer's,
            which means it was trained against a different token set.
    """
    resolved = get_device() if device is None else device

    processor: WhisperProcessor = WhisperProcessor.from_pretrained(model_name)
    if extra_tokens:
        added = processor.tokenizer.add_special_tokens(
            {"additional_special_tokens": list(extra_tokens)}
        )
        logger.info("Added %d token(s) to the tokenizer: %s", added, ", ".join(extra_tokens))

    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    # Always resize, even when nothing was added: `from_pretrained` can return a
    # model whose table is larger than the tokenizer (Whisper pads its vocabulary),
    # and resizing to the tokenizer's length is what makes the two agree either way.
    if extra_tokens:
        model.resize_token_embeddings(len(processor.tokenizer))
        logger.info("Resized embeddings to %d rows", model.get_input_embeddings().weight.shape[0])

    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = state.get("model_state_dict", state)
        expected = int(model.get_input_embeddings().weight.shape[0])
        found = _vocab_rows(state)
        if found is not None and found != expected:
            raise ValueError(
                f"{checkpoint} was trained against a {found}-token vocabulary but this "
                f"model has {expected} rows. Load it with the same `extra_tokens` it "
                f"was trained with."
            )
        model.load_state_dict(state)
        logger.info("Restored weights from %s", checkpoint)

    model.to(resolved)
    return Whisper(model=model, processor=processor, device=resolved)


def _vocab_rows(state: dict[str, torch.Tensor]) -> int | None:
    """Rows in a state dict's token embedding, or None if it has no recognisable one."""
    for key in ("model.decoder.embed_tokens.weight", "proj_out.weight"):
        if key in state:
            return int(state[key].shape[0])
    return None
