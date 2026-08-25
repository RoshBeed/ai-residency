"""Turning encoder features back into text."""

import torch

from speech.loading import Whisper

__all__ = ["transcribe"]


def transcribe(
    whisper: Whisper,
    features: torch.Tensor,
    *,
    max_new_tokens: int | None = None,
    skip_special_tokens: bool = True,
) -> str:
    """Decode one clip's features to text.

    Runs under `no_grad` and leaves the model in whatever train/eval mode it was in.
    That matters for the fine-tuning demo, which decodes *between* gradient steps: a
    helper that flipped the model to eval and left it there would silently disable
    dropout for the rest of training.

    Args:
        whisper: A loaded model and processor.
        features: `(80, 3000)` for one clip, or `(batch, 80, 3000)`.
        max_new_tokens: Cap on generated length. None uses the model's own default.
        skip_special_tokens: Drop `<|startoftranscript|>` and friends from the output.
            Pass False when the token itself is the prediction -- an emotion label,
            for instance.

    Returns:
        The decoded text for the first item in the batch.
    """
    if features.dim() == 2:
        features = features.unsqueeze(0)

    was_training = whisper.model.training
    whisper.model.eval()
    try:
        with torch.no_grad():
            kwargs = {} if max_new_tokens is None else {"max_new_tokens": max_new_tokens}
            generated = whisper.model.generate(features.to(whisper.device), **kwargs)
    finally:
        whisper.model.train(was_training)

    decoded: list[str] = whisper.processor.batch_decode(
        generated, skip_special_tokens=skip_special_tokens
    )
    return decoded[0]
