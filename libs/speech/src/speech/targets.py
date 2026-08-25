"""Text in, decoder labels out -- and the padding rule that goes with them.

Whisper's decoder is trained on token ids that carry a prefix of their own:
start-of-transcript, then language and task, then the text. The tokenizer emits that
prefix itself, which is worth knowing because one of these services used to build it
by hand against `openai-whisper`'s special-token ids.
"""

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import WhisperProcessor

__all__ = ["IGNORE_INDEX", "collate", "encode_target"]

IGNORE_INDEX = -100
"""Label positions to skip when computing loss.

Cross-entropy ignores this index, so padded positions contribute nothing rather than
teaching the decoder to predict padding. Padding with anything else -- 0, or the
tokenizer's pad id -- trains the model to emit it.
"""


def encode_target(processor: WhisperProcessor, text: str) -> torch.Tensor:
    """Encode the text the decoder should learn to produce.

    Args:
        processor: Supplies the tokenizer.
        text: Ground truth -- a transcript, or a single token like `"<|happy|>"` if
            the tokenizer has been extended with it.

    Returns:
        Token ids of shape `(sequence,)`, including Whisper's own special prefix.
    """
    encoded = processor.tokenizer(text, add_special_tokens=True, return_tensors="pt")
    labels: torch.Tensor = encoded.input_ids[0]
    return labels


def collate(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack a batch of `(features, labels)` pairs.

    Features are a fixed shape so they stack directly; labels vary in length and are
    padded with `IGNORE_INDEX`.

    Args:
        batch: Pairs from a dataset.

    Returns:
        `(features, labels)`, batched.
    """
    features = torch.stack([features for features, _ in batch])
    labels = pad_sequence(
        [labels for _, labels in batch], batch_first=True, padding_value=IGNORE_INDEX
    )
    return features, labels
