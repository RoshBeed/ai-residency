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


def encode_target(
    processor: WhisperProcessor,
    text: str,
    *,
    decoder_start_token_id: int | None = None,
) -> torch.Tensor:
    """Encode the text the decoder should learn to produce.

    Drops the leading `<|startoftranscript|>` the tokenizer emits, and that is the
    whole reason this function exists rather than being one inline call.

    `WhisperForConditionalGeneration` builds its decoder input by shifting the labels
    right and prepending `decoder_start_token_id`, which *is*
    `<|startoftranscript|>`. Hand it labels that already begin with that token and the
    decoder input starts `[SOT, SOT, ...]` -- the model is trained one position out of
    step with what it will be asked at inference, and every loss is computed against a
    sequence shifted by one.

    It does not fail. It trains, the loss goes down, and the model learns to emit
    `<|startoftranscript|>` at the position you will later read its answer from.
    Measured on whisper-tiny against one clip: 7.4856 with the doubled token, 3.9923
    without -- and 3.9923 is exactly what `openai-whisper` computes for the same pair,
    which is how the equivalence was confirmed.

    Args:
        processor: Supplies the tokenizer.
        text: Ground truth -- a transcript, or a single token like `"<|happy|>"` if
            the tokenizer has been extended with it.
        decoder_start_token_id: The id the model prepends. Defaults to the
            tokenizer's own `<|startoftranscript|>`, which is the same thing for every
            Whisper checkpoint; pass the model's `config.decoder_start_token_id` to be
            certain.

    Returns:
        Token ids of shape `(sequence,)`, ready to pass as `labels`.
    """
    encoded = processor.tokenizer(text, add_special_tokens=True, return_tensors="pt")
    labels: torch.Tensor = encoded.input_ids[0]

    start = decoder_start_token_id
    if start is None:
        start = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    if labels.numel() and int(labels[0]) == start:
        labels = labels[1:]
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
