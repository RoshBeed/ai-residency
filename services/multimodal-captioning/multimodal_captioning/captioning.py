"""Single-image captioning inference.

The inference half of the same pipeline `training` fits: CLIP encodes the image, the
trained adapter projects it into one soft token, and Qwen3 continues from that token
plus a short text prompt. Both the Streamlit UI and the FastAPI service call
`generate_caption`, so there is one inference path in this repo rather than three.

Greedy decoding by default (`do_sample=False`), which is what the README's results table
was produced with. Sampling makes captions read better and makes them impossible to
compare between checkpoints.

`caption_image` uses HuggingFace's batched `generate()` for the full sequence at once.
`caption_image_with_callback` runs a manual autoregressive loop that calls
`on_token(step, token_str, probability)` after each greedy pick, so an SSE endpoint can
relay every token the moment the model emits it rather than waiting for the whole
caption.
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch
from PIL.Image import Image
from torch import nn

from ai_residency_common import get_logger
from config import Config
from multimodal_captioning.models import (
    Backbones,
    image_soft_tokens,
    load_adapter,
    load_backbones,
)

__all__ = [
    "CaptionResult",
    "TokenTrace",
    "caption_image",
    "caption_image_with_callback",
    "generate_caption",
]

logger = get_logger(__name__)


@dataclass
class TokenTrace:
    """One generated token and how confident the model was in picking it."""

    token: str
    probability: float


@dataclass
class CaptionResult:
    """A generated caption plus its token-by-token trace.

    `token_trace` has one entry per generated token, in generation order, so a UI can
    replay the caption being written out token by token alongside each token's own
    softmax probability (of the token actually picked, since decoding is greedy).
    """

    caption: str
    token_trace: list[TokenTrace]


def caption_image(
    image: Image,
    adapter: nn.Module,
    backbones: Backbones,
    config: Config,
    *,
    prompt: str | None = None,
    max_new_tokens: int | None = None,
) -> CaptionResult:
    """Caption one image with an already-loaded adapter.

    Args:
        image: Input image; converted to RGB first, since CLIP's processor expects three
            channels and some sample files are palettised.
        adapter: Trained projection from CLIP's space into Qwen3's.
        backbones: Frozen CLIP and Qwen3.
        config: Supplies the default prompt and generation length.
        prompt: Text the image's soft token is prepended to. Defaults to `config.prompt`.
        max_new_tokens: Generation cap. Defaults to `config.max_new_tokens`.

    Returns:
        The generated caption, stripped, plus its per-token trace. Decoding is greedy
        (`do_sample=False`), so each token's probability is how confident the model was
        in the token it actually picked, not a sampling weight.
    """
    device = backbones.device
    prompt = config.prompt if prompt is None else prompt
    max_new_tokens = config.max_new_tokens if max_new_tokens is None else max_new_tokens

    processed = backbones.image_processor(image.convert("RGB"), return_tensors="pt")
    image_inputs = {key: value.to(device) for key, value in processed.items()}

    with torch.no_grad():
        image_latent = image_soft_tokens(adapter, backbones, image_inputs)

        input_ids = backbones.qwen_tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        text_embeddings = backbones.qwen_model.model.embed_tokens(input_ids)

        full_embeddings = torch.cat([image_latent, text_embeddings], dim=1)
        attention_mask = torch.ones(full_embeddings.shape[:2], dtype=torch.long, device=device)

        generation = backbones.qwen_model.generate(
            inputs_embeds=full_embeddings,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=backbones.qwen_tokenizer.pad_token_id,
            eos_token_id=backbones.qwen_tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )

    generated_ids = generation.sequences
    token_trace: list[TokenTrace] = []
    for step_logits, token_id in zip(generation.scores, generated_ids[0], strict=False):
        step_probabilities = torch.softmax(step_logits[0], dim=-1)
        token_trace.append(
            TokenTrace(
                token=backbones.qwen_tokenizer.decode([token_id]),
                probability=step_probabilities[token_id].item(),
            )
        )

    caption = backbones.qwen_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return CaptionResult(caption=str(caption).strip(), token_trace=token_trace)


def caption_image_with_callback(
    image: Image,
    adapter: nn.Module,
    backbones: Backbones,
    config: Config,
    *,
    prompt: str | None = None,
    max_new_tokens: int | None = None,
    on_token: Callable[[int, str, float], None] | None = None,
) -> CaptionResult:
    """Caption one image, calling `on_token(step, token_str, probability)` per token.

    Same output as `caption_image`, but the generation runs as a manual autoregressive
    loop instead of a single `model.generate()` call. Each greedy token pick fires
    `on_token` the moment the model emits it, so an SSE caller can show the caption
    being written token by token rather than waiting for the full sequence.

    Args:
        image: Input image; converted to RGB first.
        adapter: Trained projection from CLIP's space into Qwen3's.
        backbones: Frozen CLIP and Qwen3.
        config: Supplies the default prompt and generation length.
        prompt: Text the image's soft token is prepended to.
        max_new_tokens: Generation cap.
        on_token: Called as `on_token(step, token_str, probability)` after each
            greedy token pick, with `step` zero-indexed from the first generated
            token. May be None if no per-token relay is needed.

    Returns:
        The generated caption plus its per-token trace.
    """
    device = backbones.device
    prompt = config.prompt if prompt is None else prompt
    max_new_tokens = config.max_new_tokens if max_new_tokens is None else max_new_tokens

    processed = backbones.image_processor(image.convert("RGB"), return_tensors="pt")
    image_inputs = {key: value.to(device) for key, value in processed.items()}

    with torch.no_grad():
        image_latent = image_soft_tokens(adapter, backbones, image_inputs)

        input_ids = backbones.qwen_tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        text_embeddings = backbones.qwen_model.model.embed_tokens(input_ids)

        full_embeddings = torch.cat([image_latent, text_embeddings], dim=1)
        attention_mask = torch.ones(full_embeddings.shape[:2], dtype=torch.long, device=device)

        outputs = backbones.qwen_model(
            inputs_embeds=full_embeddings,
            attention_mask=attention_mask,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :]

        generated_ids: list[int] = []
        token_trace: list[TokenTrace] = []

        for step in range(max_new_tokens):
            probs = torch.softmax(logits, dim=-1)
            next_token_id = int(torch.argmax(probs, dim=-1).item())
            prob_value = float(probs[0, next_token_id].item())

            generated_ids.append(next_token_id)
            token_str = backbones.qwen_tokenizer.decode([next_token_id])
            token_trace.append(TokenTrace(token=token_str, probability=prob_value))

            if on_token is not None:
                on_token(step, token_str, prob_value)

            if next_token_id == backbones.qwen_tokenizer.eos_token_id:
                break

            next_embed = backbones.qwen_model.model.embed_tokens(
                torch.tensor([[next_token_id]], device=device)
            )
            outputs = backbones.qwen_model(
                inputs_embeds=next_embed,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

    caption = backbones.qwen_tokenizer.decode(generated_ids, skip_special_tokens=True)
    return CaptionResult(caption=str(caption).strip(), token_trace=token_trace)


def generate_caption(
    image: Image,
    config: Config | None = None,
    *,
    checkpoint_name: str | None = None,
    prompt: str | None = None,
    max_new_tokens: int | None = None,
) -> CaptionResult:
    """Caption an image, loading the backbones and the requested checkpoint as needed.

    Args:
        image: Input image.
        config: Run configuration. A default `Config` is used when omitted.
        checkpoint_name: Adapter checkpoint under `checkpoint_dir`. Defaults to
            `config.checkpoint_name`.
        prompt: Text the image's soft token is prepended to.
        max_new_tokens: Generation cap.

    Returns:
        The generated caption plus its per-token trace.
    """
    config = Config() if config is None else config
    checkpoint_name = config.checkpoint_name if checkpoint_name is None else checkpoint_name

    backbones = load_backbones(config)
    adapter = load_adapter(config, checkpoint_name, backbones)
    return caption_image(
        image,
        adapter,
        backbones,
        config,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
    )
