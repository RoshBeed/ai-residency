"""Model construction: the frozen backbones and the one trainable piece between them.

Only the adapter is trained. CLIP's vision tower turns an image into a 512-d embedding,
the adapter projects that into Qwen3's 1024-d embedding space as a single soft token,
and Qwen3 continues from it as if it had read a word. Both backbones stay frozen, which
is the whole point of the design: ~1.6M trainable parameters bridging a 596M-parameter
decoder.

`build_adapter` takes `final_layer_norm` rather than hardcoding a shape, and that flag
is what fixes this repo's longest-standing bug. `adapter_flickr_batched.pth` was saved
from a five-module adapter ending in a `LayerNorm`; that layer was later deleted from
the code, and nothing recorded that the checkpoint on disk still expected it, so every
subsequent load raised `Unexpected key(s) in state_dict: "4.weight", "4.bias"`. Each
checkpoint now carries its own architecture in `extra`, so a checkpoint can describe
the code that wrote it instead of the code having to guess.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
from ai_residency_common import get_device, get_logger, load_checkpoint, resolve_checkpoint
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPModel,
    CLIPTokenizer,
    PreTrainedModel,
)

from config import CHECKPOINTS, Config

__all__ = ["Backbones", "build_adapter", "load_adapter", "load_backbones"]

logger = get_logger(__name__)


@dataclass
class Backbones:
    """The frozen pretrained models both halves of this repo run on.

    Attributes:
        clip_model: Full CLIP model; also supplies the two projection heads.
        vision_encoder: CLIP's vision tower, a reference into `clip_model`.
        text_encoder: CLIP's text tower, used only by the zero-shot demo.
        image_processor: Turns a PIL image into CLIP's expected pixel tensor.
        clip_tokenizer: Tokenizer for the zero-shot demo's label prompts.
        qwen_model: The frozen decoder that writes the caption.
        qwen_tokenizer: Tokenizer for prompts and captions.
        clip_dim: Width of CLIP's projected image embedding, the adapter's input.
        qwen_dim: Width of Qwen3's token embeddings, the adapter's output.
        device: Where these models live. Carried on the dataclass so callers do not each
            re-derive it and risk putting the adapter somewhere the backbones are not.
    """

    clip_model: CLIPModel
    vision_encoder: PreTrainedModel
    text_encoder: PreTrainedModel
    image_processor: CLIPImageProcessor
    clip_tokenizer: CLIPTokenizer
    qwen_model: PreTrainedModel
    qwen_tokenizer: AutoTokenizer
    clip_dim: int
    qwen_dim: int
    device: torch.device


@lru_cache(maxsize=1)
def _load_backbones(clip_model_name: str, qwen_model_name: str, device: torch.device) -> Backbones:
    """Load and cache the backbones, keyed on the names and device asked for."""
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
    image_processor = CLIPImageProcessor.from_pretrained(clip_model_name)
    clip_tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)

    # Explicit float32. Qwen3's published weights default to bfloat16 under current
    # `transformers`, which then crashes the moment they meet the float32 adapter
    # output: `RuntimeError: mat1 and mat2 must have the same dtype`.
    qwen_model = AutoModelForCausalLM.from_pretrained(qwen_model_name, dtype=torch.float32).to(
        device
    )
    qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_model_name)

    clip_model.eval()
    qwen_model.eval()
    for parameter in clip_model.parameters():
        parameter.requires_grad = False
    for parameter in qwen_model.parameters():
        parameter.requires_grad = False

    backbones = Backbones(
        clip_model=clip_model,
        vision_encoder=clip_model.vision_model,
        text_encoder=clip_model.text_model,
        image_processor=image_processor,
        clip_tokenizer=clip_tokenizer,
        qwen_model=qwen_model,
        qwen_tokenizer=qwen_tokenizer,
        clip_dim=int(clip_model.visual_projection.out_features),
        qwen_dim=int(qwen_model.model.embed_tokens.embedding_dim),
        device=device,
    )
    logger.info(
        "Loaded %s (%d-d) and %s (%d-d) on %s",
        clip_model_name,
        backbones.clip_dim,
        qwen_model_name,
        backbones.qwen_dim,
        device,
    )
    return backbones


def load_backbones(config: Config, device: torch.device | None = None) -> Backbones:
    """Load CLIP and Qwen3, frozen and in eval mode.

    Cached, because both are expensive and neither is ever trained here: the UI, the
    API and the training loop all share one copy per process.

    Args:
        config: Supplies the two model names.
        device: Where to put them. Defaults to the best available, which is what every
            inference caller wants. Training passes CPU explicitly; see `training.train`.

    Returns:
        The loaded backbones.
    """
    device = get_device() if device is None else device
    return _load_backbones(config.clip_model_name, config.qwen_model_name, device)


def build_adapter(clip_dim: int, qwen_dim: int, *, final_layer_norm: bool = False) -> nn.Sequential:
    """Build the CLIP-to-Qwen3 projection.

    Kept as a bare `nn.Sequential` rather than a named `nn.Module` subclass on purpose:
    the three committed checkpoints have positional keys (`0.weight`, `1.weight`, ...),
    and wrapping the stack would rename every one of them.

    Args:
        clip_dim: Width of CLIP's projected image embedding.
        qwen_dim: Width of Qwen3's token embeddings.
        final_layer_norm: Append a trailing `LayerNorm`. Needed by
            `adapter_flickr_batched.pth`, which was trained with one.

    Returns:
        The adapter, on CPU and untrained.
    """
    layers: list[nn.Module] = [
        nn.Linear(clip_dim, qwen_dim),
        nn.LayerNorm(qwen_dim),
        nn.GELU(),
        nn.Linear(qwen_dim, qwen_dim),
    ]
    if final_layer_norm:
        layers.append(nn.LayerNorm(qwen_dim))
    return nn.Sequential(*layers)


# Where each loaded adapter's weights came from, keyed by file name. Reported by the
# API's health check so a container serving without its checkpoints is visible.
CHECKPOINT_SOURCES: dict[str, str] = {}


def load_adapter(config: Config, checkpoint_name: str, backbones: Backbones) -> nn.Sequential:
    """Load a committed adapter checkpoint into the architecture it was trained with.

    The architecture comes from the checkpoint's own `extra` metadata, falling back to
    the registry in `config.CHECKPOINTS` for the file names this repo ships. That is the
    fix for the mismatch documented in the module docstring: the shape is read from the
    artifact, not assumed by the caller.

    Args:
        config: Supplies `checkpoint_dir`.
        checkpoint_name: File name under `checkpoint_dir`.
        backbones: Supplies the two dimensions and the device to place the adapter on.

    Returns:
        The adapter in eval mode with the checkpoint's weights loaded.

    Raises:
        FileNotFoundError: If the checkpoint is neither on disk nor on the Hub.
    """
    # This repo's adapters are small enough to host on the Hub, so they are cheap to
    # pull when absent -- but the build context excludes checkpoints/, so a container
    # without a mount has none, and the Hub fallback is what makes that case work.
    checkpoint_path, adapter_source = resolve_checkpoint(
        Path(config.checkpoint_dir) / checkpoint_name, config.hf_repo_id
    )
    if checkpoint_path is None:
        raise FileNotFoundError(
            f"Checkpoint {checkpoint_name} not found in {config.checkpoint_dir} or on the Hub"
        )
    CHECKPOINT_SOURCES[checkpoint_name] = adapter_source

    payload = load_checkpoint(checkpoint_path)
    known = CHECKPOINTS.get(checkpoint_name)
    final_layer_norm = bool(
        payload["extra"].get(
            "final_layer_norm",
            known.final_layer_norm if known is not None else config.final_layer_norm,
        )
    )

    adapter = build_adapter(
        backbones.clip_dim, backbones.qwen_dim, final_layer_norm=final_layer_norm
    )
    adapter.load_state_dict(payload["model_state_dict"])
    adapter.to(backbones.device)
    adapter.eval()
    return adapter
