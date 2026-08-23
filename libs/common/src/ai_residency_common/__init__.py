"""Shared helpers for the AI Residency portfolio.

Six standalone repos (one per residency week) depend on this package so that device
selection, logging, checkpoint format, config fields, seeding, artifact hosting and
experiment tracking are identical across all of them instead of reimplemented six
times.

Importing this package loads a `.env` file if one is nearby, so `HF_TOKEN` and
`WANDB_API_KEY` reach `push_checkpoint` and `init_tracking` without being exported by
hand. See `env.py` for how the file is located and why real environment wins.
"""

from ai_residency_common.checkpoints import load_checkpoint, save_checkpoint
from ai_residency_common.config import BaseConfig, set_seed
from ai_residency_common.device import get_device
from ai_residency_common.env import load_env
from ai_residency_common.hub import (
    ResolvedCheckpoint,
    get_hf_token,
    pull_checkpoint,
    push_checkpoint,
    resolve_checkpoint,
)
from ai_residency_common.logging import get_logger
from ai_residency_common.serving import (
    decode_audio,
    decode_image,
    load_model,
    resolve_required,
    run_app,
)
from ai_residency_common.sse import format_sse, stream_sse
from ai_residency_common.tracking import finish_tracking, init_tracking, log_metrics

__version__ = "0.7.2"

# Before anything reads os.environ: Config field overrides are applied at
# construction time and the tokens are read when training starts pushing or logging.
load_env()

__all__ = [
    "BaseConfig",
    "ResolvedCheckpoint",
    "decode_audio",
    "decode_image",
    "finish_tracking",
    "format_sse",
    "get_device",
    "get_hf_token",
    "get_logger",
    "init_tracking",
    "load_checkpoint",
    "load_env",
    "load_model",
    "log_metrics",
    "pull_checkpoint",
    "push_checkpoint",
    "resolve_checkpoint",
    "resolve_required",
    "run_app",
    "save_checkpoint",
    "set_seed",
    "stream_sse",
]
