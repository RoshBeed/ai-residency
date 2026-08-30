"""Run configuration.

One `Config` for the word-vector stage. The hyperparameters used to live in a
separate `cbow_ns.yml` with SHOUTY_CASE keys, read at runtime out of `wandb.config`,
which meant two config systems in one repo and a training stage that could not start
unless a tracking run had been opened first. Everything lives here now, so
`AI_RESIDENCY_<FIELD_NAME>` overrides reach every stage alike.

The fields lost their `cbow_` prefix when the service started training skip-gram as
well, and the ones `BaseConfig` already declares -- `batch_size`, `learning_rate`,
`num_epochs`, `seed` -- are now inherited rather than shadowed under a second name.
That also retires `AI_RESIDENCY_CBOW_NUM_EPOCHS`, a per-service spelling of an
override that is supposed to be uniform.

Paths are relative to the repo root, which is where every `make` target runs from.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from ai_residency_common import BaseConfig

__all__ = ["Architecture", "Config"]

Architecture = Literal["cbow", "skipgram"]


@dataclass
class Config(BaseConfig):
    """Configuration for the word-vector stage.

    Attributes:
        architecture: `cbow` averages the context to predict the centre word;
            `skipgram` predicts each context word from the centre. Skip-gram is the
            default because it won the sweep -- see the README's Experimental
            Results -- and it is therefore what the API serves. Both are trained and
            published; `AI_RESIDENCY_ARCHITECTURE=cbow` switches without a rebuild.
        embed_size: Width of the learned word vectors.
        window_size: Context words taken either side of the target.
        num_negatives: Negative samples drawn per positive pair.
        min_count: Words occurring fewer times than this are dropped from the
            vocabulary and from the corpus.
        subsample_threshold: The `t` in Mikolov's frequent-word discard rule.
        max_analogy_vocab: Analogy answers are searched over this many of the most
            frequent words, the usual restriction for the Google set.
        eval_every: Optimizer steps between training-metric logs.
        probe_words: Words whose nearest neighbours are logged each epoch.
        save_checkpoints: Whether to write a checkpoint at all. False during sweeps,
            where two dozen runs would write several GB nobody reads.
        text8_file: Local path for the text8 corpus.
        text8_repo_id: Hub dataset the corpus is mirrored at.
        eval_dir: Local directory for the intrinsic-evaluation benchmark files.
        eval_repo_id: Hub dataset the benchmark files are mirrored at.
    """

    architecture: Architecture = "skipgram"
    embed_size: int = 300
    window_size: int = 5
    num_negatives: int = 5
    min_count: int = 5
    subsample_threshold: float = 1e-4
    max_analogy_vocab: int = 30_000
    eval_every: int = 500
    save_checkpoints: bool = True

    # Frequent enough in text8 to be well trained, and spread across senses that a
    # human can judge at a glance -- which is the whole point of a probe.
    probe_words: tuple[str, ...] = ("king", "france", "computer", "three", "river", "war")

    batch_size: int = 4096
    learning_rate: float = 1e-3
    num_epochs: int = 10

    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    text8_file: Path = field(default_factory=lambda: Path("data/text8"))
    text8_repo_id: str | None = "roshbeed/ai-residency-text8"
    eval_dir: Path = field(default_factory=lambda: Path("data/eval"))
    eval_repo_id: str | None = "roshbeed/ai-residency-word-embeddings-eval"

    hf_repo_id: str | None = "roshbeed/ai-residency-word-embeddings"
    wandb_project: str | None = "ai-residency-word-embeddings"

    @property
    def embedding_checkpoint(self) -> Path:
        """Where this run's checkpoint is written.

        Named after the architecture, so the two do not overwrite each other and the
        CBOW run keeps the `checkpoints/cbow/cbow_model.pt` path that
        `content-scoring` resolves against and the Hub already serves.
        """
        return self.checkpoint_dir / self.architecture / f"{self.architecture}_model.pt"

    @property
    def embedding_checkpoint_dir(self) -> Path:
        """Directory this run writes its artifacts into."""
        return self.embedding_checkpoint.parent
