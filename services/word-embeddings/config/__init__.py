"""Run configuration.

One `Config` for the CBOW embedding stage. The hyperparameters used to live in a
separate `cbow_ns.yml` with SHOUTY_CASE keys, read at runtime out of `wandb.config`,
which meant two config systems in one repo and a training stage that could not start
unless a tracking run had been opened first. Everything lives here now, so
`AI_RESIDENCY_<FIELD_NAME>` overrides reach every stage alike.

Paths are relative to the repo root, which is where every `make` target runs from.
"""

from dataclasses import dataclass, field
from pathlib import Path

from ai_residency_common import BaseConfig

__all__ = ["Config"]


@dataclass
class Config(BaseConfig):
    """Configuration for the CBOW embedding stage.

    Attributes:
        cbow_embed_size: Width of the learned word vectors.
        cbow_window_size: Context words taken either side of the target.
        cbow_num_negatives: Negative samples drawn per positive pair.
        cbow_batch_size: Examples per CBOW batch.
        cbow_learning_rate: Adam learning rate for the CBOW stage.
        cbow_num_epochs: Passes over the corpus.
        cbow_eval_every: Batches between analogy/similarity evaluations.
        cbow_checkpoint: Where the trained CBOW checkpoint is written.
        corpus_file: Merged corpus text the CBOW stage trains on.
        word_index_file: Word-to-index mapping produced by the ETL stage.
        vocab_file: Word-to-lemma-index mapping produced by the ETL stage.
        posts_file: Cleaned Hacker News posts, whose titles feed the merged corpus.
    """

    cbow_embed_size: int = 128
    cbow_window_size: int = 4
    cbow_num_negatives: int = 15
    cbow_batch_size: int = 4096
    cbow_learning_rate: float = 1e-3
    cbow_num_epochs: int = 1
    cbow_eval_every: int = 100

    cbow_checkpoint: Path = field(default_factory=lambda: Path("checkpoints/cbow/cbow_model.pt"))
    corpus_file: Path = field(default_factory=lambda: Path("data/combined_data.txt"))
    word_index_file: Path = field(default_factory=lambda: Path("data/combined_word_to_index.json"))
    vocab_file: Path = field(default_factory=lambda: Path("data/combined_word_to_lemma_index.json"))
    posts_file: Path = field(default_factory=lambda: Path("data/hn_data_cleaned.json"))

    hf_repo_id: str | None = "roshbeed/ai-residency-week1-word-embeddings"
    # The Hacker News extract lives here rather than in git. It came from a Postgres
    # mirror that is not reachable from every network, so it is the one input the
    # corpus stage cannot simply fetch or regenerate — which used to make committing
    # it the only option.
    hf_dataset_repo_id: str | None = "roshbeed/ai-residency-week1-hn-posts"
    wandb_project: str | None = "ai-residency-week1-word-embeddings"

    @property
    def cbow_checkpoint_dir(self) -> Path:
        """Directory the CBOW stage writes its artifacts into."""
        return self.cbow_checkpoint.parent
