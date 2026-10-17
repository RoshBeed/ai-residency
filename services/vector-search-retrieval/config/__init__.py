"""Run configuration.

Everything that used to be a literal buried in a script lives here: the CBOW
hyperparameters that were inline in `01_train_w2v.py`, the tower hyperparameters that
only existed inside a `wandb.config` object, the Redis connection details that were
duplicated across four files, and every file path.

The paths matter more than usual in this repo. Before this pass, the artifact paths
were written when the CBOW stage still lived at `src/cbow/` and wrote its outputs next
to itself. A later restructuring moved the vocabulary into `data/` and the embedding
checkpoints into `checkpoints/cbow/`, but the seven modules that read them were never
updated, so every consumer of a trained artifact — training, indexing, the CLI, the
API and the UI — failed with `FileNotFoundError`. Deriving all of them from
`data_dir` and `checkpoint_dir` here is what stops that class of drift recurring.

The sweep in `config/sweep.yaml` varies `learning_rate`, `hidden_size`, `margin` and
`num_epochs`, all of which are fields below, so a sweep and a plain `make train` now
configure the same object.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

from ai_residency_common import BaseConfig

__all__ = ["Config"]

# Redis connection details come from the environment (or a local `.env`), which is how
# the compose services and the portfolio hub point this repo at their own Redis.
load_dotenv()


@dataclass
class Config(BaseConfig):
    """Configuration for every stage of the retrieval pipeline.

    Attributes:
        learning_rate: Adam learning rate for the GRU towers.
        num_epochs: Passes over the training triples.
        batch_size: Triples per optimizer step. Under `in_batch` this is also how
            many negatives each query is scored against, so it is a property of the
            objective rather than only a throughput knob.
        loss: `in_batch` scores every query against every other document in the batch;
            `triplet` scores it against the one sampled negative in its own triple.
        temperature: Softmax temperature for the `in_batch` loss. Cosine similarities
            live in [-1, 1], so without dividing by something small the logits are too
            flat for cross-entropy to separate them.
        margin: Triplet-loss margin the positive must beat the negative by. Unused
            under `in_batch`.
        recall_at_k: K for the recall@K the evaluator reports.
        document_pool_size: Documents a held-out query is ranked against. Recall@K is
            meaningless without it: against 10 documents it is nearly free, against
            5,000 it is not.
        early_stopping_patience: Epochs without a new best before training stops.
        embedding_dim: Width of the CBOW word embeddings the towers sit on.
        hidden_size: Width of each tower's GRU hidden state, and therefore of the
            vectors stored in the Redis index.
        min_word_count: Words rarer than this are dropped from the corpus entirely.
        context_size: Words of context on each side of the CBOW target word.
        cbow_learning_rate: Adam learning rate for the CBOW stage.
        cbow_batch_size: Examples per batch for the CBOW stage.
        cbow_num_epochs: Passes over the corpus for the CBOW stage.
        cbow_eval_word: Word whose nearest neighbours are printed during CBOW
            training, as a cheap sanity check that the embeddings are going somewhere.
        cbow_eval_every: Batches between those nearest-neighbour printouts.
        text8_url: Where the raw corpus is downloaded from.
        regressor_learning_rate: Adam learning rate for the legacy regressor demo.
        regressor_steps: Optimizer steps for the legacy regressor demo.
        redis_host: Redis hostname.
        redis_port: Redis port.
        redis_password: Redis password; empty for the local throwaway container.
        index_name: RediSearch index the document vectors live in.
        top_k: Default number of results a query returns.
        api_host: Interface the FastAPI service binds to.
        api_port: Port the FastAPI service binds to.
    """

    # Tower training. `learning_rate`, `num_epochs`, `margin` and `hidden_size` are
    # what `config/sweep.yaml` sweeps over.
    learning_rate: float = 0.001
    num_epochs: int = 10
    batch_size: int = 64
    loss: Literal["in_batch", "triplet"] = "in_batch"
    temperature: float = 0.05
    margin: float = 0.2
    embedding_dim: int = 128
    hidden_size: int = 128
    early_stopping_patience: int = 3

    # Evaluation. 5,000 documents is the pool the published comparison used, which is
    # what makes its 0.0020 chance recall@10 the right floor to read these against.
    recall_at_k: int = 10
    document_pool_size: int = 5000

    # CBOW word embeddings.
    min_word_count: int = 5
    context_size: int = 2
    cbow_learning_rate: float = 0.003
    cbow_batch_size: int = 256
    cbow_num_epochs: int = 5
    cbow_eval_word: str = "computer"
    cbow_eval_every: int = 10_000
    text8_url: str = (
        "https://huggingface.co/datasets/roshbeed/ai-residency-text8/resolve/main/text8"
    )

    # Legacy single-example regressor demo, kept because the README documents it.
    regressor_learning_rate: float = 0.005
    regressor_steps: int = 100

    # Redis / RediSearch.
    redis_host: str = field(default_factory=lambda: os.environ.get("REDIS_HOST", "localhost"))
    redis_port: int = field(default_factory=lambda: int(os.environ.get("REDIS_PORT", "6390")))
    redis_password: str = field(default_factory=lambda: os.environ.get("REDIS_PASSWORD", ""))
    index_name: str = "doc_index"
    top_k: int = 5

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # MS MARCO's canonical home. The bare name `ms_marco` worked for as long as the Hub
    # resolved unqualified dataset names; current huggingface_hub rejects it outright
    # ("Repository id must be 'namespace/name'"), which broke every stage that reads the
    # corpus. Named once here rather than in each of the three data-prep modules.
    source_dataset_repo_id: str = "microsoft/ms_marco"
    source_dataset_config: str = "v1.1"

    hf_repo_id: str | None = "roshbeed/ai-residency-vector-search-retrieval"
    # The tokenizer vocabulary, the CBOW corpus and the tokenized triples. All are
    # derived here rather than downloaded, and none are committed, so without a home
    # off this machine they existed only on the laptop that built them — which is why
    # the portfolio hub had to bind-mount data/ from a sibling repo.
    hf_dataset_repo_id: str | None = "roshbeed/ai-residency-vector-search-retrieval-data"
    wandb_project: str | None = "ai-residency-vector-search-retrieval"

    # --- Data artifacts -----------------------------------------------------

    @property
    def text8_path(self) -> Path:
        """Raw `text8` corpus, as downloaded."""
        return self.data_dir / "text8"

    @property
    def corpus_path(self) -> Path:
        """Preprocessed corpus: a flat list of words."""
        return self.data_dir / "corpus.pkl"

    @property
    def words_to_ids_path(self) -> Path:
        """Vocabulary mapping word to token id."""
        return self.data_dir / "tkn_words_to_ids.pkl"

    @property
    def ids_to_words_path(self) -> Path:
        """Vocabulary mapping token id back to word."""
        return self.data_dir / "tkn_ids_to_words.pkl"

    @property
    def triples_path(self) -> Path:
        """MS MARCO triples as text."""
        return self.data_dir / "triples_small.json"

    @property
    def tokenized_triples_path(self) -> Path:
        """MS MARCO triples with token ids alongside the text."""
        return self.data_dir / "tokenized_triples.json"

    # --- Trained artifacts --------------------------------------------------

    @property
    def cbow_checkpoint_dir(self) -> Path:
        """Directory the CBOW stage writes one checkpoint per epoch into."""
        return self.checkpoint_dir / "cbow"

    @property
    def query_tower_path(self) -> Path:
        """Trained query-side GRU tower."""
        return self.checkpoint_dir / "qry_tower.pth"

    @property
    def document_tower_path(self) -> Path:
        """Trained document-side GRU tower."""
        return self.checkpoint_dir / "doc_tower.pth"
