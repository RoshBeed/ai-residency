"""Run configuration.

The word vectors come from the sibling `word-embeddings` service: skip-gram, 71,290
words of text8, pulled from that repo's Hugging Face repo rather than retrained here.
This service extends that vocabulary onto the Hacker News title corpus (`vocab.py`) and
trains a score-prediction head on top (`train.py`).

Paths are relative to the repo root, which is where every `make` target runs from.

Every field below is overridable as `AI_RESIDENCY_<FIELD>`, which is what makes the
sweep in `sweep.yaml` need no plumbing in the trainer at all. `fusion` is a `Literal`
and is swept like any other field -- `BaseConfig._coerce` validates it against the
allowed set.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from ai_residency_common import BaseConfig

__all__ = ["Config", "Fusion", "Objective", "Selection"]

Objective = Literal["poisson", "log_mse"]
"""What the head is trained to predict.

* `poisson` -- Poisson negative log-likelihood with a log link. The model emits
  `log(lambda)` and `exp` of it *is* the expected upvote count, so there is no
  back-transform to be biased. This is what a count target asks for.
* `log_mse` -- squared error on `log1p(score)`. Ranks well and is what the first version
  of this service did, but `expm1` of a conditional mean-of-logs is close to a geometric
  mean, which on a distribution with a median of 3 and a maximum of 5,710 lands about
  five times low. Kept because that gap is a result worth showing rather than hiding.
"""

Selection = Literal["mae", "spearman"]
"""Which validation metric picks the best epoch, and what a sweep optimises.

`mae` for a service that predicts counts; `spearman` for one that ranks. Selecting on
rank while training a count objective is how the shipped hyperparameters came to be tuned
for a question the service had stopped asking.
"""

Fusion = Literal["text_only", "features_only", "early", "late"]

# How the isolated layer's width is divided between the input groups. See
# `Config.block_allocation` for why proportional is not the neutral choice it looks like.
Allocation = Literal["proportional", "equal"]

# Whether the groups are resized before they meet. See `Config.input_projection`.
Projection = Literal["none", "text_down", "features_up", "all"]
"""How the title and the metadata features reach the regression head.

* `text_only` -- the pooled title vector alone. An ablation: it says whether the title
  carries any signal at all.
* `features_only` -- the metadata alone, no text. The other ablation, and on Hacker
  News a genuinely strong model.
* `early` -- one MLP over `concat(title, every feature)`. The network is free to find
  interactions between the text and the metadata, and has to learn the grouping itself.
* `late` -- the same layer with the cross-group weights removed, so no input sees another
  until the head. Same input, hidden width, depth and head as `early`, and strictly
  smaller because it is `early` with weights deleted -- which is what makes the gap
  between them the value of the crosstalk and nothing else.
"""


@dataclass
class Config(BaseConfig):
    """Configuration for the Hacker News score predictor.

    Attributes:
        fusion: Which architecture to build. See `Fusion`.
        objective: What the head predicts. See `Objective`.
        selection_metric: Which validation metric picks the best epoch. See `Selection`.
        hidden_widths: Width of each hidden layer, in order. The number of layers and
            their sizes are one field rather than two, so a tapering stack is a value
            rather than a code change: `AI_RESIDENCY_HIDDEN_WIDTHS=320,256,192` is a
            three-layer taper. `BaseConfig._coerce` parses the comma-separated form, so
            this is sweepable with no trainer plumbing.
        tower_shapes: An explicit pre-fusion tower per input group, as
            `text:128;temporal:16,16;author:8,8,8`. Empty leaves the towers derived from
            `hidden_widths` and `fusion_layer`, which forces every group to the same
            depth. Setting it lets each input have the *shape* it wants -- narrow and
            deep, or wide and shallow -- which is the one structural thing late fusion
            can do that early fusion cannot. `hidden_widths` is then the shared trunk.

            Note this deliberately gives up the property the fusion-point sweep rests on:
            towers of different depths no longer tile a single dense layer, so the model
            is not early fusion with weights deleted any more. That is the right trade for
            a shape experiment and the wrong one for a connectivity experiment, so the two
            sets of numbers must not be pooled.
        block_allocation: How the isolated layer's width is split between the input
            groups. `proportional` gives each group a share of the hidden units in
            proportion to how many columns it brings, which sounds fair and is not: the
            title is a 300-d *mean of word vectors*, so its width is an artifact of the
            embedding size rather than a measure of information, and it takes 233 of 256
            units while the author's history takes 2. Measured on this data the 29
            metadata columns out-predict the 300 title columns, so the rule hands the
            weaker input eight times the capacity. `equal` gives every group the same
            share. Either way the blocks sum to exactly the layer's width, so a
            block-diagonal layer stays the same size as the dense one it is carved from.
        input_projection: Resize the groups before they reach the fusion layer, so that
            "300 versus 3" stops being the thing under test. `text_down` projects the
            title to `projection_dim`; `features_up` projects each metadata group up to
            it; `all` puts every group at it. `none` leaves the raw widths.
        projection_dim: Target width for `input_projection`.
        fusion_layer: The layer at which the input groups first meet. Layers before it
            are block-diagonal -- each group reaches only its own hidden units -- and
            layers from it on are dense and shared. `1` is early fusion; `2` lets each
            group have one private layer first; `len(hidden_widths)` is the latest
            fusion that still leaves a shared layer. Values past the end clamp, so a
            sweep drawing `5` against a two-layer stack costs a run rather than wasting
            one. Where the modalities meet is the experiment, so it is a number rather
            than the difference between two classes.
        dropout: Dropout between hidden layers.
        domain_embed_dim: Width of the learned domain embedding.
        freeze_embeddings: Keep the word vectors fixed. When false they are fine-tuned
            with the head, which is swept but not published -- see `vocab.py`.
        num_epochs: Maximum epochs; early stopping usually ends it first.
        patience: Epochs without improvement before stopping.
        weight_decay: Adam weight decay.
        eval_every: Steps between `train/*` log points.
        log_tables: Log the per-epoch slice and probe tables to W&B. On for a run
            someone will read; off for a grid, where each run would upload two table
            artifacts per epoch and 490 runs would queue ~20,000 uploads. That backlog
            is not theoretical -- it wedged a run for 25 minutes after its metrics were
            already computed, and starved the W&B service into failing an earlier one
            outright. The scalars the curve is drawn from are logged either way.
        save_checkpoints: Sweep runs set this false. Two dozen runs writing a
            checkpoint each is disk nobody reads.
        promote: Whether a winning seed may replace the served checkpoint.
        max_title_tokens: Titles are truncated to this many tokens. The 99.9th
            percentile is well under it; the bound exists so a pathological title
            cannot size a batch.
        embedding_repo_id: Hub repo holding the word-embeddings checkpoint.
        embedding_checkpoint: Where that checkpoint lives (or is downloaded to).
        hn_repo_id: Hub dataset holding the raw Hacker News mirror.
        raw_posts_file: Where that mirror lives (or is downloaded to).
        upstream_repo_id: Where the mirror was copied from, once, by
            `etl.py --source=upstream`. Recorded so the provenance is in the repo.
        data_cache_dir: Derived splits and the title corpus. Local only, never pushed.
        train_end: Posts strictly before this date are training data.
        val_end: Posts before this and on/after `train_end` are validation.
        min_domain_count: Domains rarer than this in the training split collapse to
            `<rare>`.
        vocab_min_count: Hacker News words rarer than this are not added to the
            vocabulary.
        extend_epochs: Epochs of continued word2vec training on the title corpus.
        extend_learning_rate: Learning rate for that stage.
        extend_window_size: Context width for the extension, capped against the
            pretrained run's own window.
        extend_batch_size: Pairs per step while extending.
        freeze_pretrained: Hold the text8 rows fixed while extending.
        embeddings_name: File name for the extended embedding matrix.
        info_name: File name for the JSON sidecar describing a checkpoint.
    """

    fusion: Fusion = "early"
    objective: Objective = "poisson"
    selection_metric: Selection = "mae"
    hidden_widths: tuple[int, ...] = (256, 256)
    fusion_layer: int = 2
    tower_shapes: str = ""
    block_allocation: Allocation = "proportional"
    input_projection: Projection = "none"
    projection_dim: int = 64
    dropout: float = 0.0
    domain_embed_dim: int = 8
    freeze_embeddings: bool = True

    num_epochs: int = 20
    patience: int = 5
    weight_decay: float = 1.835e-4
    batch_size: int = 1024
    learning_rate: float = 3.737e-4
    eval_every: int = 100
    log_tables: bool = True
    save_checkpoints: bool = True
    # False for the objective A/B runs: they exist to be compared, not served,
    # and must not overwrite the checkpoint the API is loading.
    promote: bool = True
    max_title_tokens: int = 32

    embedding_repo_id: str | None = "roshbeed/ai-residency-word-embeddings"
    embedding_checkpoint: Path = field(
        default_factory=lambda: Path("checkpoints/skipgram/skipgram_model.pt")
    )

    # The raw mirror is the only published data artifact, and it is written once and
    # never updated -- which is what makes a date-based split reproducible without
    # publishing the splits themselves. Public, so nothing here needs a token to read.
    hn_repo_id: str | None = "roshbeed/ai-residency-hn-posts"
    raw_posts_file: Path = field(default_factory=lambda: Path("data/hn_stories.parquet"))
    upstream_repo_id: str = "julien040/hacker-news-posts"
    data_cache_dir: Path = field(default_factory=lambda: Path("data"))

    # Time-based, not random. Score norms drift and posts from the same day correlate,
    # so a random split lets the model see the future of its own test set.
    train_end: str = "2026-03-01"
    val_end: str = "2026-06-01"

    min_domain_count: int = 50
    vocab_min_count: int = 5
    extend_epochs: int = 5
    extend_learning_rate: float = 1e-3
    # Wider than the pretrained run's window buys nothing on ten-token titles and
    # doubles the pair count; the extension takes the smaller of the two.
    extend_window_size: int = 5
    # Larger than the predictor's batch. This stage is embedding gathers and a small
    # bmm, which is launch-bound on MPS rather than compute-bound, so wider batches are
    # most of the speed.
    extend_batch_size: int = 8192
    freeze_pretrained: bool = False

    embeddings_name: str = "hn_embeddings.pt"
    info_name: str = "predictor_info.json"

    hf_repo_id: str | None = "roshbeed/ai-residency-content-scoring"
    wandb_project: str | None = "ai-residency-content-scoring"

    @property
    def predictor_checkpoint(self) -> Path:
        """Where this run's checkpoint is written.

        Named after the architecture, the objective **and the seed**. Without the seed,
        three seeds overwrite one path, `make evaluate` scores whichever ran last, and a
        table that says "three seeds" reports one -- which is exactly what happened here,
        and it hid a seed spread large enough to erase the gap between two architectures.

        The objective is in the name because the two are different models: comparing them
        requires both sets on disk at once, and sharing a filename made that impossible.
        """
        return self.checkpoint_dir / self.fusion / f"{self.fusion}_{self.objective}_s{self.seed}.pt"

    @property
    def served_checkpoint(self) -> Path:
        """The seed promoted for serving: whichever scored best on validation.

        Separate from `predictor_checkpoint` so the thing that ships is a deliberate
        choice rather than an accident of loop order, and so the Hub carries one file per
        architecture instead of one per run.
        """
        return self.checkpoint_dir / self.fusion / f"{self.fusion}_predictor.pt"

    @property
    def embeddings_path(self) -> Path:
        """The extended embedding matrix, shared by all four architectures."""
        return self.checkpoint_dir / self.embeddings_name

    @property
    def info_path(self) -> Path:
        """Full path to this run's JSON sidecar."""
        return self.predictor_checkpoint.with_suffix(".json")

    @property
    def served_info_path(self) -> Path:
        """Sidecar for the promoted checkpoint, which records which seed won."""
        return self.served_checkpoint.with_suffix(".json")

    def seed_checkpoints(self) -> list[Path]:
        """Every seed trained for this architecture *under this objective*.

        Scoped to the objective so an aggregate never silently averages two different
        models -- the mistake this whole path exists to prevent.
        """
        directory = self.checkpoint_dir / self.fusion
        return sorted(directory.glob(f"{self.fusion}_{self.objective}_s*.pt"))

    def split_path(self, split: str) -> Path:
        """Local path of one derived split.

        Args:
            split: `train`, `val` or `test`.

        Returns:
            The parquet path under `data_cache_dir`. These are a cache, rebuilt from
            the frozen mirror by `make etl`, and are never uploaded.
        """
        return self.data_cache_dir / f"hn_posts_{split}.parquet"

    @property
    def titles_path(self) -> Path:
        """The tokenised training titles, the corpus `vocab.py` extends on."""
        return self.data_cache_dir / "hn_titles.txt"
