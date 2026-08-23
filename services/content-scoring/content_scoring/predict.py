"""Prediction: load the saved models and score new titles.

The prediction path is deliberately identical to how training built its inputs:
average a title's word vectors (skipping words outside the vocabulary), then run
that single vector through the scorer. A title with no known words yields a zero
vector. The CBOW vectors come from the word-embeddings repo's checkpoint; the
scorer is this repo's own.

Run ``python -m content_scoring.predict "your title here"`` from the command line;
``serving/api.py`` wraps the same functions in an HTTP API.
"""

import argparse
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from ai_residency_common import get_logger, load_checkpoint, resolve_required

from config import Config
from content_scoring.embeddings import load_cbow_embeddings, load_cbow_vocabulary
from content_scoring.model import ScorePredictor

__all__ = ["ModelState", "create_title_embedding", "load_models", "main", "predict_score"]

logger = get_logger(__name__)


@dataclass
class ModelState:
    """Everything loaded once and reused across predictions.

    Held in one object rather than a set of module-level globals so the loaded and
    not-yet-loaded states are distinguishable without a separate flag.
    """

    predictor: ScorePredictor | None = None
    embeddings: torch.Tensor | None = None
    word_to_index: dict[str, int] = field(default_factory=dict)
    embedding_dim: int = 0
    cbow_source: str = "none"
    predictor_source: str = "none"

    @property
    def is_loaded(self) -> bool:
        """Whether the state is ready to score titles."""
        return self.predictor is not None and self.embeddings is not None


def create_title_embedding(
    title: str,
    word_to_index: dict[str, int],
    embeddings: torch.Tensor,
    embedding_dim: int,
) -> torch.Tensor:
    """Average a title's known word vectors into a single embedding.

    Words outside the vocabulary are skipped, matching how training built title
    embeddings. A title with no known words yields a zero vector.
    """
    word_embeddings: list[np.ndarray[Any, Any]] = []
    for word in title.lower().split():
        index = word_to_index.get(word)
        if index is None:
            logger.debug("Word %r not in vocabulary", word)
        elif index < embeddings.shape[0]:
            word_embeddings.append(embeddings[index].cpu().numpy())
        else:
            logger.debug("Word %r index %d exceeds the embedding matrix", word, index)

    if not word_embeddings:
        logger.warning("No known words in %r; using a zero vector", title)
        return torch.zeros(embedding_dim, dtype=torch.float32)

    title_embedding = np.mean(word_embeddings, axis=0)
    return torch.tensor(title_embedding, dtype=torch.float32)


def load_models(config: Config | None = None) -> ModelState:
    """Populate a `ModelState` from the checkpoints, pulling from the Hub if absent.

    Both artifacts are gitignored (the CBOW weights alone run to ~800MB), so on any
    host that did not train them there is nothing on disk and they are pulled from
    the Hub instead. The CBOW checkpoint comes from the word-embeddings repo; the
    scorer checkpoint is this repo's own.
    """
    config = config or Config()
    state = ModelState()

    cbow_path, state.cbow_source = resolve_required(config.cbow_checkpoint, config.cbow_repo_id)
    state.word_to_index = load_cbow_vocabulary(cbow_path)
    state.embeddings = load_cbow_embeddings(cbow_path)
    state.embedding_dim = int(state.embeddings.shape[1])

    predictor_path, state.predictor_source = resolve_required(
        config.checkpoint_path, config.hf_repo_id
    )
    payload = load_checkpoint(predictor_path)
    input_dim = int(payload.get("extra", {}).get("input_dim", state.embedding_dim))
    predictor = ScorePredictor(input_dim=input_dim)
    predictor.load_state_dict(payload["model_state_dict"])
    predictor.eval()
    state.predictor = predictor

    logger.info("Loaded models; vocabulary holds %d words", len(state.word_to_index))
    return state


def predict_score(state: ModelState, title: str) -> float:
    """Score one title with the loaded models."""
    predictor = state.predictor
    embeddings = state.embeddings
    if predictor is None or embeddings is None:
        raise RuntimeError("Models not loaded")
    title_vector = create_title_embedding(
        title, state.word_to_index, embeddings, state.embedding_dim
    )
    with torch.no_grad():
        score = float(predictor(title_vector).item())
    logger.info("Predicted %.2f for %r", score, title)
    return score


def main() -> None:
    """Entry point: `python -m content_scoring.predict "title"`."""
    parser = argparse.ArgumentParser(description="Predict a content score from a title")
    parser.add_argument("title", help="The post title to score")
    args = parser.parse_args()

    state = load_models()
    score = predict_score(state, args.title)
    print(f"Predicted score for {args.title!r}: {score:.2f}")


if __name__ == "__main__":
    main()
