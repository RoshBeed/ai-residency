"""Vocabulary and CBOW checkpoint loading.

Five modules need the same three things: the word-to-id map, the newest CBOW
checkpoint, and a frozen `nn.Embedding` built from it. Each of them used to open its
own hardcoded path, which is how they all ended up pointing at a `src/cbow/` directory
that no longer holds anything. They share these functions now.
"""

import hashlib
import pickle
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ai_residency_common import get_logger, load_checkpoint, resolve_required
from config import Config
from word2vec import state_dict_from_legacy

__all__ = [
    "CANONICAL_CBOW_CHECKPOINT",
    "build_embedding_layer",
    "build_embedding_placeholder",
    "check_tokenized_against_vocabulary",
    "latest_cbow_checkpoint",
    "load_ids_to_words",
    "load_words_to_ids",
    "vocabulary_fingerprint",
]

logger = get_logger(__name__)

# Documents re-tokenized when a file carries no fingerprint. The real mismatch
# disagreed on every row, so a sample settles it without reading a 325MB file twice.
_RETOKENIZE_SAMPLE = 200

# The CBOW trainer names each local checkpoint after its run timestamp and epoch, so
# there is no filename a downstream host could ask the Hub for. Uploads use this fixed
# name instead, and the Hub fallback below asks for it. It still ends in `.cbow.pth` so
# a pulled copy is picked up by the same local glob as a freshly-trained one.
CANONICAL_CBOW_CHECKPOINT = "cbow_model.cbow.pth"


def vocabulary_fingerprint(words_to_ids: dict[str, int]) -> str:
    """A short, stable identifier for one word-to-id mapping.

    Token ids only mean anything against the vocabulary that produced them, and nothing
    about a list of integers says which that was. This is what lets a consumer check
    rather than assume. It hashes the mapping itself, not just its size: two vocabularies
    of 63,642 words that order them differently are not interchangeable, and the size
    alone would call them equal.

    Args:
        words_to_ids: The mapping to fingerprint.

    Returns:
        The first 16 hex characters of a SHA-256 over the id-sorted mapping.
    """
    digest = hashlib.sha256()
    for word, index in sorted(words_to_ids.items(), key=lambda pair: pair[1]):
        digest.update(f"{index}\t{word}\n".encode())
    return digest.hexdigest()[:16]


def check_tokenized_against_vocabulary(
    payload: dict[str, Any], words_to_ids: dict[str, int], source: str
) -> None:
    """Raise if a tokenized artifact was built against a different vocabulary.

    Token ids are integers and nothing about them says which mapping produced them, so
    a stale file is consumed in silence. That happened: triples tokenized against a
    7,692-word vocabulary were read against a 63,642-word one, every embedding lookup
    returned a different word's vector, and the only thing that visibly broke was
    serving -- training and evaluation shared the same wrong mapping, so the loss fell
    and recall@10 reproduced.

    Args:
        payload: The loaded artifact, which may carry `vocabulary_fingerprint`.
        words_to_ids: The vocabulary it is about to be used with.
        source: Path or name of the artifact, for the message.

    Raises:
        ValueError: If the recorded fingerprint does not match, or if the ids cannot
            belong to this vocabulary at all.
    """
    expected = vocabulary_fingerprint(words_to_ids)
    recorded = payload.get("vocabulary_fingerprint")

    if recorded is not None and recorded != expected:
        raise ValueError(
            f"{source} was tokenized against vocabulary {recorded}, but this run uses "
            f"{expected} ({len(words_to_ids):,} words). Rebuild it with "
            "`make train-triples`; its ids index a different word for every token."
        )

    # A file from before fingerprints existed still has to be checked, and a bound on
    # the ids is not enough: the stale triples' ids all ran below 7,692 and so fitted
    # inside the 63,642-word vocabulary perfectly while naming different words. What
    # does catch it is that the file keeps the original text beside the ids, so the ids
    # can simply be recomputed and compared. Exact, and a sample is plenty -- the real
    # case disagreed on every row.
    if recorded is None:
        from data_prep.tokenize_triples import tokenize_text

        disagreed = checked = 0
        for rows in payload.values():
            if not isinstance(rows, list):
                continue
            for row in rows[:_RETOKENIZE_SAMPLE]:
                text, tokens = row.get("positive_document"), row.get("positive_document_tokens")
                if text is None or tokens is None:
                    continue
                checked += 1
                disagreed += tokenize_text(text, words_to_ids) != tokens
            if checked >= _RETOKENIZE_SAMPLE:
                break

        if checked and disagreed:
            raise ValueError(
                f"{source} disagrees with this vocabulary on {disagreed} of {checked} sampled "
                f"documents: re-tokenizing their text gives different ids than the file stores, "
                f"so it was built against a different {len(words_to_ids):,}-word mapping. "
                "Rebuild it with `make train-triples`."
            )
        logger.info(
            "%s records no vocabulary fingerprint; re-tokenized %d documents and they agree",
            source,
            checked,
        )


def load_words_to_ids(config: Config) -> dict[str, int]:
    """Load the word-to-token-id vocabulary written by `train_tokenizer`.

    Args:
        config: Supplies `words_to_ids_path`.

    Returns:
        Mapping from word to token id.

    Raises:
        FileNotFoundError: If the tokenizer stage has not been run.
    """
    path, _ = resolve_required(
        config.words_to_ids_path,
        config.hf_dataset_repo_id,
        repo_type="dataset",
        rebuild_hint="Build the vocabulary with `make train-cbow`.",
    )
    with path.open("rb") as handle:
        words_to_ids: dict[str, int] = pickle.load(handle)
    return words_to_ids


def load_ids_to_words(config: Config) -> dict[int, str]:
    """Load the token-id-to-word vocabulary written by `train_tokenizer`.

    Args:
        config: Supplies `ids_to_words_path`.

    Returns:
        Mapping from token id to word.

    Raises:
        FileNotFoundError: If the tokenizer stage has not been run.
    """
    path, _ = resolve_required(
        config.ids_to_words_path,
        config.hf_dataset_repo_id,
        repo_type="dataset",
        rebuild_hint="Build the vocabulary with `make train-cbow`.",
    )
    with path.open("rb") as handle:
        ids_to_words: dict[int, str] = pickle.load(handle)
    return ids_to_words


def latest_cbow_checkpoint(config: Config) -> Path:
    """Return the most recently written CBOW checkpoint, pulling it if absent.

    Every other loader in this module resolves through `resolve_required`, so it
    falls back to the Hub when a host has no local copy. This one used to glob the
    local directory and stop there, which meant a fresh host (a container with no
    checkpoint mount, the cloud-serving case) could never start no matter what was
    published -- it failed with "train it first" while the artifact sat on the Hub.

    A glob has no single filename to ask the Hub for, so the fallback asks for the
    canonical name the training stage writes and uploads.

    Args:
        config: Supplies `cbow_checkpoint_dir` and `hf_repo_id`.

    Returns:
        Path to the newest local `*.cbow.pth` file, or the pulled canonical one.

    Raises:
        FileNotFoundError: If the checkpoint is neither on disk nor on the Hub.
    """
    candidates = sorted(config.cbow_checkpoint_dir.glob("*.cbow.pth"))
    if candidates:
        return max(candidates, key=lambda path: path.stat().st_mtime)

    path, source = resolve_required(
        config.cbow_checkpoint_dir / CANONICAL_CBOW_CHECKPOINT,
        config.hf_repo_id,
        rebuild_hint="Train word embeddings first with `make train-cbow`.",
    )
    logger.info("Loaded CBOW checkpoint from %s", source)
    return path


def build_embedding_placeholder(config: Config) -> nn.Embedding:
    """Build an unpopulated embedding layer of the right shape, for inference.

    A trained tower checkpoint carries its own `embedding.weight` -- the embeddings
    are frozen while the tower trains, so the copy inside the tower *is* the CBOW
    matrix. Loading the tower therefore overwrites whatever this layer holds, and
    building it from the CBOW checkpoint first only reads a file to throw its
    contents away.

    That made every inference path depend on a training-stage artifact it does not
    use, which is a hard failure on a host that has the published towers but not the
    embeddings they were trained on. Only the shape matters here, and the vocabulary
    already gives that.

    Args:
        config: Supplies the vocabulary path and `embedding_dim`.

    Returns:
        An `nn.Embedding` of the correct shape, to be populated by a checkpoint.
    """
    embedding_layer = nn.Embedding(len(load_words_to_ids(config)), config.embedding_dim)
    embedding_layer.weight.requires_grad = False
    return embedding_layer


def build_embedding_layer(config: Config) -> nn.Embedding:
    """Build a frozen embedding layer from the newest CBOW checkpoint.

    Training only. Inference should use `build_embedding_placeholder`, since the
    tower checkpoint it is about to load already contains these same weights.

    The towers train on top of these weights without updating them, so the layer is
    returned with `requires_grad` already off. Sharing one instance between the query
    and document towers is intentional: both sides must read the same word vectors for
    their cosine similarities to mean anything.

    Args:
        config: Supplies the vocabulary and checkpoint paths and `embedding_dim`.

    Returns:
        An `nn.Embedding` holding the trained CBOW word vectors.
    """
    words_to_ids = load_words_to_ids(config)
    checkpoint_path = latest_cbow_checkpoint(config)
    payload = load_checkpoint(checkpoint_path)
    state_dict = payload["model_state_dict"]

    # `in_embed.weight` is what libs/word2vec writes. `emb.weight` is what this
    # service's own CBOW wrote before the two implementations were merged -- the same
    # matrix under the old name, so a checkpoint from either loads. The legacy branch
    # goes away once the embeddings have been retrained.
    if "in_embed.weight" in state_dict:
        weights = state_dict["in_embed.weight"]
    else:
        weights = state_dict_from_legacy(state_dict, "vector-search-retrieval", strict=False)[
            "in_embed.weight"
        ]

    embedding_layer = nn.Embedding(len(words_to_ids), config.embedding_dim)
    with torch.no_grad():
        embedding_layer.weight.copy_(weights)
    embedding_layer.weight.requires_grad = False
    logger.info("Loaded CBOW embeddings from %s", checkpoint_path)
    return embedding_layer
