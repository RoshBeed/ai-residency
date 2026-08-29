"""Nearest-neighbour lookup over an embedding matrix.

Three services had their own copy of this -- the serving API's `/nearest`, the
training-time evaluation probe, and the retrieval service's qualitative check --
and all three carried the same bug, which is the usual argument for one
implementation rather than three.

The bug is that an embedding matrix can have more rows than the vocabulary has
words. `topk` searches every row, so it can return a row index that maps to no
word, and the lookup that follows raises `KeyError` rather than answering. See
`nearest` for why the holes exist and what happens to them.
"""

from collections.abc import Iterable

import torch
import torch.nn.functional as F

__all__ = ["nearest_words", "valid_row_mask"]


def valid_row_mask(rows: int, index_to_word: dict[int, str]) -> torch.Tensor:
    """A `(rows,)` bool mask, True where the row has a word in `index_to_word`.

    Args:
        rows: Number of rows in the embedding matrix.
        index_to_word: Vocabulary mapping row index to word.

    Returns:
        A boolean tensor selecting the rows that can be named.
    """
    mask = torch.zeros(rows, dtype=torch.bool)
    indices = [index for index in index_to_word if 0 <= index < rows]
    if indices:
        mask[torch.tensor(indices, dtype=torch.long)] = True
    return mask


def nearest_words(
    embeddings: torch.Tensor,
    vector: torch.Tensor,
    index_to_word: dict[int, str],
    *,
    k: int = 5,
    exclude: str | Iterable[str] | None = None,
    mask: torch.Tensor | None = None,
) -> list[tuple[str, float]]:
    """Return the `k` words whose vectors are closest to `vector` by cosine similarity.

    Rows with no word are excluded from the search rather than allowed to win it.
    A vocabulary is not always a contiguous `0..N-1` block: it can be built with
    reserved ids, filtered after the fact, or map two spellings onto one row, and the
    matrix is then sized to `max(index) + 1`. Every unused row in between holds
    whatever it was initialised with, which is a real vector and can out-rank real
    words -- so the naive version does not merely risk a `KeyError`, it also returns
    junk on the occasions it does not raise.

    Args:
        embeddings: `(rows, dim)` embedding matrix.
        vector: `(dim,)` query vector.
        index_to_word: Vocabulary mapping row index to word.
        k: How many neighbours to return.
        exclude: A word, or words, to drop from the results. For a plain neighbour
            lookup that is the query itself, which is always its own nearest
            neighbour. For an analogy `a - b + c` it is all three of a, b and c: the
            query vector sits a short hop from c, so without excluding them the
            answer is almost always one of the words that were asked about. Scoring
            an analogy set without this reports a near-constant zero.
        mask: Precomputed `valid_row_mask`. Pass it on a hot path to avoid rebuilding
            the mask per request; it is derived from `index_to_word` either way.

    Returns:
        `(word, cosine_similarity)` pairs, most similar first. Shorter than `k` if
        the vocabulary has fewer than `k` usable rows.
    """
    if mask is None:
        mask = valid_row_mask(embeddings.shape[0], index_to_word)

    excluded = {exclude} if isinstance(exclude, str) else set(exclude or ())

    similarities = F.cosine_similarity(embeddings, vector.unsqueeze(0)).cpu()
    similarities = similarities.masked_fill(~mask, float("-inf"))

    # Ask for one extra so dropping `exclude` still leaves k, and never ask for more
    # than there are namable rows -- topk on a larger k returns the -inf padding,
    # whose indices are exactly the rows that have no word.
    wanted = min(k + len(excluded), int(mask.sum()))
    if wanted <= 0:
        return []

    top = similarities.topk(wanted)
    pairs = [
        (index_to_word[int(index)], float(similarity))
        for index, similarity in zip(top.indices, top.values, strict=True)
    ]
    if excluded:
        pairs = [pair for pair in pairs if pair[0] not in excluded]
    return pairs[:k]
