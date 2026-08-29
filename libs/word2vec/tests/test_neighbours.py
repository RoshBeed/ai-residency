"""Nearest-neighbour lookup, including the vocabulary holes that used to 500."""

import pytest
import torch

from word2vec import nearest_words, valid_row_mask


@pytest.fixture
def contiguous() -> tuple[torch.Tensor, dict[int, str]]:
    """Four rows, four words, no holes."""
    embeddings = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [-1.0, 0.0]],
        dtype=torch.float32,
    )
    return embeddings, {0: "king", 1: "queen", 2: "apple", 3: "opposite"}


def test_returns_the_closest_words_in_order(
    contiguous: tuple[torch.Tensor, dict[int, str]],
) -> None:
    embeddings, index_to_word = contiguous
    result = nearest_words(embeddings, embeddings[0], index_to_word, k=2)
    assert [word for word, _ in result] == ["king", "queen"]
    assert result[0][1] == pytest.approx(1.0)


def test_exclude_drops_the_query_and_still_returns_k(
    contiguous: tuple[torch.Tensor, dict[int, str]],
) -> None:
    embeddings, index_to_word = contiguous
    result = nearest_words(embeddings, embeddings[0], index_to_word, k=2, exclude="king")
    assert [word for word, _ in result] == ["queen", "apple"], (
        "excluding the query must not cost a result slot"
    )


def test_a_row_with_no_word_is_never_returned() -> None:
    """The production failure: more rows than words, and the hole ranks first.

    `/nearest` on the deployed service raised `KeyError: 247768` because the
    checkpoint's matrix had 258,196 rows and only 230,701 of them mapped to a word.
    Row 1 here stands in for that: it is the single best match for the query, so a
    lookup that trusts `topk` reaches for a word that is not there.
    """
    embeddings = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    index_to_word = {0: "king", 2: "apple"}  # no word at row 1

    result = nearest_words(embeddings, torch.tensor([1.0, 0.0]), index_to_word, k=2)

    assert [word for word, _ in result] == ["king", "apple"]


def test_k_larger_than_the_vocabulary_returns_what_exists() -> None:
    """topk padding is -inf at exactly the rows that have no word."""
    embeddings = torch.eye(4, dtype=torch.float32)
    index_to_word = {0: "one", 3: "four"}

    result = nearest_words(embeddings, embeddings[0], index_to_word, k=10)

    assert len(result) == 2
    assert all(word in {"one", "four"} for word, _ in result)


def test_an_empty_vocabulary_returns_nothing_rather_than_raising() -> None:
    assert nearest_words(torch.eye(3), torch.ones(3), {}, k=3) == []


def test_the_mask_ignores_indices_outside_the_matrix() -> None:
    """A vocabulary can name an index the matrix does not have; that must not crash."""
    mask = valid_row_mask(2, {0: "a", 5: "beyond", -1: "negative"})
    assert mask.tolist() == [True, False]


def test_a_precomputed_mask_matches_building_it_each_time() -> None:
    embeddings = torch.randn(6, 3)
    index_to_word = {0: "a", 2: "b", 5: "c"}
    mask = valid_row_mask(6, index_to_word)

    with_mask = nearest_words(embeddings, embeddings[0], index_to_word, k=3, mask=mask)
    without = nearest_words(embeddings, embeddings[0], index_to_word, k=3)

    assert with_mask == without


def test_exclude_accepts_several_words_for_an_analogy() -> None:
    """`a - b + c` lands near c, so an analogy must exclude all three query words.

    Excluding only the query -- which is all a single `exclude` string allowed --
    left `king - man + woman` answering `king`, and any analogy benchmark scored
    that way reports a near-constant zero regardless of the embeddings.
    """
    embeddings = torch.eye(4)
    index_to_word = {0: "a", 1: "b", 2: "c", 3: "d"}

    found = nearest_words(embeddings, embeddings[2], index_to_word, k=2, exclude=("a", "b", "c"))

    assert [word for word, _ in found] == ["d"]


def test_exclude_still_accepts_a_bare_string() -> None:
    embeddings = torch.eye(3)
    index_to_word = {0: "a", 1: "b", 2: "c"}

    found = nearest_words(embeddings, embeddings[0], index_to_word, k=1, exclude="a")

    assert [word for word, _ in found] != ["a"]
