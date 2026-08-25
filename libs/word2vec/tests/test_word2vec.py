"""Tests for the shared word2vec implementation.

The interesting ones are not "does it run" but "does each variant behave the way the
algorithm says it should" -- softmax starting at ln(vocab_size), negative sampling
pushing targets up and negatives down, the 3/4 sampling power sitting between uniform
and raw frequency, and a checkpoint from either older implementation still loading.
"""

import math

import pytest
import torch

from word2vec import (
    NegativeSampler,
    Word2Vec,
    Word2VecConfig,
    state_dict_from_legacy,
    unigram_distribution,
)

VARIANTS = [
    (arch, obj) for arch in ("cbow", "skipgram") for obj in ("negative_sampling", "softmax")
]


def build(**overrides: object) -> Word2Vec:
    """A small model, with sensible defaults for a test."""
    fields: dict[str, object] = {"vocab_size": 50, "embed_size": 8}
    fields.update(overrides)
    return Word2Vec(Word2VecConfig(**fields))  # type: ignore[arg-type]


def batch(size: int = 4, context: int = 4, vocab: int = 50) -> tuple[torch.Tensor, torch.Tensor]:
    """One batch of (centre, context) ids."""
    return torch.randint(0, vocab, (size,)), torch.randint(0, vocab, (size, context))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        {"vocab_size": 0},
        {"vocab_size": 10, "embed_size": 0},
        {"vocab_size": 10, "architecture": "cbwo"},
        {"vocab_size": 10, "objective": "hierarchical_softmax"},
        {"vocab_size": 10, "num_negatives": 0},
        {"vocab_size": 10, "window_size": 0},
    ],
)
def test_config_rejects_impossible_combinations(bad: dict[str, object]) -> None:
    """Caught at construction, not at the first batch."""
    with pytest.raises(ValueError):
        Word2VecConfig(**bad)  # type: ignore[arg-type]


def test_config_accepts_zero_negatives_under_softmax() -> None:
    """num_negatives is meaningless for softmax, so it must not be validated."""
    assert Word2VecConfig(vocab_size=10, objective="softmax", num_negatives=0)


# ---------------------------------------------------------------------------
# The model, across all four variants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("architecture", "objective"), VARIANTS)
def test_every_variant_returns_a_scalar_loss(architecture: str, objective: str) -> None:
    """The point of the abstraction: one training loop, four variants."""
    model = build(architecture=architecture, objective=objective)
    centre, context = batch()
    negatives = torch.randint(0, 50, (4, 5))

    loss = model(centre, context, negatives)

    assert loss.shape == ()
    assert torch.isfinite(loss)


@pytest.mark.parametrize(("architecture", "objective"), VARIANTS)
def test_the_first_step_moves_only_the_output_matrix(architecture: str, objective: str) -> None:
    """A real consequence of zero-initialising out_embed, worth pinning down.

    Gradient reaches in_embed only *through* out_embed, so while out_embed is all
    zeros the input vectors get exactly zero gradient. The first optimiser step
    therefore moves only out_embed. This is what the reference implementation does
    and it is not a bug -- but it does mean "loss decreased after one step" says
    nothing about the vectors you actually keep.
    """
    model = build(architecture=architecture, objective=objective)
    centre, context = batch()

    model(centre, context, torch.randint(0, 50, (4, 5))).backward()

    in_grad, out_grad = model.in_embed.weight.grad, model.out_embed.weight.grad
    assert in_grad is not None and out_grad is not None
    assert torch.all(in_grad == 0)
    assert torch.any(out_grad != 0)


@pytest.mark.parametrize(("architecture", "objective"), VARIANTS)
def test_the_kept_vectors_learn_once_the_output_matrix_is_nonzero(
    architecture: str, objective: str
) -> None:
    """The step after that, in_embed is being trained like anything else."""
    model = build(architecture=architecture, objective=objective)
    centre, context = batch()
    negatives = torch.randint(0, 50, (4, 5))

    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    model(centre, context, negatives).backward()
    optimizer.step()
    optimizer.zero_grad()
    model(centre, context, negatives).backward()

    in_grad = model.in_embed.weight.grad
    assert in_grad is not None
    assert torch.any(in_grad != 0)


@pytest.mark.parametrize("architecture", ["cbow", "skipgram"])
def test_softmax_starts_at_ln_vocab_size(architecture: str) -> None:
    """out_embed is initialised to zeros, so every logit is 0 and the distribution
    is uniform. Cross-entropy on a uniform distribution over V classes is ln(V) --
    a specific number the implementation has to hit, not a vague "it decreases"."""
    model = build(vocab_size=100, architecture=architecture, objective="softmax")
    centre, context = batch(vocab=100)

    loss = model(centre, context)

    assert loss.item() == pytest.approx(math.log(100), abs=1e-5)


def test_negative_sampling_separates_targets_from_negatives() -> None:
    """After training on one pair, the target should score above a negative.

    This is the whole objective in one assertion: push the true target up, push the
    sampled words down.
    """
    torch.manual_seed(0)
    model = build(vocab_size=20, embed_size=8, objective="negative_sampling")
    centre = torch.tensor([1])
    context = torch.tensor([[2, 3, 4, 5]])
    negatives = torch.tensor([[10, 11, 12, 13, 14]])

    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    for _ in range(50):
        optimizer.zero_grad()
        model(centre, context, negatives).backward()
        optimizer.step()

    pooled = model.in_embed(context).mean(dim=1)
    target_score = pooled @ model.out_embed.weight[1]
    negative_score = pooled @ model.out_embed.weight[10]
    assert target_score.item() > negative_score.item()


def test_negative_sampling_without_negatives_is_a_clear_error() -> None:
    model = build(objective="negative_sampling")
    centre, context = batch()

    with pytest.raises(ValueError, match="needs `negatives`"):
        model(centre, context)


def test_softmax_ignores_negatives_it_is_given() -> None:
    """Passing negatives to a softmax model is harmless, not a silent behaviour change."""
    torch.manual_seed(0)
    model = build(objective="softmax")
    centre, context = batch()

    with_negatives = model(centre, context, torch.randint(0, 50, (4, 5)))
    without = model(centre, context)

    assert with_negatives.item() == pytest.approx(without.item())


def test_cbow_and_skipgram_read_different_inputs() -> None:
    """CBOW predicts *from* the context; skip-gram predicts *from* the centre word.

    Both must therefore respond to a changed centre word, but for opposite reasons:
    for CBOW it is the thing being predicted, for skip-gram it is the input vector.
    out_embed is randomised first, because at its zero initialisation every softmax
    loss is exactly ln(vocab_size) regardless of input.
    """
    torch.manual_seed(0)
    context = torch.tensor([[2, 3, 4, 5]])

    cbow = build(objective="softmax", architecture="cbow")
    torch.nn.init.normal_(cbow.out_embed.weight)
    skipgram = build(objective="softmax", architecture="skipgram")
    # Same weights, so only the architecture differs.
    skipgram.load_state_dict(cbow.state_dict())

    cbow_a = cbow(torch.tensor([1]), context)
    cbow_b = cbow(torch.tensor([7]), context)
    skip_a = skipgram(torch.tensor([1]), context)
    skip_b = skipgram(torch.tensor([7]), context)

    assert cbow_a.item() != pytest.approx(cbow_b.item())
    assert skip_a.item() != pytest.approx(skip_b.item())
    # And the two architectures do not merely agree by accident.
    assert cbow_a.item() != pytest.approx(skip_a.item())


def test_embeddings_property_is_the_input_matrix() -> None:
    model = build()
    assert model.embeddings is model.in_embed.weight
    assert model.embeddings.shape == (50, 8)


# ---------------------------------------------------------------------------
# Negative sampling distribution
# ---------------------------------------------------------------------------


def test_unigram_distribution_sums_to_one() -> None:
    probs = unigram_distribution(torch.tensor([10.0, 5.0, 1.0]))
    assert probs.sum().item() == pytest.approx(1.0)


def test_the_three_quarter_power_sits_between_uniform_and_raw_frequency() -> None:
    """The reason the exponent exists. A word 100x more common than another should
    end up more likely than under uniform sampling, but far less than 100x."""
    counts = torch.tensor([100.0, 1.0])

    raw = unigram_distribution(counts, power=1.0)
    damped = unigram_distribution(counts, power=0.75)
    uniform = unigram_distribution(counts, power=0.0)

    ratio_raw = (raw[0] / raw[1]).item()
    ratio_damped = (damped[0] / damped[1]).item()
    ratio_uniform = (uniform[0] / uniform[1]).item()

    assert ratio_uniform == pytest.approx(1.0)
    assert ratio_raw == pytest.approx(100.0)
    assert ratio_uniform < ratio_damped < ratio_raw


def test_zero_count_words_are_never_sampled() -> None:
    """Padding and unused ids must not turn up as negatives."""
    sampler = NegativeSampler(torch.tensor([0.0, 5.0, 5.0]))
    drawn = sampler(64, 4)
    assert 0 not in set(drawn.flatten().tolist())


def test_sampler_shape_and_range() -> None:
    sampler = NegativeSampler(torch.ones(30))
    drawn = sampler(8, 5)
    assert drawn.shape == (8, 5)
    assert int(drawn.min()) >= 0
    assert int(drawn.max()) < 30


def test_sampler_is_reproducible_with_a_generator() -> None:
    sampler = NegativeSampler(torch.ones(30))
    first = sampler(4, 3, generator=torch.Generator().manual_seed(7))
    second = sampler(4, 3, generator=torch.Generator().manual_seed(7))
    assert torch.equal(first, second)


@pytest.mark.parametrize("counts", [torch.zeros(5), torch.tensor([-1.0, 2.0])])
def test_unusable_counts_are_rejected(counts: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        unigram_distribution(counts)


# ---------------------------------------------------------------------------
# Loading the checkpoints that already exist
# ---------------------------------------------------------------------------


def test_word_embeddings_layout_loads_unchanged() -> None:
    """That implementation already used these names, so the map is the identity."""
    model = build(vocab_size=12, embed_size=4)
    legacy = {
        "in_embed.weight": torch.randn(12, 4),
        "out_embed.weight": torch.randn(12, 4),
    }

    model.load_state_dict(state_dict_from_legacy(legacy, "word-embeddings"))

    assert torch.equal(model.embeddings, legacy["in_embed.weight"])


def test_retrieval_layout_maps_the_softmax_projection_onto_out_embed() -> None:
    """`ffw` is Linear(embed, vocab, bias=False), so its weight is (vocab, embed) --
    the same shape and the same role as an output embedding matrix. That is what
    makes this a rename rather than a retrain."""
    model = build(vocab_size=12, embed_size=4)
    legacy = {"emb.weight": torch.randn(12, 4), "ffw.weight": torch.randn(12, 4)}

    model.load_state_dict(state_dict_from_legacy(legacy, "vector-search-retrieval"))

    assert torch.equal(model.embeddings, legacy["emb.weight"])
    assert torch.equal(model.out_embed.weight, legacy["ffw.weight"])


def test_a_wrong_layout_guess_is_an_error_not_a_silent_partial_load() -> None:
    legacy = {"emb.weight": torch.randn(12, 4), "ffw.weight": torch.randn(12, 4)}

    with pytest.raises(ValueError, match="does not look like"):
        state_dict_from_legacy(legacy, "word-embeddings")


def test_unexpected_keys_are_rejected_under_strict() -> None:
    legacy = {
        "emb.weight": torch.randn(12, 4),
        "ffw.weight": torch.randn(12, 4),
        "something.else": torch.randn(3),
    }

    with pytest.raises(ValueError, match="unexpected keys"):
        state_dict_from_legacy(legacy, "vector-search-retrieval")

    dropped = state_dict_from_legacy(legacy, "vector-search-retrieval", strict=False)
    assert set(dropped) == {"in_embed.weight", "out_embed.weight"}


def test_an_unknown_layout_names_the_ones_that_exist() -> None:
    with pytest.raises(KeyError, match="word-embeddings"):
        state_dict_from_legacy({}, "some-other-team")
