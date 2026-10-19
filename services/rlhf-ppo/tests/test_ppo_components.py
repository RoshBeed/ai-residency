"""Tests for the PPO components.

The fixtures that need real model weights are session-scoped: every one of the four
model roles is a separate LoRA-wrapped copy of SmolLM2-135M, so rebuilding them per
test would load 135M parameters four times over for each of a dozen tests.
"""

import math
from collections.abc import Iterator
from itertools import pairwise
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from config import PPOConfig
from rlhf_ppo.advantage import GAEAdvantage, ReturnCalculator, TDError
from rlhf_ppo.base_setup import BaseModelSetup
from rlhf_ppo.buffer import ExperienceBuffer
from rlhf_ppo.models import PolicyModel, RewardModel, SFTModel, ValueModel
from rlhf_ppo.ppo import PPOTrainer, SequenceGenerator
from rlhf_ppo.reward_kl import KLDivergence, RewardAdjuster
from rlhf_ppo.token_attribution import TokenAttribution
from rlhf_ppo.training import PolicyLoss, PolicyRatio, ValueLoss
from rlhf_ppo.types import SAPair

DEVICE = torch.device("cpu")


@pytest.fixture(scope="session")
def config() -> PPOConfig:
    """The default run configuration."""
    return PPOConfig()


@pytest.fixture(scope="session")
def base_setup(config: PPOConfig) -> BaseModelSetup:
    """Tokenizer, LoRA config and base-model factory."""
    return BaseModelSetup(config)


@pytest.fixture
def sa_pairs() -> list[SAPair]:
    """Two (s, a) pairs from one prompt, as generation would produce them."""
    return [
        {
            "state_text": "SUBREDDIT: r/test\nTITLE: Test prompt\nPOST: This is a test.\nTL;DR:",
            # Generation records the ids it scored alongside the decoded text; the
            # PPO ratio reads these, so a fixture standing in for generation carries
            # both. The exact ids do not matter here, only that they travel.
            "state_token_ids": [101, 202, 303],
            "action_token": " Test",
            "action_token_id": 1234,
            "logprob": -2.5,
            "ref_logprob": -2.8,
            "prompt_idx": 0,
            "step": 0,
        },
        {
            "state_text": (
                "SUBREDDIT: r/test\nTITLE: Test prompt\nPOST: This is a test.\nTL;DR: Test"
            ),
            "state_token_ids": [101, 202, 303, 1234],
            "action_token": " response",
            "action_token_id": 5678,
            "logprob": -1.8,
            "ref_logprob": -2.1,
            "prompt_idx": 0,
            "step": 1,
        },
    ]


def test_config_inherits_the_shared_base_fields(config: PPOConfig) -> None:
    """PPOConfig extends ai_residency_common's BaseConfig rather than redeclaring its fields."""
    assert config.model_name
    assert config.batch_size > 0
    assert config.learning_rate > 0
    assert config.value_learning_rate > 0
    # Inherited from BaseConfig.
    assert config.seed == 42
    assert config.checkpoint_dir.name == "checkpoints"
    assert config.device.type in {"cuda", "mps", "cpu"}


def test_base_setup_exposes_tokenizer_and_lora_config(base_setup: BaseModelSetup) -> None:
    """BaseModelSetup wraps the LoRA config one level deeper, on LoRASetup."""
    assert base_setup.tokenizer is not None
    assert base_setup.lora_setup.lora_config is not None


def test_tokenizer_pads_on_the_left(base_setup: BaseModelSetup) -> None:
    """A decoder-only model batched through generate() must be left-padded."""
    assert base_setup.tokenizer.padding_side == "left"


def test_models_build_and_expose_forward(base_setup: BaseModelSetup) -> None:
    """All four PPO roles construct from the same base setup."""
    models = [
        SFTModel(base_setup).to(DEVICE),
        PolicyModel(base_setup).to(DEVICE),
        RewardModel(base_setup).to(DEVICE),
        ValueModel(base_setup).to(DEVICE),
    ]
    for model in models:
        assert callable(model.forward)


def test_kl_divergence(sa_pairs: list[SAPair]) -> None:
    """KL is the per-token difference of the two log-probabilities."""
    for pair in KLDivergence().add_kl_to_sa_pairs(sa_pairs):
        assert pair["kl_div"] == pytest.approx(pair["logprob"] - pair["ref_logprob"])


def test_reward_adjuster(sa_pairs: list[SAPair]) -> None:
    """The sequence reward is penalised per token by that token's KL."""
    rewards = [0.5, -0.3]
    kl_coef = 0.1
    adjusted = RewardAdjuster().adjust_rewards_with_kl(
        KLDivergence().add_kl_to_sa_pairs(sa_pairs), rewards, kl_coef
    )
    for pair in adjusted:
        expected = rewards[pair["prompt_idx"]] - kl_coef * pair["kl_div"]
        assert pair["adjusted_reward"] == pytest.approx(expected)


def test_td_error_uses_zero_for_the_final_state(sa_pairs: list[SAPair]) -> None:
    """The last token of a sequence has no successor, so its next value is 0."""
    for pair in sa_pairs:
        pair["value"] = 0.1
        pair["adjusted_reward"] = 0.5

    gamma = 0.99
    computed = TDError().compute_td_errors(sa_pairs, gamma=gamma)

    assert computed[0]["td_error"] == pytest.approx(0.5 + gamma * 0.1 - 0.1)
    assert computed[1]["td_error"] == pytest.approx(0.5 + gamma * 0.0 - 0.1)


def test_gae_advantage_accumulates_backwards(sa_pairs: list[SAPair]) -> None:
    """Each step's advantage discounts the advantage of the step after it."""
    for pair in sa_pairs:
        pair["td_error"] = 0.2

    gamma, lam = 0.99, 0.95
    computed = GAEAdvantage().compute_gae_advantages(sa_pairs, gamma=gamma, lam=lam)

    assert computed[1]["advantage"] == pytest.approx(0.2)
    assert computed[0]["advantage"] == pytest.approx(0.2 + gamma * lam * 0.2)


def test_return_calculator(sa_pairs: list[SAPair]) -> None:
    """Reward-to-go is recovered as advantage plus value."""
    for pair in sa_pairs:
        pair["advantage"] = 0.3
        pair["value"] = 0.1

    for pair in ReturnCalculator().add_reward_to_go(sa_pairs):
        assert pair["reward_to_go"] == pytest.approx(pair["advantage"] + pair["value"])


def test_experience_buffer(sa_pairs: list[SAPair]) -> None:
    """The buffer keeps only the fields a policy update reads."""
    for pair in sa_pairs:
        pair["advantage"] = 0.3
        pair["reward_to_go"] = 0.4

    buffer = ExperienceBuffer()
    buffer.add_experience(sa_pairs)

    assert len(buffer.get_batch(1)) == 1
    assert buffer.buffer[0]["old_logprob"] == sa_pairs[0]["logprob"]

    buffer.clear()
    assert buffer.buffer == []


def test_policy_loss_can_be_negative() -> None:
    """With mixed-sign advantages the PPO-clip loss has no positivity invariant."""
    loss = PolicyLoss.ppo_clip_loss(
        torch.tensor([-1.0, -2.0]),
        torch.tensor([-1.1, -2.1]),
        torch.tensor([0.5, -0.3]),
        clip_epsilon=0.2,
    )
    assert isinstance(loss, torch.Tensor)
    assert loss.item() == pytest.approx(-0.1105, abs=1e-4)


def test_value_loss() -> None:
    """The value loss is a plain MSE against reward-to-go."""
    loss = ValueLoss.mse_loss(torch.tensor([0.5, 0.3]), torch.tensor([0.6, 0.4]))
    assert loss.item() == pytest.approx(0.01, abs=1e-6)


def test_generated_pairs_carry_the_ids_that_were_scored(
    base_setup: BaseModelSetup,
) -> None:
    """`state_token_ids` must be the tokens themselves, not a re-encoded string.

    This is the invariant behind the PPO ratio. If the ids were reconstructed by
    encoding `state_text`, they would silently be whatever the tokenizer produced on
    the way back, and the ratio would compare two different sequences. So the check
    is that the ids decode to the text and grow by exactly one token per step --
    never that re-encoding the text reproduces the ids, which is the assumption the
    old implementation made and the thing that was not true.
    """
    from rlhf_ppo.ppo.generator import SequenceGenerator

    model = base_setup.create_base_model()
    generator = SequenceGenerator(
        model, model, base_setup.tokenizer, PPOConfig(max_new_tokens=3), torch.device("cpu")
    )

    pairs, _ = generator.generate_sequences(["TL;DR:"])

    assert pairs, "generation produced no pairs"
    for pair in pairs:
        assert isinstance(pair["state_token_ids"], list)
        assert all(isinstance(token_id, int) for token_id in pair["state_token_ids"])
        assert pair["state_token_ids"], "a state must have at least the prompt in it"

    lengths = [len(pair["state_token_ids"]) for pair in pairs]
    assert lengths == sorted(lengths), "the state only ever grows"
    assert lengths[-1] - lengths[0] == len(pairs) - 1, "one token per step, no gaps"


def test_re_encoding_a_state_is_not_the_same_as_carrying_its_ids(
    base_setup: BaseModelSetup,
) -> None:
    """Why the PPO ratio takes ids: decode-then-encode is not the identity.

    `PolicyRatio` used to re-tokenize `state_text`. That is only correct if encoding
    a decoded state returns the ids it came from, and it does not -- for a
    meaningful share of sequences the round trip comes back with different ids, and
    sometimes a different *number* of them. The two sides of the ratio then scored
    different states, which is why the ratio was not 1 before any gradient step.

    The generated tokens that trigger it are low-probability ones, so sampling finds
    the case only occasionally; that is exactly what made the old failure an
    intermittent xfail rather than a visible bug. Searching for it directly makes the
    property deterministic.
    """
    tokenizer = base_setup.tokenizer
    prompt_ids = list(tokenizer("TL;DR:").input_ids)

    generator = torch.Generator().manual_seed(0)
    mismatch: tuple[list[int], list[int]] | None = None
    for _ in range(2000):
        candidate = (
            prompt_ids + torch.randint(0, tokenizer.vocab_size, (3,), generator=generator).tolist()
        )
        re_encoded = list(
            tokenizer(tokenizer.decode(candidate, skip_special_tokens=True)).input_ids
        )
        if re_encoded != candidate:
            mismatch = (candidate, re_encoded)
            break

    assert mismatch is not None, "expected at least one state that does not round-trip"
    original, re_encoded = mismatch
    assert original != re_encoded, (
        "the round trip changed the state -- re-tokenizing `state_text` would have "
        "scored these ids instead of the ones the generator scored"
    )


def test_token_attribution_grows_the_state_one_token_at_a_time(
    base_setup: BaseModelSetup,
) -> None:
    """Step t's state is the prompt plus every token generated before t.

    The number of log-probabilities is taken from the tokenizer rather than assumed:
    `TokenAttribution` re-tokenizes generated *text*, and that round trip does not
    have to produce the same number of tokens that were originally generated. That
    mismatch is exactly why `SequenceGenerator` builds its pairs inline from the
    generated ids instead of going through this class.
    """
    attribution = TokenAttribution(base_setup.tokenizer)
    generated_text = " a b"
    num_tokens = len(base_setup.tokenizer.tokenize(generated_text))

    pairs = attribution.extract_sa_pairs(
        prompts=["TL;DR:"],
        generated_texts=[generated_text],
        policy_logprobs=[[-1.0] * num_tokens],
        ref_logprobs=[[-1.1] * num_tokens],
    )

    assert len(pairs) == num_tokens
    assert [pair["step"] for pair in pairs] == list(range(len(pairs)))
    assert pairs[0]["state_text"] == "TL;DR:"
    for earlier, later in pairwise(pairs):
        assert later["state_text"].startswith(earlier["state_text"])


@pytest.fixture(scope="module")
def trainer() -> Iterator[PPOTrainer]:
    """A PPO trainer on the default configuration.

    Four LoRA-wrapped copies of a 135M model, so it is built once for the module and
    shared. The tests below run in declaration order and the last one takes a real
    gradient step, which is why it comes last.
    """
    yield PPOTrainer(PPOConfig())


class TestTrainedPipeline:
    """Tests that need the full four-model trainer."""

    def test_trainer_wires_up_every_stage(self, trainer: PPOTrainer) -> None:
        assert trainer.generator is not None
        assert trainer.reward_computer is not None
        assert trainer.value_adder is not None
        assert trainer.model_trainer is not None

    def test_generated_logprobs_are_finite_under_aggressive_top_p(
        self, trainer: PPOTrainer
    ) -> None:
        """Regression test for the top-p log-probability bug.

        `SequenceGenerator` used to read each sampled token's log-probability out of
        `generate()`'s `scores`, which are the distribution *after* temperature, top-k
        and top-p filtering. Top-p's mask and the categorical draw disagreed at the
        nucleus boundary, so a token that had genuinely just been sampled could carry
        a probability of exactly zero in those scores, giving `-inf` when logged. That
        `-inf` propagated through the KL term, GAE and both losses.

        A very small `top_p` makes the nucleus a single token, which is the regime
        where that disagreement showed up. The invariant asserted here is the one the
        fix guarantees: log-probabilities come from an unwarped forward pass, so they
        are finite for every sampled token no matter how aggressive the warping was.
        """
        config = PPOConfig(top_p=0.01, top_k=1, max_new_tokens=4)
        generator = SequenceGenerator(
            trainer.policy_model,
            trainer.sft_model,
            trainer.base_setup.tokenizer,
            config,
            trainer.device,
        )

        pairs, generated_texts = generator.generate_sequences(["TL;DR:", "In summary, the"])

        assert len(generated_texts) == 2
        assert len(pairs) == 2 * config.max_new_tokens
        for pair in pairs:
            assert math.isfinite(pair["logprob"]), pair
            assert math.isfinite(pair["ref_logprob"]), pair

    def test_ppo_ratio_is_one_before_any_update(self, trainer: PPOTrainer) -> None:
        """Regression test for the missing attention mask during generation.

        `SequenceGenerator` records `logprob`, and `PolicyRatio.get_new_logprobs`
        recomputes the same quantity during the update. Before any gradient step the
        policy has not changed, so `exp(new - old)` must be exactly 1.

        It was not. The generator scored the state's raw token ids with no attention
        mask, so for every prompt except the longest in the batch it attended to the
        left padding, while `get_new_logprobs` re-tokenizes the decoded state text,
        which has none. On this repo's own example prompts the ratio reached 2.38 —
        outside PPO's own clip range — so the clipping was firing on a padding
        artefact rather than on real policy movement.

        The two prompts here are deliberately different lengths, which is what makes
        the shorter one padded.

        This used to be an xfail. Generation samples, so which tokens it produces
        varies between runs, and `PolicyRatio` re-tokenized the *decoded* state --
        which meant a low-probability token could produce a state string that did not
        round-trip back to the same ids. The two sides of the ratio then scored
        different token sequences and the ratio was not 1. Carrying `state_token_ids`
        through removes the round trip rather than working around it, so this test
        needs no seeding: the ratio is 1 whatever gets sampled, which is the property
        actually worth pinning.
        """
        generator = SequenceGenerator(
            trainer.policy_model,
            trainer.sft_model,
            trainer.base_setup.tokenizer,
            PPOConfig(max_new_tokens=3),
            trainer.device,
        )
        pairs, _ = generator.generate_sequences(
            ["TL;DR:", "SUBREDDIT: r/test\nTITLE: a much longer prompt than the other\nTL;DR:"]
        )

        with torch.no_grad():
            new_logprobs = PolicyRatio.get_new_logprobs(
                trainer.policy_model,
                trainer.base_setup.tokenizer,
                [list(pair["state_token_ids"]) for pair in pairs],
                [pair["action_token_id"] for pair in pairs],
                trainer.device,
                trainer.config.batch_size,
            ).cpu()

        old_logprobs = torch.tensor([pair["logprob"] for pair in pairs])
        ratio = torch.exp(new_logprobs - old_logprobs)
        assert torch.allclose(ratio, torch.ones_like(ratio), atol=1e-3), ratio

    def test_generated_logprob_matches_an_unwarped_forward_pass(self, trainer: PPOTrainer) -> None:
        """The recorded log-probability is the model's real one, not a warped one.

        Finiteness alone would still pass if the log-probability were taken from a
        merely less aggressive warping, so this pins down which distribution it comes
        from: an independent, unwarped forward pass over the same state.
        """
        config = PPOConfig(top_p=0.5, top_k=5, max_new_tokens=2)
        generator = SequenceGenerator(
            trainer.policy_model,
            trainer.sft_model,
            trainer.base_setup.tokenizer,
            config,
            trainer.device,
        )
        tokenizer = trainer.base_setup.tokenizer

        pairs, _ = generator.generate_sequences(["TL;DR:"])

        for pair in pairs:
            state_ids = tokenizer(
                pair["state_text"], return_tensors="pt", add_special_tokens=False
            )["input_ids"].to(trainer.device)
            with torch.no_grad():
                logits = trainer.policy_model(state_ids).logits[0, -1, :]
            expected = F.log_softmax(logits, dim=-1)[pair["action_token_id"]].item()
            assert pair["logprob"] == pytest.approx(expected, abs=1e-3)

    def test_ppo_step_produces_finite_losses(self, trainer: PPOTrainer) -> None:
        """A whole step runs end to end and neither loss is inf or nan."""
        results = trainer.ppo_step(["SUBREDDIT: r/test\nTITLE: t\nPOST: p\nTL;DR:"])

        assert math.isfinite(results["value_loss"])
        assert math.isfinite(results["policy_loss"])
        assert len(results["generated_texts"]) == 1
        assert len(results["rewards"]) == 1
        for pair in results["sa_pairs"]:
            for key in ("kl_div", "adjusted_reward", "value", "td_error", "advantage"):
                assert math.isfinite(pair[key]), (key, pair)


def test_empty_prompts_are_422_not_400() -> None:
    """One status code for one class of failure, across every service in the repo.

    A well-formed request whose content cannot be processed is 422. `rlhf-ppo` was
    the last place still answering 400 for that, after the same alignment in the
    image and audio services.
    """
    from fastapi.testclient import TestClient

    from serving.api import app

    with TestClient(app) as client:
        assert client.post("/generate", json={"prompt": "   "}).status_code == 422
        assert client.post("/ppo_step", json={"prompts": ["  "]}).status_code == 422


def test_a_saved_adapter_is_actually_loaded_back(
    tmp_path: Path, base_setup: BaseModelSetup
) -> None:
    """A checkpoint round trip has to move the weights, not just log that it did.

    `save_checkpoint` wraps the tensors in a record alongside `optimizer_state_dict`,
    `epoch`, `metrics` and `extra`. `_load_ckpt` passed that whole record to
    `load_state_dict(strict=False)`, which matched none of the model's parameters,
    loaded nothing, and reported "5 unexpected" -- those five wrapper keys -- under a
    log line beginning "Loaded". Every run therefore started from a fresh adapter no
    matter how many checkpoints had been written, and the only symptom was a resumed
    policy behaving like an untrained one.
    """
    from ai_residency_common import load_checkpoint, save_checkpoint

    policy_model = PolicyModel(base_setup).to(DEVICE)
    path = tmp_path / "policy_lora.pth"

    # LoRA's B matrices start at zero, so writing into one makes "did the load apply?"
    # a question with an unambiguous answer.
    state = policy_model.state_dict()
    b_keys = [key for key in state if "lora_B" in key]
    assert b_keys, "the policy is expected to be LoRA-adapted"
    marked = {key: torch.full_like(state[key], 0.5) for key in b_keys}
    save_checkpoint(path, policy_model, state_dict={**state, **marked})

    record = load_checkpoint(path, map_location="cpu")
    assert set(record) >= {"model_state_dict", "epoch", "metrics"}, (
        "the saved record wraps the tensors; this test exists because that was missed"
    )

    # Handing the record straight to load_state_dict is the bug: nothing matches.
    fresh = policy_model.state_dict()
    unmatched = sum(1 for key in record if key in fresh)
    assert unmatched == 0, "the wrapper keys must not look like parameter names"

    # Unwrapping it is the fix.
    tensors = record["model_state_dict"]
    applied = sum(1 for key in tensors if key in fresh)
    assert applied == len(fresh), "every parameter in the file should match the model"

    _, unexpected = policy_model.load_state_dict(tensors, strict=False)
    assert not unexpected
    reloaded = policy_model.state_dict()
    assert all(
        torch.allclose(reloaded[key], torch.full_like(reloaded[key], 0.5)) for key in b_keys
    ), "the marked B matrices did not survive the round trip"
