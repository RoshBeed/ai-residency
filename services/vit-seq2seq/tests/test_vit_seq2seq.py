"""Tests for the multi-digit encoder-decoder Vision Transformer.

The generation tests are the point. `generate` used to terminate the whole batch the
instant any one sequence emitted EOS, silently truncating every other sequence still
decoding -- including in the trainer's own evaluation loop, which is why the reported
accuracy was wrong rather than merely noisy.
"""

import base64
import io
from functools import partial

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from config import Config, Seq2SeqArch
from datasets.multi_digit_dataset import MultiDigitDataset, MultiDigitMNISTDataset
from serving.api import MNIST_MEAN, MNIST_STD, app, preprocess_image
from vit_seq2seq.seq2seq import Seq2SeqVisionTransformer
from vit_seq2seq.trainer import sequence_to_number


def _tiny_seq2seq_arch() -> Seq2SeqArch:
    """A seq2seq model small enough to build and run inside a test."""
    return Seq2SeqArch(
        d_model=16,
        num_encoder_layers=1,
        num_decoder_layers=1,
        heads=2,
        head_dim=8,
        ff_hidden_dim=32,
    )


def test_causal_mask_blocks_only_future_positions() -> None:
    model = Seq2SeqVisionTransformer(_tiny_seq2seq_arch())
    mask = model.create_causal_mask(4, torch.device("cpu"))
    assert mask.dtype == torch.bool
    # True means "do not attend", so the strict upper triangle is masked and the
    # diagonal is not: position i may read itself and everything before it.
    assert not bool(mask.diagonal().any())
    assert bool(mask[0, 1]) and not bool(mask[1, 0])


class _ScriptedDecoderModel(Seq2SeqVisionTransformer):
    """A seq2seq model whose decoder emits a fixed script.

    `generate()` is the loop under test and it drives everything through `decode()`,
    so scripting `decode()` isolates the loop from whether the model is trained. Row 0
    emits EOS on its very first step; every other row emits digits 1, 2, 3 and only
    then EOS. That gap between "the first row finished" and "every row finished" is
    exactly what the bug got wrong.
    """

    def decode(
        self,
        encoder_output: torch.Tensor,
        target_sequence: torch.Tensor,
        target_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, step = target_sequence.size()
        logits = torch.full((batch_size, step, self.vocab_size), -50.0)
        for row in range(batch_size):
            token = self.eos_token if (row == 0 or step >= 4) else step
            logits[row, -1, token] = 50.0
        return logits


def test_generate_does_not_truncate_slower_sequences() -> None:
    """One row finishing early must not end generation for the whole batch.

    Regression test for a real bug: `generate()` used to break the loop the instant
    *any* row emitted EOS, silently truncating every row still mid-sequence. It
    corrupted every batched evaluation while leaving the single-image API path
    correct, because at batch size one "any finished" and "all finished" are the same
    condition, so nothing caught it.
    """
    arch = _tiny_seq2seq_arch()
    model = _ScriptedDecoderModel(arch)
    model.eval()

    max_length = 5
    images = torch.randn(4, arch.in_channels, arch.img_size, arch.img_size)
    with torch.no_grad():
        generated = model.generate(images, start_token=arch.start_token, max_length=max_length)

    # The pre-fix implementation returned width 2 here: row 0's EOS on step one ended
    # the loop before rows 1-3 had emitted a single digit.
    assert generated.shape == (4, max_length)

    # Row 0 finished first and then just repeats EOS rather than sampling on.
    assert bool((generated[0, 1:] == arch.eos_token).all())

    # The other rows still got their full sequence.
    for row in range(1, 4):
        assert sequence_to_number(generated[row], arch.eos_token, arch.start_token) == 123


class _SkewedDecoderModel(Seq2SeqVisionTransformer):
    """A decoder whose argmax is token 7 but which keeps mass on everything else.

    The margin is deliberately small, so sampling lands on something other than 7
    often enough that a handful of steps is a reliable test rather than a flaky one.
    """

    def decode(
        self,
        encoder_output: torch.Tensor,
        target_sequence: torch.Tensor,
        target_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, step = target_sequence.size()
        logits = torch.zeros((batch_size, step, self.vocab_size))
        logits[:, -1, 7] = 1.0
        logits[:, -1, self.eos_token] = -50.0
        return logits


def test_generate_takes_the_argmax_rather_than_sampling() -> None:
    """Reading a number has one right answer, so the decode rule must be greedy.

    Both decode loops used `torch.multinomial`, which costs 21.5 points of
    exact-sequence accuracy on the shipped checkpoint -- 77.03% against 98.53%, both out
    of `make evaluate` -- and made the served answer a different draw per request: one
    call read a clear 206 as 22, with the second digit drawn at probability 0.007. It
    stayed hidden because the trainer's own validation pass sampled as well, so the
    number it reported agreed with the number `make evaluate` reported, and two agreeing
    numbers read as a verified one.
    """
    arch = _tiny_seq2seq_arch()
    model = _SkewedDecoderModel(arch)
    model.eval()

    images = torch.randn(2, arch.in_channels, arch.img_size, arch.img_size)
    with torch.no_grad():
        greedy = model.generate(images, start_token=arch.start_token, max_length=6)

    # Every generated position is the argmax, run to run, with no seed anywhere.
    assert bool((greedy[:, 1:] == 7).all()), greedy

    # Sampling is still reachable, and on this distribution it does something else.
    torch.manual_seed(0)
    with torch.no_grad():
        sampled = model.generate(images, start_token=arch.start_token, max_length=6, sample=True)
    assert not bool((sampled[:, 1:] == 7).all()), sampled


def test_streamed_generation_takes_the_argmax_too() -> None:
    """The SSE path is a second copy of the loop, and it had the same sampling call.

    The streamed confidence must also be the model's own probability for the token it
    chose, not one rescaled by a temperature that does not apply to a greedy decode.
    """
    arch = _tiny_seq2seq_arch()
    model = _SkewedDecoderModel(arch)
    model.eval()

    seen: list[tuple[int, str, float]] = []
    images = torch.randn(1, arch.in_channels, arch.img_size, arch.img_size)
    with torch.no_grad():
        generated = model.generate_with_callback(
            images,
            start_token=arch.start_token,
            max_length=4,
            on_digit=lambda step, digit, prob: seen.append((step, digit, prob)),
        )

    assert bool((generated[:, 1:] == 7).all()), generated
    assert [digit for _, digit, _ in seen] == ["7", "7", "7"]
    expected = float(torch.softmax(torch.tensor([1.0] + [0.0] * (arch.vocab_size - 2)), 0)[0])
    for _, _, prob in seen:
        assert prob == pytest.approx(expected, abs=1e-6)


def test_generate_runs_to_max_length_when_nothing_emits_eos() -> None:
    """With EOS unreachable no row can finish, so the loop must use every step."""
    arch = _tiny_seq2seq_arch()
    model = Seq2SeqVisionTransformer(arch)
    model.eval()

    with torch.no_grad():
        model.output_projection.bias.zero_()
        model.output_projection.bias[arch.eos_token] = -50.0

    max_length = 5
    images = torch.randn(4, arch.in_channels, arch.img_size, arch.img_size)
    with torch.no_grad():
        generated = model.generate(images, start_token=arch.start_token, max_length=max_length)

    assert generated.shape == (4, max_length)
    assert not bool((generated == arch.eos_token).any())


def test_sequence_to_number_stops_at_eos() -> None:
    sequence = torch.tensor([10, 4, 0, 7, 11, 3])
    assert sequence_to_number(sequence, eos_token=11, start_token=10) == 407


def test_an_interior_zero_is_a_digit() -> None:
    """407 is not 47, and the metric used to say it was.

    `sequence_to_number` skipped every 0 rather than only START, so interior zeros
    vanished. Prediction and target both went through it, which made the accuracy
    reproducible and wrong: a model that dropped a zero scored a hit.
    """
    read = partial(sequence_to_number, eos_token=11, start_token=10)
    assert read(torch.tensor([10, 4, 0, 7, 11])) != read(torch.tensor([10, 4, 7, 11]))
    assert read(torch.tensor([10, 1, 0, 0, 11])) == 100
    assert read(torch.tensor([10, 5, 0, 0, 11])) == 500
    # Leading zeros still collapse, which is what zero-padded targets require.
    assert read(torch.tensor([10, 0, 0, 7, 11])) == 7


def test_checkpoint_names_follow_the_architecture() -> None:
    """The name has to track the architecture, which is why it is derived here.

    Asserting a literal meant every legitimate size change failed this test and the
    literal got updated, which checks nothing. What matters is that two models with
    different input sizes cannot share a filename and silently overwrite each other --
    the canvas went from 84px to 112px precisely so no pixel falls outside the patch
    grid, and the two checkpoints must stay distinguishable.
    """
    arch = Config().seq2seq_arch
    expected = (
        f"vit_seq2seq_{arch.img_size}x{arch.img_size}_patch{arch.patch_size}"
        f"_d{arch.d_model}_enc{arch.num_encoder_layers}_dec{arch.num_decoder_layers}"
        f"_heads{arch.heads}_best.pth"
    )
    assert Config().seq2seq_checkpoint_path.name == expected
    assert f"{arch.img_size}x{arch.img_size}" in expected


def test_multi_digit_composites_match_the_requested_size() -> None:
    dataset = MultiDigitMNISTDataset(
        data_dir="data",
        image_size=84,
        max_digits=3,
        samples_per_class=1,
        batch_size=4,
        num_workers=0,
        download=True,
        return_digit_sequence=True,
        add_start_eos=True,
    )
    _, _, test_loader = dataset.get_data_loaders()
    images, targets = next(iter(test_loader))
    assert images.shape[1:] == (1, 84, 84)
    # START, three digits, EOS.
    assert targets.shape[1] == 5
    assert bool((targets[:, 0] == 10).all())
    assert bool((targets[:, -1] == 11).all())


def test_a_composite_survives_the_round_trip_the_api_puts_it_through() -> None:
    """The tensor the loader trains on and the tensor the API serves must agree.

    This is the test the service did not have. `_load_base_mnist` applied
    `transforms.Normalize` and then `__getitem__` normalised the finished composite
    again, so training inputs ran from -1.80 to 8.73 while `preprocess_image` -- which
    decodes a PNG in [0, 1] and normalises once -- produced -0.42 to 2.82 for the same
    picture. Nothing errored and every structural check passed. The cost was 33.7 points
    of exact-sequence accuracy: the checkpoint shipped at the time read 97.93% on the
    loader's own tensors and 64.23% on the tensors the API builds from a faithful PNG of
    the same 3,000 composites. After the fix and a retrain the two paths agree on all
    3,000 of them.

    The check is the round trip rather than the arithmetic: recover pixels from the
    training tensor, encode them as the PNG a caller would send, and push that through
    the real serving path. A tensor whose pixels do not fit [0, 1] cannot survive it,
    which is what makes a second normalisation visible here.
    """
    config = Config()
    arch = config.seq2seq_arch
    rng = np.random.default_rng(0)
    digit_images = {
        digit: [torch.from_numpy(rng.random((1, 28, 28)).astype("float32"))] for digit in range(10)
    }
    dataset = MultiDigitDataset(
        digit_images=digit_images,
        numbers=[407],
        image_size=arch.img_size,
        max_digits=config.max_digits,
        digit_size=config.digit_size,
        augment_probability=0.0,
    )
    trained, _ = dataset[0]

    pixels = trained.squeeze(0).numpy() * MNIST_STD + MNIST_MEAN
    assert pixels.min() >= -1e-6 and pixels.max() <= 1 + 1e-6, (
        f"training pixels land outside [0, 1] ({pixels.min():.3f} to {pixels.max():.3f}), "
        "so the composite is normalised more than once"
    )

    buffer = io.BytesIO()
    Image.fromarray(np.round(pixels * 255).astype("uint8"), mode="L").save(buffer, format="PNG")
    served = preprocess_image(base64.b64encode(buffer.getvalue()).decode(), arch.img_size)

    assert served.shape == trained.unsqueeze(0).shape
    # One 8-bit step, converted into normalised units -- the only loss a PNG imposes.
    tolerance = (1.0 / 255) / MNIST_STD
    assert torch.allclose(served.cpu(), trained.unsqueeze(0), atol=tolerance)


def test_root_endpoint_reports_the_model() -> None:
    with TestClient(app) as client:
        response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["model_loaded"] is body["checkpoint_loaded"]
    assert body["checkpoint_source"] in {"local", "hub", "none"}
    assert body["device"] in {"cuda", "mps", "cpu"}


def test_predict_returns_a_number_for_a_real_image() -> None:
    import base64
    import io

    import numpy as np
    from PIL import Image

    config = Config()
    arch = config.seq2seq_arch
    array = (np.random.default_rng(0).random((arch.img_size, arch.img_size)) * 255).astype("uint8")
    buffer = io.BytesIO()
    Image.fromarray(array, mode="L").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()

    if not config.seq2seq_checkpoint_path.exists():
        pytest.skip("no seq2seq checkpoint on disk to serve (gitignored; on the Hub)")

    with TestClient(app) as client:
        response = client.post("/predict", json={"image_data": encoded})

    assert response.status_code == 200
    body = response.json()
    assert body["pred"].isdigit()
    assert 0.0 <= body["confidence"] <= 1.0


def test_a_truncated_image_is_rejected_rather_than_crashing() -> None:
    """`Image.open` reads the header and decodes the pixels lazily.

    A PNG whose header is intact but whose pixel data is cut off therefore opens
    cleanly and fails later, during `convert`/`resize` -- past whatever guard the
    decode sat behind, so the caller got a 500 for a malformed request. Routing
    through `decode_image`, which calls `.load()` inside its own guard, is what turns
    that into a 422.
    """
    image = Image.new("L", (28, 28), 0)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    full = buffer.getvalue()
    truncated = base64.b64encode(full[: len(full) * 2 // 3]).decode()

    with TestClient(app) as client:
        response = client.post("/predict", json={"image_data": truncated})

    assert response.status_code == 422, response.text
    assert "image" in response.json()["detail"].lower()


def test_undecodable_payloads_are_all_422() -> None:
    """One status code for one class of failure, whatever shape the junk arrives in."""
    with TestClient(app) as client:
        for payload in ("not base64!!", "", "AAAA"):
            response = client.post("/predict", json={"image_data": payload})
            assert response.status_code == 422, f"{payload!r} gave {response.status_code}"
