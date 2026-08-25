"""Tests for the multi-digit encoder-decoder Vision Transformer.

The generation tests are the point. `generate` used to terminate the whole batch the
instant any one sequence emitted EOS, silently truncating every other sequence still
decoding -- including in the trainer's own evaluation loop, which is why the reported
accuracy was wrong rather than merely noisy.
"""

import base64
import io

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from config import Config, Seq2SeqArch
from datasets.multi_digit_dataset import MultiDigitMNISTDataset
from serving.api import app
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
    assert sequence_to_number(sequence, eos_token=11, start_token=10) == 47


def test_checkpoint_names_follow_the_architecture() -> None:
    config = Config()
    assert config.seq2seq_checkpoint_path.name == (
        "vit_seq2seq_84x84_patch16_d256_enc6_dec6_heads8_best.pth"
    )


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
