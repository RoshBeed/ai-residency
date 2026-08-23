"""Tests for the Week 3 Vision Transformer repo.

Three things are worth guarding here: that the from-scratch transformer blocks are
shape-correct, that a training run is reproducible, and that batched generation does
not truncate sequences. The last one is a regression test for a real bug (see
`test_generate_does_not_truncate_the_batch`).
"""

from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from config import ClassifierArch, Config, Seq2SeqArch
from datasets.multi_digit_dataset import MultiDigitMNISTDataset
from models.classifier.classifier import VisionTransformer
from models.classifier.trainer import train as train_classifier
from models.seq2seq.seq2seq import Seq2SeqVisionTransformer
from models.seq2seq.trainer import sequence_to_number
from serving.api import app
from transformer.attention import MultiHeadAttention
from transformer.patch_embedding import PatchEmbedding


def _tiny_classifier_arch() -> ClassifierArch:
    """A classifier small enough to build and run inside a test."""
    return ClassifierArch(d_model=16, num_encoder_layers=1, heads=2, head_dim=8, ff_hidden_dim=32)


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


def test_patch_embedding_produces_one_token_per_patch() -> None:
    arch = _tiny_classifier_arch()
    embedding = PatchEmbedding(arch.img_size, arch.patch_size, arch.in_channels, arch.d_model)
    output = embedding(torch.randn(2, arch.in_channels, arch.img_size, arch.img_size))
    assert output.shape == (2, (arch.img_size // arch.patch_size) ** 2, arch.d_model)


def test_attention_rejects_a_width_it_cannot_split() -> None:
    with pytest.raises(ValueError, match="divisible"):
        MultiHeadAttention(heads=5, d_model=16, head_dim=4)


def test_causal_mask_blocks_only_future_positions() -> None:
    model = Seq2SeqVisionTransformer(_tiny_seq2seq_arch())
    mask = model.create_causal_mask(4, torch.device("cpu"))
    assert mask.dtype == torch.bool
    # True means "do not attend", so the strict upper triangle is masked and the
    # diagonal is not: position i may read itself and everything before it.
    assert not bool(mask.diagonal().any())
    assert bool(mask[0, 1]) and not bool(mask[1, 0])


def test_classifier_returns_one_logit_per_class() -> None:
    arch = _tiny_classifier_arch()
    model = VisionTransformer(arch)
    logits = model(torch.randn(3, arch.in_channels, arch.img_size, arch.img_size))
    assert logits.shape == (3, arch.num_classes)


def test_classifier_rejects_the_wrong_input_size() -> None:
    model = VisionTransformer(_tiny_classifier_arch())
    with pytest.raises(ValueError, match="Expected input size"):
        model(torch.randn(1, 1, 32, 32))


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
    assert config.classifier_checkpoint_path.name == (
        "vit_cls_28x28_patch4_d128_layers6_heads8_best.pth"
    )
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


def _fast_classifier_config(tmp_path: Path) -> Config:
    """A classifier run small enough to train inside a test."""
    return Config(
        classifier_epochs=1,
        batch_size=256,
        num_workers=0,
        checkpoint_dir=tmp_path,
        classifier_arch=_tiny_classifier_arch(),
        hf_repo_id=None,
        wandb_project=None,
    )


def test_classifier_training_is_reproducible_and_writes_a_checkpoint(tmp_path: Path) -> None:
    """Same seed, same result. Unseeded runs hid real bugs elsewhere in this portfolio."""
    first = train_classifier(_fast_classifier_config(tmp_path / "a"))
    second = train_classifier(_fast_classifier_config(tmp_path / "b"))
    assert first == second
    assert _fast_classifier_config(tmp_path / "a").classifier_checkpoint_path.exists()


def test_root_endpoint_reports_both_models() -> None:
    with TestClient(app) as client:
        response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert set(body["models_loaded"]) == {"vit", "seq2seq"}
    assert body["device"] in {"cuda", "mps", "cpu"}


def test_served_predictions_match_in_process_predictions() -> None:
    """The API must feed the model exactly what the training loaders feed it.

    Regression test for a real bug: `preprocess_image` resized and scaled to [0, 1]
    but never applied the MNIST normalisation both dataset wrappers apply, so the
    served classifier scored 15.6% on a test batch it scores 95.3% on in process.
    Nothing raised — the model was simply shown values it had never been trained on.
    """
    config = Config()
    if not config.classifier_checkpoint_path.exists():
        pytest.skip("no classifier checkpoint on disk to serve (gitignored; on the Hub)")

    import base64
    import io

    import numpy as np
    from PIL import Image

    from ai_residency_common import load_checkpoint
    from datasets.mnist_dataset import MNISTDataset

    mnist = MNISTDataset(data_dir=config.data_dir, image_size=28, batch_size=16, num_workers=0)
    _, _, loader = mnist.get_data_loaders()
    images, labels = next(iter(loader))

    model = VisionTransformer(config.classifier_arch)
    load_checkpoint(config.classifier_checkpoint_path, model)
    model.eval()
    with torch.no_grad():
        in_process = model(images).argmax(dim=-1)

    served: list[int] = []
    with TestClient(app) as client:
        for index in range(len(labels)):
            # Undo the loader's normalisation to get back to the PNG a caller sends.
            array = ((images[index, 0].numpy() * 0.3081 + 0.1307) * 255).clip(0, 255)
            buffer = io.BytesIO()
            Image.fromarray(array.astype(np.uint8), mode="L").save(buffer, format="PNG")
            encoded = base64.b64encode(buffer.getvalue()).decode()
            response = client.post("/predict", json={"image_data": encoded, "model_type": "vit"})
            assert response.status_code == 200
            served.append(int(response.json()["pred"]))

    agreement = sum(int(a == b) for a, b in zip(served, in_process.tolist(), strict=True))
    # PNG round-tripping costs a little precision, so this is not exact equality.
    assert agreement >= len(labels) - 1, f"served {served} vs in-process {in_process.tolist()}"


def test_predict_returns_a_digit_for_a_real_image() -> None:
    import base64
    import io

    import numpy as np
    from PIL import Image

    config = Config()
    array = (np.random.default_rng(0).random((28, 28)) * 255).astype("uint8")
    buffer = io.BytesIO()
    Image.fromarray(array, mode="L").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()

    if not config.classifier_checkpoint_path.exists():
        pytest.skip("no classifier checkpoint on disk to serve (gitignored; on the Hub)")

    with TestClient(app) as client:
        response = client.post("/predict", json={"image_data": encoded, "model_type": "vit"})

    assert response.status_code == 200
    body = response.json()
    assert body["pred"] in [str(digit) for digit in range(10)]
    assert 0.0 <= body["confidence"] <= 1.0
