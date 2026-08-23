"""Tests for the single-digit Vision Transformer classifier.

Two things are worth guarding: that the from-scratch transformer blocks are
shape-correct, and that a training run is reproducible and writes a checkpoint whose
name matches the architecture that produced it.
"""

from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from config import ClassifierArch, Config
from serving.api import app
from vision_transformer.attention import MultiHeadAttention
from vision_transformer.patch_embedding import PatchEmbedding
from vit_classifier.classifier import VisionTransformer
from vit_classifier.trainer import train as train_classifier


def _tiny_classifier_arch() -> ClassifierArch:
    """A classifier small enough to build and run inside a test."""
    return ClassifierArch(d_model=16, num_encoder_layers=1, heads=2, head_dim=8, ff_hidden_dim=32)


def test_patch_embedding_produces_one_token_per_patch() -> None:
    arch = _tiny_classifier_arch()
    embedding = PatchEmbedding(arch.img_size, arch.patch_size, arch.in_channels, arch.d_model)
    output = embedding(torch.randn(2, arch.in_channels, arch.img_size, arch.img_size))
    assert output.shape == (2, (arch.img_size // arch.patch_size) ** 2, arch.d_model)


def test_attention_rejects_a_width_it_cannot_split() -> None:
    with pytest.raises(ValueError, match="divisible"):
        MultiHeadAttention(heads=5, d_model=16, head_dim=4)


def test_classifier_returns_one_logit_per_class() -> None:
    arch = _tiny_classifier_arch()
    model = VisionTransformer(arch)
    logits = model(torch.randn(3, arch.in_channels, arch.img_size, arch.img_size))
    assert logits.shape == (3, arch.num_classes)


def test_classifier_rejects_the_wrong_input_size() -> None:
    model = VisionTransformer(_tiny_classifier_arch())
    with pytest.raises(ValueError, match="Expected input size"):
        model(torch.randn(1, 1, 32, 32))


def test_checkpoint_names_follow_the_architecture() -> None:
    config = Config()
    assert config.classifier_checkpoint_path.name == (
        "vit_cls_28x28_patch4_d128_layers6_heads8_best.pth"
    )


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


def test_root_endpoint_reports_the_model() -> None:
    with TestClient(app) as client:
        response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["model_loaded"] is body["checkpoint_loaded"]
    assert body["checkpoint_source"] in {"local", "hub", "none"}
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
            response = client.post("/predict", json={"image_data": encoded})
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
        response = client.post("/predict", json={"image_data": encoded})

    assert response.status_code == 200
    body = response.json()
    assert body["pred"] in [str(digit) for digit in range(10)]
    assert 0.0 <= body["confidence"] <= 1.0
