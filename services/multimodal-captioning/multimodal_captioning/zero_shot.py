"""Part 1: zero-shot image classification with CLIP alone.

No training and no adapter. Each candidate label is written as `"a photo of a {item}"`,
run through CLIP's text tower, and compared by dot product against the projected image
embedding. Whichever label scores highest wins. The point of this half of the repo is
that CLIP's joint image/text space makes classification a retrieval problem, so a new
class costs a sentence rather than a labelled dataset.

The ten sample images under `data/images/` are named after the labels they should match,
which is what makes the result checkable at a glance.
"""

from dataclasses import dataclass

import torch
from ai_residency_common import get_logger
from PIL import Image

from config import Config
from multimodal_captioning.models import load_backbones
from multimodal_captioning.samples import sample_image

__all__ = ["Prediction", "classify_images", "main"]

logger = get_logger(__name__)


@dataclass(frozen=True)
class Prediction:
    """One image's zero-shot result.

    Attributes:
        item: The label the image file is named after, which is the ground truth.
        predicted: The label CLIP scored highest.
        score: That label's similarity score.
    """

    item: str
    predicted: str
    score: float

    @property
    def correct(self) -> bool:
        """Whether the top-scoring label matches the file name."""
        return self.item == self.predicted


def classify_images(config: Config) -> list[Prediction]:
    """Classify every sample image against every label.

    Args:
        config: Supplies the label list, the image directory and the model names.

    Returns:
        One `Prediction` per label in `config.zero_shot_items`.
    """
    backbones = load_backbones(config)
    device = backbones.device
    items = list(config.zero_shot_items)
    classifiers = [f"a photo of a {item}" for item in items]

    text_inputs = backbones.clip_tokenizer(classifiers, return_tensors="pt", padding=True)
    text_inputs = {key: value.to(device) for key, value in text_inputs.items()}

    with torch.no_grad():
        text_features = backbones.text_encoder(**text_inputs).pooler_output
        text_features_proj = backbones.clip_model.text_projection(text_features)

    predictions: list[Prediction] = []
    for item in items:
        path = sample_image(config, item)
        if path is None:
            raise FileNotFoundError(
                f"Sample image {item}.jpg is neither in {config.image_dir} nor in "
                f"{config.hf_dataset_repo_id}"
            )
        image = Image.open(path)
        processed = backbones.image_processor(image, return_tensors="pt")
        image_inputs = {key: value.to(device) for key, value in processed.items()}

        with torch.no_grad():
            image_features = backbones.vision_encoder(**image_inputs).pooler_output
            image_features_proj = backbones.clip_model.visual_projection(image_features)

        similarities = (image_features_proj @ text_features_proj.T).squeeze(0)
        best_index = int(similarities.argmax().item())
        predictions.append(
            Prediction(
                item=item,
                predicted=items[best_index],
                score=float(similarities[best_index].item()),
            )
        )

    return predictions


def main() -> None:
    """Entry point for `make zero-shot`."""
    config = Config()
    predictions = classify_images(config)
    for prediction in predictions:
        logger.info(
            "%s: best match is 'a photo of a %s' (score: %.3f)%s",
            prediction.item,
            prediction.predicted,
            prediction.score,
            "" if prediction.correct else "  <- wrong",
        )
    correct = sum(prediction.correct for prediction in predictions)
    logger.info("Zero-shot accuracy: %d/%d", correct, len(predictions))


if __name__ == "__main__":
    main()
