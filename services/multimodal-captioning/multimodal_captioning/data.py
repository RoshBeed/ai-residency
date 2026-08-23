"""Dataset construction for adapter training.

Flickr30k on the Hub ships everything in one `test` split of 31,014 images, five human
captions each. A run samples the first `num_samples` of them and splits that sample into
train and held-out sets, so the loss the best checkpoint is chosen on is not the loss it
was fitted on.

Only the first of the five captions per image is used. That is what the original
training run did and it is worth being explicit about rather than silently averaging:
the other four are perfectly good supervision this repo simply never touches.
"""

from typing import Any

import torch
from ai_residency_common import get_logger
from datasets import load_dataset
from PIL.Image import Image
from torch.utils.data import Dataset, random_split

from config import Config
from multimodal_captioning.models import Backbones

__all__ = ["Batch", "Example", "FlickrDataset", "build_collate_fn", "build_datasets"]

logger = get_logger(__name__)

# One image and the caption it should produce. PEP 695 `type` aliases are unavailable at
# the 3.11 floor this portfolio targets, so these are plain aliases.
Example = tuple[Image, str]

# What `collate_fn` hands the training loop: CLIP's pixel tensors, and the tokenized
# captions with their attention mask.
Batch = tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]


class FlickrDataset(Dataset[Example]):
    """Image/caption pairs drawn from a Hugging Face dataset split.

    Attributes:
        rows: The underlying split, already sliced to the sample size.
    """

    def __init__(self, rows: Any) -> None:
        """Wrap an already-sliced Hugging Face split.

        Args:
            rows: A `datasets.Dataset` yielding `image` and `caption` fields.
        """
        self.rows = rows

    def __len__(self) -> int:
        """Number of image/caption pairs."""
        return len(self.rows)

    def __getitem__(self, index: int) -> Example:
        """Return one image and its first caption.

        Args:
            index: Position in the split.

        Returns:
            The PIL image and the caption string.
        """
        row = self.rows[index]
        caption = row["caption"]
        if isinstance(caption, list):
            caption = caption[0]
        image: Image = row["image"]
        return image, str(caption)


def build_datasets(config: Config) -> tuple[Dataset[Example], Dataset[Example]]:
    """Build the train and held-out datasets.

    The split uses torch's global RNG, so `set_seed` must run first for it to be
    reproducible. That ordering is what makes a reported baseline comparable between
    runs.

    Args:
        config: Supplies the dataset name, revision, split, sample size and train share.

    Returns:
        A `(train_dataset, val_dataset)` pair.
    """
    splits = load_dataset(config.dataset_name, revision=config.dataset_revision)
    rows = splits[config.dataset_split].select(range(config.num_samples))
    dataset = FlickrDataset(rows)

    train_size = int(config.train_fraction * len(dataset))
    val_size = len(dataset) - train_size
    logger.info(
        "Sampled %d %s pairs: %d train, %d held out",
        len(dataset),
        config.dataset_name,
        train_size,
        val_size,
    )
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset


def build_collate_fn(config: Config, backbones: Backbones) -> Any:
    """Return a `collate_fn` that batches images and captions.

    Images and captions are processed as whole batches rather than one at a time, which
    is what makes this the real training path: CLIP's processor takes a list of images,
    and the tokenizer pads the captions to a common length.

    Args:
        config: Supplies the caption truncation length.
        backbones: Supplies the image processor and the Qwen3 tokenizer.

    Returns:
        A callable suitable for `DataLoader(collate_fn=...)`.
    """

    def collate_fn(batch: list[Example]) -> Batch:
        images = [image for image, _ in batch]
        captions = [caption for _, caption in batch]
        image_inputs = backbones.image_processor(images, return_tensors="pt")
        tokenized = backbones.qwen_tokenizer(
            captions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.max_caption_length,
        )
        return dict(image_inputs), dict(tokenized)

    return collate_fn
