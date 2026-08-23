"""Dataset wrappers for the two tasks."""

from datasets.mnist_dataset import MNISTDataset
from datasets.multi_digit_dataset import (
    MultiDigitDataset,
    MultiDigitMNISTDataset,
)

__all__ = ["MNISTDataset", "MultiDigitDataset", "MultiDigitMNISTDataset"]
