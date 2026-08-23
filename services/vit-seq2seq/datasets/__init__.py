"""Multi-digit MNIST wrapper: three digits side by side, read as a sequence."""

from datasets.multi_digit_dataset import (
    MultiDigitDataset,
    MultiDigitMNISTDataset,
)

__all__ = ["MultiDigitDataset", "MultiDigitMNISTDataset"]
