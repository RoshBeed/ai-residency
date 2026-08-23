"""Dataset construction.

The template generates a small synthetic regression set so `make train` runs end to
end with no download. A real repo replaces `build_datasets` with its own loading and
preprocessing, keeping the signature: take a `Config`, return train and validation
`Dataset` objects already split.
"""

import torch
from torch.utils.data import Dataset, TensorDataset, random_split

from config import Config

__all__ = ["Example", "build_datasets"]

# What a dataset in this repo yields. `TensorDataset` is typed as a variadic tuple,
# so this matches it exactly rather than claiming a fixed pair.
Example = tuple[torch.Tensor, ...]


def build_datasets(config: Config) -> tuple[Dataset[Example], Dataset[Example]]:
    """Build the train and validation datasets.

    The split uses torch's global RNG, so `set_seed` must have been called first for
    the split to be reproducible. That ordering is what makes reported numbers
    comparable between runs.

    Args:
        config: Run configuration.

    Returns:
        A `(train_dataset, val_dataset)` pair, split 80/20.
    """
    features = torch.randn(config.num_examples, config.input_dim)
    weights = torch.randn(config.input_dim, 1)
    targets = features @ weights + 0.1 * torch.randn(config.num_examples, 1)

    dataset = TensorDataset(features, targets)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset
