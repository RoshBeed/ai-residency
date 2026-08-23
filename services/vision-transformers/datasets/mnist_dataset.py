"""Single-digit MNIST loading, split and batching."""

from pathlib import Path

import torch
from ai_residency_common import get_logger
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

__all__ = ["Example", "MNISTDataset"]

logger = get_logger(__name__)

# What these loaders yield per batch: images and their labels. The underlying
# dataset yields a plain int label, which the default collate turns into a tensor.
Example = tuple[torch.Tensor, torch.Tensor]


class MNISTDataset:
    """Torchvision MNIST with a fixed train/validation split and ready loaders.

    The validation split is carved out of the training set with its own seeded
    generator, so the split is identical between a training run and a later
    evaluation regardless of what else has consumed the global RNG.
    """

    def __init__(
        self,
        data_dir: str | Path = "data",
        image_size: int = 28,
        train_split: float = 0.8,
        batch_size: int = 64,
        num_workers: int = 4,
        download: bool = True,
        seed: int = 42,
    ) -> None:
        """Load MNIST and build the loaders.

        Args:
            data_dir: Where the MNIST cache lives.
            image_size: Side length images are resized to.
            train_split: Fraction of the training set kept for training.
            batch_size: Examples per batch.
            num_workers: DataLoader worker processes.
            download: Fetch MNIST if the cache is empty.
            seed: Seed for the train/validation split.
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.train_split = train_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        self._load_datasets(download)
        self._create_data_loaders()

    def _load_datasets(self, download: bool) -> None:
        """Load MNIST and split the training set."""
        full_train_dataset = datasets.MNIST(
            str(self.data_dir), train=True, download=download, transform=self.transform
        )
        self.test_dataset = datasets.MNIST(
            str(self.data_dir), train=False, download=download, transform=self.transform
        )

        train_size = int(self.train_split * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size

        self.train_dataset, self.val_dataset = random_split(
            full_train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.seed),
        )
        self.test_size: int = len(self.test_dataset)

        logger.info(
            "MNIST loaded: %d train, %d validation, %d test",
            len(self.train_dataset),
            len(self.val_dataset),
            self.test_size,
        )

    def _create_data_loaders(self) -> None:
        """Build the three loaders."""
        self.train_loader: DataLoader[Example] = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        self.val_loader: DataLoader[Example] = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        self.test_loader: DataLoader[Example] = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def get_data_loaders(
        self,
    ) -> tuple[DataLoader[Example], DataLoader[Example], DataLoader[Example]]:
        """Return the train, validation and test loaders."""
        return self.train_loader, self.val_loader, self.test_loader

    def get_dataset_sizes(self) -> tuple[int, int, int]:
        """Return the number of examples in each split."""
        return len(self.train_dataset), len(self.val_dataset), self.test_size
