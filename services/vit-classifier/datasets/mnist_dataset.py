"""Single-digit MNIST loading, split and batching."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from ai_residency_common import get_logger

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
        augment_translate: float = 0.25,
        augment_scale: tuple[float, float] = (0.6, 1.4),
        augment_degrees: float = 12.0,
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
            augment_translate: Largest shift, as a fraction of the frame, applied to
                training digits.
            augment_scale: Scale range applied to training digits.
            augment_degrees: Rotation range, plus or minus, applied to training digits.
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.train_split = train_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Validation and test see MNIST as it ships, so the headline number stays
        # comparable with every published MNIST result.
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        # Training does not. **MNIST is size-normalised and centred by centre of mass**:
        # every digit's ink is scaled into a 20x20 box inside the 28x28 frame and moved
        # so its mass sits in the middle. Nothing a person draws has either property.
        # `serving/api.py` plain-resizes whatever arrives, so a digit drawn off-centre
        # reaches a model that has never seen one -- measured on the checkpoint trained
        # without this, at 60% fill: 96.5% centred, 66.5% a quarter of the way to a
        # corner, **24.5% halfway**, 29.0% pushed to the top. Position was not a
        # nuisance the model had learned to ignore; it was a constant it had learned to
        # rely on.
        self.train_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomAffine(
                    degrees=augment_degrees,
                    translate=(augment_translate, augment_translate),
                    scale=augment_scale,
                    fill=0,
                ),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        self._load_datasets(download)
        self._create_data_loaders()

    def _load_datasets(self, download: bool) -> None:
        """Load MNIST and split the training set.

        The same underlying images are opened twice, under the two transforms, and the
        split indexes into whichever copy that half should see -- `random_split` cannot
        do this, because it hands both halves one dataset and so one transform. The
        indices come from a seeded permutation, so the split is identical between a
        training run and a later evaluation however the global RNG has been used.
        """
        augmented = datasets.MNIST(
            str(self.data_dir), train=True, download=download, transform=self.train_transform
        )
        plain = datasets.MNIST(
            str(self.data_dir), train=True, download=download, transform=self.transform
        )
        self.test_dataset = datasets.MNIST(
            str(self.data_dir), train=False, download=download, transform=self.transform
        )

        train_size = int(self.train_split * len(augmented))
        order = torch.randperm(len(augmented), generator=torch.Generator().manual_seed(self.seed))
        self.train_dataset = Subset(augmented, order[:train_size].tolist())
        self.val_dataset = Subset(plain, order[train_size:].tolist())
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
