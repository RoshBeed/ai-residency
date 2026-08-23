"""Multi-digit MNIST: composites of N single digits, generated on the fly.

Nothing off the shelf produces "an image of the number 407", so this builds them:
pick a real MNIST sample per digit, resize, jitter and occasionally rotate it, then
paste the digits side by side onto one canvas. Generating per access rather than
materialising a fixed set means every epoch sees different handwriting for the same
number, which is the cheap augmentation this task needs.
"""

import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from ai_residency_common import get_logger

__all__ = ["Example", "MultiDigitDataset", "MultiDigitMNISTDataset"]

logger = get_logger(__name__)

# An image and its label: either the digit-token sequence the seq2seq decoder is
# trained against, or a scalar tensor holding the number the composite depicts.
Example = tuple[torch.Tensor, torch.Tensor]


class MultiDigitDataset(Dataset[Example]):
    """Composites of `max_digits` MNIST digits, built on access."""

    def __init__(
        self,
        digit_images: dict[int, list[torch.Tensor]],
        numbers: Sequence[int],
        image_size: int,
        max_digits: int,
        return_digit_sequence: bool = False,
        add_start_eos: bool = False,
        start_token: int = 10,
        eos_token: int = 11,
    ) -> None:
        """Build the dataset.

        Args:
            digit_images: Real MNIST samples, keyed by the digit they show.
            numbers: The numbers this split covers, one entry per example.
            image_size: Side length of the square canvas.
            max_digits: Digits per composite.
            return_digit_sequence: Return a token sequence instead of the number.
            add_start_eos: Wrap the token sequence in START and EOS.
            start_token: Id of the START token.
            eos_token: Id of the EOS token.
        """
        self.digit_images = digit_images
        self.numbers = numbers
        self.image_size = image_size
        self.max_digits = max_digits
        self.return_digit_sequence = return_digit_sequence
        self.add_start_eos = add_start_eos
        self.start_token = start_token
        self.eos_token = eos_token

    def __len__(self) -> int:
        """Number of examples in this split."""
        return len(self.numbers)

    def __getitem__(self, index: int) -> Example:
        """Compose one example.

        Args:
            index: Position in this split.

        Returns:
            The composite image and its label.
        """
        number = self.numbers[index]
        digits = [int(d) for d in str(number).zfill(self.max_digits)]
        image = self._create_multi_digit_image(digits)
        # MNIST's own normalisation, matching what the single-digit loader applies.
        image = (image - 0.1307) / 0.3081

        if not self.return_digit_sequence:
            return image, torch.tensor(number, dtype=torch.long)

        sequence = list(digits)
        if self.add_start_eos:
            sequence = [self.start_token, *sequence, self.eos_token]
        return image, torch.tensor(sequence, dtype=torch.long)

    def _create_multi_digit_image(self, digits: list[int]) -> torch.Tensor:
        """Paste one real MNIST sample per digit onto a single canvas."""
        digit_size = self.image_size // self.max_digits
        spacing = (self.image_size - self.max_digits * digit_size) // (self.max_digits + 1)
        canvas = torch.zeros(1, self.image_size, self.image_size)

        for position, digit in enumerate(digits):
            digit_image = random.choice(self.digit_images[digit])
            digit_image = F.interpolate(
                digit_image.unsqueeze(0),
                size=(digit_size, digit_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

            max_jitter = spacing // 2
            x_jitter = random.randint(-max_jitter, max_jitter) if max_jitter > 0 else 0
            y_jitter = random.randint(-max_jitter, max_jitter) if max_jitter > 0 else 0
            x_start = spacing + position * (digit_size + spacing) + x_jitter
            y_start = (self.image_size - digit_size) // 2 + y_jitter
            x_start = max(0, min(self.image_size - digit_size, x_start))
            y_start = max(0, min(self.image_size - digit_size, y_start))

            if random.random() < 0.3:
                digit_image = self._random_affine(digit_image)

            canvas[:, y_start : y_start + digit_size, x_start : x_start + digit_size] = digit_image

        return canvas

    def _random_affine(self, digit_image: torch.Tensor) -> torch.Tensor:
        """Apply a small random rotation and scale to a single digit."""
        angle = random.uniform(-10, 10)
        scale = random.uniform(0.9, 1.1)
        cos, sin = np.cos(np.radians(angle)), np.sin(np.radians(angle))
        theta = torch.tensor(
            [[scale * cos, -scale * sin, 0.0], [scale * sin, scale * cos, 0.0]],
            dtype=torch.float32,
        ).unsqueeze(0)
        grid = F.affine_grid(theta, list(digit_image.unsqueeze(0).size()), align_corners=False)
        return F.grid_sample(digit_image.unsqueeze(0), grid, align_corners=False).squeeze(0)


class MultiDigitMNISTDataset:
    """Builds balanced train/validation/test splits of multi-digit composites.

    Every number in range appears in the training split, which is deliberate: the
    task is reading digits off a canvas, not extrapolating to unseen numbers, so
    holding numbers out would measure something the model was never asked to do.
    Validation and test cover a subset of the same numbers with freshly composed
    handwriting.
    """

    def __init__(
        self,
        data_dir: str | Path = "data",
        image_size: int = 84,
        max_digits: int = 3,
        train_split: float = 0.8,
        samples_per_class: int = 10,
        batch_size: int = 64,
        num_workers: int = 4,
        download: bool = True,
        seed: int = 42,
        return_digit_sequence: bool = False,
        add_start_eos: bool = False,
        start_token: int = 10,
        eos_token: int = 11,
    ) -> None:
        """Load base MNIST, build the splits and their loaders.

        Args:
            data_dir: Where the MNIST cache lives.
            image_size: Side length of the square canvas.
            max_digits: Digits per composite.
            train_split: Kept for interface parity; every number is used for training.
            samples_per_class: Composites generated per distinct number.
            batch_size: Examples per batch.
            num_workers: DataLoader worker processes.
            download: Fetch MNIST if the cache is empty.
            seed: Seed for the number shuffles and the split sampling.
            return_digit_sequence: Return token sequences instead of numbers.
            add_start_eos: Wrap those sequences in START and EOS.
            start_token: Id of the START token.
            eos_token: Id of the EOS token.
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.max_digits = max_digits
        self.train_split = train_split
        self.samples_per_class = samples_per_class
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.return_digit_sequence = return_digit_sequence
        self.add_start_eos = add_start_eos
        self.start_token = start_token
        self.eos_token = eos_token

        random.seed(seed)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self._load_base_mnist(download)
        self._split_numbers()
        self._create_data_loaders()

    def _load_base_mnist(self, download: bool) -> None:
        """Bucket every MNIST training image by the digit it shows."""
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        full_mnist = datasets.MNIST(
            str(self.data_dir), train=True, download=download, transform=transform
        )

        self.digit_images: dict[int, list[torch.Tensor]] = {digit: [] for digit in range(10)}
        for image, label in full_mnist:
            digit = int(label.item()) if isinstance(label, torch.Tensor) else int(label)
            self.digit_images[digit].append(image)

        logger.info(
            "Base MNIST loaded: %s",
            ", ".join(f"{digit}={len(images)}" for digit, images in self.digit_images.items()),
        )

    def _split_numbers(self) -> None:
        """Choose which numbers each split covers, and how many times each repeats."""
        all_numbers = list(range(10**self.max_digits))
        random.shuffle(all_numbers)

        self.train_numbers = list(all_numbers)
        split_size = min(100, len(all_numbers) // 10)
        self.val_numbers = random.sample(all_numbers, split_size)
        remaining = [n for n in all_numbers if n not in set(self.val_numbers)]
        self.test_numbers = random.sample(remaining, split_size)

        self.train_indices = self.train_numbers * self.samples_per_class
        self.val_indices = self.val_numbers * self.samples_per_class
        self.test_indices = self.test_numbers * self.samples_per_class

        random.shuffle(self.train_indices)
        random.shuffle(self.val_indices)
        random.shuffle(self.test_indices)

        logger.info(
            "Multi-digit splits: %d train / %d val / %d test examples "
            "over %d / %d / %d distinct numbers",
            len(self.train_indices),
            len(self.val_indices),
            len(self.test_indices),
            len(set(self.train_numbers)),
            len(set(self.val_numbers)),
            len(set(self.test_numbers)),
        )

    def _build_split(self, numbers: Sequence[int]) -> MultiDigitDataset:
        """Wrap one split's numbers in a dataset."""
        return MultiDigitDataset(
            self.digit_images,
            numbers,
            self.image_size,
            self.max_digits,
            self.return_digit_sequence,
            self.add_start_eos,
            self.start_token,
            self.eos_token,
        )

    def _create_data_loaders(self) -> None:
        """Build the three datasets and their loaders."""
        self.train_dataset = self._build_split(self.train_indices)
        self.val_dataset = self._build_split(self.val_indices)
        self.test_dataset = self._build_split(self.test_indices)

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
        return len(self.train_indices), len(self.val_indices), len(self.test_indices)
