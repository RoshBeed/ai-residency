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
        digit_size: int = 28,
        min_digit_size: int = 10,
        min_gap: int = 2,
        augment_probability: float = 1.0,
        rotation_degrees: float = 15.0,
        shear_degrees: float = 12.0,
        scale_range: tuple[float, float] = (0.75, 1.15),
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
            max_digits: Widest composite, which fixes how many cells there are.
            digit_size: Largest side length a digit is pasted at. The actual cell is
                drawn per composite between `min_digit_size` and whatever fits, so the
                model sees every scale a person can draw rather than one.
            min_digit_size: Smallest cell a digit is pasted at.
            min_gap: Smallest gap between digits, and from the canvas edge.
            augment_probability: Share of digits given a random affine warp.
            rotation_degrees: Rotation range, plus or minus.
            shear_degrees: Shear range, plus or minus.
            scale_range: Scale range as (low, high).
            return_digit_sequence: Return a token sequence instead of the number.
            add_start_eos: Wrap the token sequence in START and EOS.
            start_token: Id of the START token.
            eos_token: Id of the EOS token.
        """
        self.digit_images = digit_images
        self.numbers = numbers
        self.image_size = image_size
        self.max_digits = max_digits
        self.digit_size = digit_size
        self.min_digit_size = min_digit_size
        self.min_gap = min_gap
        self.augment_probability = augment_probability
        self.rotation_degrees = rotation_degrees
        self.shear_degrees = shear_degrees
        self.scale_range = scale_range
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
        # The label is the number's own digits, not a zero-padded field. `zfill` made
        # every sequence exactly `max_digits` long, which pinned EOS to one position and
        # taught the decoder to emit a fixed count rather than to decide when to stop.
        digits = [int(d) for d in str(number)]
        image = self._create_multi_digit_image(digits)
        # MNIST's own normalisation, applied once, to the finished composite. This is
        # the same arithmetic `preprocess_image` in `serving/api.py` performs on a
        # decoded PNG, which is what makes the training tensor and the served tensor
        # the same thing.
        image = (image - 0.1307) / 0.3081

        if not self.return_digit_sequence:
            return image, torch.tensor(number, dtype=torch.long)

        sequence = list(digits)
        if self.add_start_eos:
            sequence = [self.start_token, *sequence, self.eos_token]
            # Padded to a fixed width so a batch collates, and padded *with EOS* rather
            # than a new PAD id on purpose: a PAD token would widen the decoder's output
            # head and make every existing checkpoint unloadable, which would destroy the
            # before/after comparison this change exists to report. The trainer masks
            # every position after the first EOS out of the loss, so the repeats cost
            # nothing and the model is never trained to predict padding.
            width = self.max_digits + 2
            sequence.extend([self.eos_token] * (width - len(sequence)))
        return image, torch.tensor(sequence, dtype=torch.long)

    def _create_multi_digit_image(self, digits: list[int]) -> torch.Tensor:
        """Paste one real MNIST sample per digit onto a single canvas.

        **The geometry varies per composite, not just the glyphs inside it.** The first
        version of this pasted every digit at exactly `digit_size` into one of
        `max_digits` evenly spaced slots in a centred strip, and applied rotation, scale,
        shear and translation *within* that fixed cell. That warps the handwriting while
        leaving the layout identical in every example, so the model learns one scale and
        one position. Measured on the shipped checkpoint, reading numbers composed at
        other cell sizes: 92-100% exact-sequence between 22 and 47 pixels, **3.3% at 20
        pixels and 10.0% at 95**. A person drawing a single digit on the frontend's
        280x280 canvas fills most of it, which lands around 95 pixels once scaled, so the
        service was at its worst exactly where its only human input lives.

        So the cell size is drawn per composite across the whole range that fits, the
        gaps between digits vary, and the group is placed anywhere on the canvas rather
        than centred. Three big digits do not fit in a small square, so the upper bound
        depends on how many are drawn -- that is a property of the canvas rather than a
        leak, and the ranges overlap heavily at the small end, so scale does not tell the
        decoder how many digits to expect.
        """
        drawn = len(digits)
        fits = (self.image_size - (drawn + 1) * self.min_gap) // drawn
        largest = max(min(fits, self.digit_size), self.min_digit_size)
        cell = random.randint(self.min_digit_size, largest)

        # The gaps have to come out of what the digits leave behind, or three large
        # cells plus three generous gaps run off the edge and the last digit is pasted
        # into a slice that does not exist.
        budget = self.image_size - drawn * cell - 2 * self.min_gap
        per_gap = budget // max(drawn - 1, 1)
        high = max(self.min_gap, min(cell // 2, per_gap))
        gaps = [random.randint(self.min_gap, high) for _ in range(drawn - 1)]

        span = drawn * cell + sum(gaps)
        x = random.randint(0, max(0, self.image_size - span))
        top = random.randint(0, max(0, self.image_size - cell))

        canvas = torch.zeros(1, self.image_size, self.image_size)
        for position, digit in enumerate(digits):
            digit_image = random.choice(self.digit_images[digit])
            if random.random() < self.augment_probability:
                digit_image = self._random_affine(digit_image)
            digit_image = (
                F.interpolate(
                    digit_image.unsqueeze(0),
                    size=(cell, cell),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(0)
                .clamp(0.0, 1.0)
            )

            # Each digit sits on the same baseline, wandering a little, the way a written
            # number does -- not independently placed, which would stop being a number.
            y = min(max(top + random.randint(-cell // 8, cell // 8), 0), self.image_size - cell)
            x = min(x, self.image_size - cell)
            # `maximum` rather than assignment: cells can touch at the smallest gaps, and
            # overwriting would erase the edge of the digit already there.
            canvas[:, y : y + cell, x : x + cell] = torch.maximum(
                canvas[:, y : y + cell, x : x + cell], digit_image
            )
            x += cell + (gaps[position] if position < len(gaps) else 0)

        return canvas

    def _random_affine(self, digit_image: torch.Tensor) -> torch.Tensor:
        """Warp one digit: rotation, scale, shear and a sub-cell translation.

        Shear is the addition that matters most. Rotation and scale alone leave the
        strokes locally axis-aligned, and with digits pinned to three evenly spaced cells
        the encoder could learn each cell's appearance rather than how to find a glyph.
        Slanting them breaks that without changing what the digit is.

        The composed matrix is scale -> shear -> rotation, applied to the sampling grid,
        so `theta` is the inverse mapping `affine_grid` expects.
        """
        angle = np.radians(random.uniform(-self.rotation_degrees, self.rotation_degrees))
        shear = np.tan(np.radians(random.uniform(-self.shear_degrees, self.shear_degrees)))
        low, high = self.scale_range
        scale = random.uniform(low, high)

        cos, sin = float(np.cos(angle)), float(np.sin(angle))
        rotation = np.array([[cos, -sin], [sin, cos]], dtype=np.float32)
        shear_matrix = np.array([[1.0, float(shear)], [0.0, 1.0]], dtype=np.float32)
        linear = rotation @ shear_matrix * scale

        # Translation in normalised grid units, bounded so the glyph stays inside its own
        # cell -- the cell is what gets pasted, so a larger shift would clip the digit.
        shift_x = random.uniform(-0.15, 0.15)
        shift_y = random.uniform(-0.15, 0.15)
        theta = torch.tensor(
            [
                [linear[0, 0], linear[0, 1], shift_x],
                [linear[1, 0], linear[1, 1], shift_y],
            ],
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
        min_digits: int = 1,
        digit_size: int = 28,
        min_digit_size: int = 10,
        min_gap: int = 2,
        augment_probability: float = 1.0,
        rotation_degrees: float = 15.0,
        shear_degrees: float = 12.0,
        scale_range: tuple[float, float] = (0.75, 1.15),
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
            max_digits: Most digits per composite, which also fixes the cell size.
            min_digits: Fewest digits per composite. 1 means the decoder has to decide
                where the number ends; setting it equal to `max_digits` restores the
                original fixed-length task.
            digit_size: Largest side length a digit is pasted at. The actual cell is
                drawn per composite between `min_digit_size` and whatever fits, so the
                model sees every scale a person can draw rather than one.
            min_digit_size: Smallest cell a digit is pasted at.
            min_gap: Smallest gap between digits, and from the canvas edge.
            augment_probability: Share of digits given a random affine warp.
            rotation_degrees: Rotation range, plus or minus.
            shear_degrees: Shear range, plus or minus.
            scale_range: Scale range as (low, high).
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
        self.min_digits = min_digits
        self.digit_size = digit_size
        self.min_digit_size = min_digit_size
        self.min_gap = min_gap
        self.augment_probability = augment_probability
        self.rotation_degrees = rotation_degrees
        self.shear_degrees = shear_degrees
        self.scale_range = scale_range
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
        """Bucket every MNIST training image by the digit it shows.

        `ToTensor` only, deliberately: the digits are pasted onto a canvas that
        `__getitem__` then normalises, so normalising here as well applied MNIST's
        statistics twice. That is not a cosmetic difference. The served path decodes a
        PNG in [0, 1] and normalises once, so a double-normalised training input put
        every stroke at roughly three times the amplitude the model would ever see
        through the API, and the background of a composite at MNIST's mean rather than
        at black. Scored on the same 3,000 held-out composites, the checkpoint that was
        shipped before this changed read 97.93% exact-sequence on the loader's tensors and
        **64.23% on the tensors serving produces from a faithful PNG of the same images**.
        The two paths score identically now. Pixels in, one normalisation, at the end.
        """
        transform = transforms.ToTensor()
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
        # Balanced by length, not by count. There are 9 one-digit numbers, 90 two-digit
        # and 900 three-digit, so drawing uniformly from `range(1000)` would make 90% of
        # training examples three digits long and the EOS decision would be learnable as
        # "almost always position 4". Each length contributes the same number of
        # examples instead.
        by_length = {
            length: [n for n in range(10**self.max_digits) if len(str(n)) == length]
            for length in range(self.min_digits, self.max_digits + 1)
        }
        widest = max(len(numbers) for numbers in by_length.values())
        all_numbers: list[int] = []
        for numbers in by_length.values():
            repeats = max(1, round(widest / len(numbers)))
            all_numbers.extend(numbers * repeats)
        random.shuffle(all_numbers)
        logger.info(
            "Number pool by digit count: %s",
            ", ".join(
                f"{length}d={len(numbers)}x{max(1, round(widest / len(numbers)))}"
                for length, numbers in by_length.items()
            ),
        )

        self.train_numbers = list(all_numbers)

        # Sampled per length, because sampling from the pooled list does not reach the
        # short numbers. There are only 9 distinct one-digit numbers, so a validation
        # split drawn from everything takes all 9 and leaves the test split with none --
        # which is exactly what happened: a run reported 2- and 3-digit accuracy and
        # silently measured nothing at all for one digit.
        per_length = max(1, min(40, min(len(numbers) for numbers in by_length.values())))
        self.val_numbers = []
        self.test_numbers = []
        for numbers in by_length.values():
            distinct = sorted(set(numbers))
            random.shuffle(distinct)
            # Short lengths have fewer distinct numbers than the two splits want, and the
            # task is reading digits off a canvas rather than generalising to unseen
            # numbers (see the class docstring), so the two splits may overlap for them.
            # They never share handwriting, which is what is actually held out.
            self.val_numbers.extend(distinct[:per_length])
            self.test_numbers.extend(
                distinct[per_length : 2 * per_length]
                if len(distinct) >= 2 * per_length
                else distinct[:per_length]
            )

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
        # Keyword arguments on purpose: this call was positional, so adding a parameter
        # in the middle of the signature silently shifted every later one by a slot.
        return MultiDigitDataset(
            digit_images=self.digit_images,
            numbers=numbers,
            image_size=self.image_size,
            max_digits=self.max_digits,
            digit_size=self.digit_size,
            min_digit_size=self.min_digit_size,
            min_gap=self.min_gap,
            augment_probability=self.augment_probability,
            rotation_degrees=self.rotation_degrees,
            shear_degrees=self.shear_degrees,
            scale_range=self.scale_range,
            return_digit_sequence=self.return_digit_sequence,
            add_start_eos=self.add_start_eos,
            start_token=self.start_token,
            eos_token=self.eos_token,
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
