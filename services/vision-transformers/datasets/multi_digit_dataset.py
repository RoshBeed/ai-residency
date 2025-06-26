import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
import os
import random
from typing import Tuple, List, Optional
from collections import defaultdict
import matplotlib.pyplot as plt

class MultiDigitMNISTDataset:
    """
    Balanced multi-digit MNIST dataset for N-digit number recognition.
    Generates balanced data with equal representation of all N-digit numbers (0-999 for 3 digits).
    Now generates images on-the-fly for memory efficiency and realism.
    """
    def __init__(self, 
                 data_dir: str = "data",
                 image_size: int = 80,
                 max_digits: int = 3,
                 train_split: float = 0.8,
                 samples_per_class: int = 10,
                 batch_size: int = 64,
                 num_workers: int = 4,
                 download: bool = True,
                 seed: int = 42,
                 return_digit_sequence: bool = False,
                 add_start_eos: bool = False):
        self.data_dir = data_dir
        self.image_size = image_size
        self.max_digits = max_digits
        self.train_split = train_split
        self.samples_per_class = samples_per_class
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.return_digit_sequence = return_digit_sequence
        self.add_start_eos = add_start_eos
        self.start_token = 10
        self.eos_token = 11
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        os.makedirs(data_dir, exist_ok=True)
        self._load_base_mnist(download)
        self._split_indices()
        self._create_data_loaders()
        self._print_statistics()

    def _load_base_mnist(self, download: bool):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        full_mnist = datasets.MNIST(
            self.data_dir, 
            train=True, 
            download=download, 
            transform=transform
        )
        self.digit_images = {i: [] for i in range(10)}
        for image, label in full_mnist:
            label_value = label.item() if isinstance(label, torch.Tensor) else label
            self.digit_images[label_value].append(image)
        print(f"Base MNIST loaded:")
        for digit, images in self.digit_images.items():
            print(f"  Digit {digit}: {len(images)} samples")

    def _split_indices(self):
        all_numbers = list(range(10**self.max_digits))  # 0-999 for 3 digits
        # Shuffle for reproducibility
        random.shuffle(all_numbers)
        
        # Ensure all numbers are in training set for complete coverage
        self.train_numbers = all_numbers.copy()  # All 1000 numbers
        
        # Create smaller validation and test sets from the same numbers
        # This allows testing generalization on unseen samples of the same numbers
        val_size = min(100, len(all_numbers) // 10)  # 10% or 100, whichever is smaller
        test_size = min(100, len(all_numbers) // 10)
        
        # Sample validation and test numbers from the full set
        self.val_numbers = random.sample(all_numbers, val_size)
        remaining_numbers = [n for n in all_numbers if n not in self.val_numbers]
        self.test_numbers = random.sample(remaining_numbers, test_size)
        
        # For each split, repeat numbers to get samples_per_class
        self.train_indices = self.train_numbers * self.samples_per_class
        self.val_indices = self.val_numbers * self.samples_per_class
        self.test_indices = self.test_numbers * self.samples_per_class
        
        random.shuffle(self.train_indices)
        random.shuffle(self.val_indices)
        random.shuffle(self.test_indices)

    def _create_data_loaders(self):
        self.train_dataset = MultiDigitDataset(self.digit_images, self.train_indices, self.image_size, self.max_digits, self.return_digit_sequence, self.add_start_eos)
        self.val_dataset = MultiDigitDataset(self.digit_images, self.val_indices, self.image_size, self.max_digits, self.return_digit_sequence, self.add_start_eos)
        self.test_dataset = MultiDigitDataset(self.digit_images, self.test_indices, self.image_size, self.max_digits, self.return_digit_sequence, self.add_start_eos)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def _print_statistics(self):
        print("\nMulti-digit Dataset Statistics:")
        print(f"Training set: {len(self.train_indices)} samples, {len(set(self.train_numbers))} unique numbers")
        print(f"Validation set: {len(self.val_indices)} samples, {len(set(self.val_numbers))} unique numbers")
        print(f"Test set: {len(self.test_indices)} samples, {len(set(self.test_numbers))} unique numbers")

    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        return self.train_loader, self.val_loader, self.test_loader

    def get_dataset_sizes(self) -> Tuple[int, int, int]:
        return len(self.train_indices), len(self.val_indices), len(self.test_indices)

    def visualize_samples(self, num_samples: int = 5, split: str = 'train'):
        if split == 'train':
            dataset = self.train_dataset
        elif split == 'val':
            dataset = self.val_dataset
        elif split == 'test':
            dataset = self.test_dataset
        else:
            raise ValueError(f"Invalid split: {split}")
        fig, axes = plt.subplots(1, num_samples, figsize=(2*num_samples, 2))
        if num_samples == 1:
            axes = [axes]
        for i in range(num_samples):
            image, label = dataset[i]
            axes[i].imshow(image.squeeze(), cmap='gray')
            axes[i].set_title(f'Label: {label}')
            axes[i].axis('off')
        plt.tight_layout()
        plt.show()

class MultiDigitDataset(Dataset):
    """Custom dataset for multi-digit images, generates images on-the-fly."""
    def __init__(self, digit_images, indices, image_size, max_digits, return_digit_sequence=False, add_start_eos=False):
        self.digit_images = digit_images
        self.indices = indices
        self.image_size = image_size
        self.max_digits = max_digits
        self.return_digit_sequence = return_digit_sequence
        self.add_start_eos = add_start_eos
        self.start_token = 10
        self.eos_token = 11
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        number = self.indices[idx]
        digits = [int(d) for d in str(number).zfill(self.max_digits)]
        image = self._create_multi_digit_image(digits)
        # Use consistent MNIST normalization instead of custom normalization
        # MNIST normalization: (x - 0.1307) / 0.3081
        image = (image - 0.1307) / 0.3081
        if self.return_digit_sequence:
            seq = digits.copy()
            if self.add_start_eos:
                seq = [self.start_token] + seq + [self.eos_token]
            return image, torch.tensor(seq, dtype=torch.long)
        else:
            return image, number
    def _create_multi_digit_image(self, digits: List[int]) -> torch.Tensor:
        digit_size = self.image_size // self.max_digits
        spacing = (self.image_size - self.max_digits * digit_size) // (self.max_digits + 1)
        canvas = torch.zeros(1, self.image_size, self.image_size)
        for i, digit in enumerate(digits):
            digit_images = self.digit_images[digit]
            digit_image = random.choice(digit_images)
            # Resize digit
            digit_image = torch.nn.functional.interpolate(
                digit_image.unsqueeze(0), 
                size=(digit_size, digit_size), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
            # Add random jitter
            max_jitter = spacing // 2
            x_jitter = random.randint(-max_jitter, max_jitter) if max_jitter > 0 else 0
            y_jitter = random.randint(-max_jitter, max_jitter) if max_jitter > 0 else 0
            x_start = spacing + i * (digit_size + spacing) + x_jitter
            y_start = (self.image_size - digit_size) // 2 + y_jitter
            # Clamp to canvas
            x_start = max(0, min(self.image_size - digit_size, x_start))
            y_start = max(0, min(self.image_size - digit_size, y_start))
            # Optionally apply random affine transform
            if random.random() < 0.3:
                angle = random.uniform(-10, 10)
                scale = random.uniform(0.9, 1.1)
                theta = torch.tensor([
                    [scale * np.cos(np.radians(angle)), -scale * np.sin(np.radians(angle)), 0],
                    [scale * np.sin(np.radians(angle)),  scale * np.cos(np.radians(angle)), 0]
                ], dtype=torch.float32)  # Ensure float32
                theta = theta.unsqueeze(0)  # (1, 2, 3)
                grid = torch.nn.functional.affine_grid(
                    theta,
                    digit_image.unsqueeze(0).size(), align_corners=False)
                digit_image = torch.nn.functional.grid_sample(
                    digit_image.unsqueeze(0), grid, align_corners=False).squeeze(0)
            # Place digit
            canvas[:, y_start:y_start+digit_size, x_start:x_start+digit_size] = digit_image
        return canvas 