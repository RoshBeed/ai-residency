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
    Balanced multi-digit MNIST dataset for 3-digit number recognition.
    Generates balanced data with equal representation of all 3-digit numbers (0-999).
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
                 seed: int = 42):
        """
        Initialize multi-digit MNIST dataset.
        
        Args:
            data_dir: Directory to store/load MNIST data
            image_size: Size of generated multi-digit images
            max_digits: Maximum number of digits (default 3)
            train_split: Fraction of data to use for training
            samples_per_class: Number of samples per 3-digit number class
            batch_size: Batch size for data loaders
            num_workers: Number of workers for data loading
            download: Whether to download MNIST if not present
            seed: Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.max_digits = max_digits
        self.train_split = train_split
        self.samples_per_class = samples_per_class
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        
        # Set random seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Load base MNIST dataset
        self._load_base_mnist(download)
        
        # Generate multi-digit data
        self._generate_multi_digit_data()
        
        # Create data loaders
        self._create_data_loaders()
        
        # Print dataset statistics
        self._print_statistics()
    
    def _load_base_mnist(self, download: bool):
        """Load base MNIST dataset for digit extraction."""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Load full MNIST dataset
        full_mnist = datasets.MNIST(
            self.data_dir, 
            train=True, 
            download=download, 
            transform=transform
        )
        
        # Organize digits by class
        self.digit_images = {i: [] for i in range(10)}
        
        for image, label in full_mnist:
            # Handle both tensor and int labels
            if isinstance(label, torch.Tensor):
                label_value = label.item()
            else:
                label_value = label
            self.digit_images[label_value].append(image)
        
        print(f"Base MNIST loaded:")
        for digit, images in self.digit_images.items():
            print(f"  Digit {digit}: {len(images)} samples")
    
    def _generate_multi_digit_data(self):
        """Generate balanced multi-digit data."""
        # Generate all possible 3-digit numbers (0-999)
        all_numbers = list(range(1000))
        
        # Create balanced dataset
        train_numbers = []
        val_numbers = []
        
        # For each number, create multiple samples
        for number in all_numbers:
            # Create samples_per_class samples for each number
            for _ in range(self.samples_per_class):
                # Randomly assign to train or validation
                if random.random() < self.train_split:
                    train_numbers.append(number)
                else:
                    val_numbers.append(number)
        
        # Shuffle the lists
        random.shuffle(train_numbers)
        random.shuffle(val_numbers)
        
        # Generate images and labels
        self.train_images, self.train_labels = self._create_multi_digit_samples(train_numbers)
        self.val_images, self.val_labels = self._create_multi_digit_samples(val_numbers)
        
        # Create test set (smaller, balanced sample)
        test_numbers = random.sample(all_numbers, min(1000, len(all_numbers)))
        self.test_images, self.test_labels = self._create_multi_digit_samples(test_numbers)
        
        print(f"Multi-digit data generated:")
        print(f"  Training samples: {len(self.train_images)}")
        print(f"  Validation samples: {len(self.val_images)}")
        print(f"  Test samples: {len(self.test_images)}")
    
    def _create_multi_digit_samples(self, numbers: List[int]) -> Tuple[List[torch.Tensor], List[int]]:
        """Create multi-digit image samples from a list of numbers."""
        images = []
        labels = []
        
        for number in numbers:
            # Convert number to digits
            digits = [int(d) for d in str(number).zfill(self.max_digits)]
            
            # Create multi-digit image
            image = self._create_multi_digit_image(digits)
            
            images.append(image)
            labels.append(number)
        
        return images, labels
    
    def _create_multi_digit_image(self, digits: List[int]) -> torch.Tensor:
        """Create a multi-digit image from a list of digits."""
        # Calculate digit size and spacing
        digit_size = self.image_size // 3  # 3 digits horizontally
        spacing = (self.image_size - 3 * digit_size) // 4  # Equal spacing
        
        # Create blank canvas
        canvas = torch.zeros(1, self.image_size, self.image_size)
        
        # Place each digit
        for i, digit in enumerate(digits):
            # Get random digit image
            digit_images = self.digit_images[digit]
            digit_image = random.choice(digit_images)
            
            # Resize digit to fit
            digit_image = torch.nn.functional.interpolate(
                digit_image.unsqueeze(0), 
                size=(digit_size, digit_size), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
            
            # Calculate position
            x_start = spacing + i * (digit_size + spacing)
            y_start = (self.image_size - digit_size) // 2
            
            # Place digit on canvas
            canvas[:, y_start:y_start+digit_size, x_start:x_start+digit_size] = digit_image
        
        return canvas
    
    def _create_data_loaders(self):
        """Create data loaders for train, validation, and test sets."""
        # Create custom datasets
        self.train_dataset = MultiDigitDataset(self.train_images, self.train_labels)
        self.val_dataset = MultiDigitDataset(self.val_images, self.val_labels)
        self.test_dataset = MultiDigitDataset(self.test_images, self.test_labels)
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def _print_statistics(self):
        """Print dataset statistics."""
        print("\nMulti-digit Dataset Statistics:")
        
        # Class distribution
        train_dist = self._get_class_distribution(self.train_labels)
        val_dist = self._get_class_distribution(self.val_labels)
        test_dist = self._get_class_distribution(self.test_labels)
        
        print(f"Training set: {len(self.train_labels)} samples")
        print(f"Validation set: {len(self.val_labels)} samples")
        print(f"Test set: {len(self.test_labels)} samples")
        
        # Check balance
        train_counts = list(train_dist.values())
        val_counts = list(val_dist.values())
        
        print(f"Training balance: min={min(train_counts)}, max={max(train_counts)}")
        print(f"Validation balance: min={min(val_counts)}, max={max(val_counts)}")
    
    def _get_class_distribution(self, labels: List[int]) -> dict:
        """Get class distribution from labels."""
        distribution = defaultdict(int)
        for label in labels:
            distribution[label] += 1
        return dict(distribution)
    
    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get train, validation, and test data loaders."""
        return self.train_loader, self.val_loader, self.test_loader
    
    def get_dataset_sizes(self) -> Tuple[int, int, int]:
        """Get sizes of train, validation, and test datasets."""
        return len(self.train_labels), len(self.val_labels), len(self.test_labels)
    
    def visualize_samples(self, num_samples: int = 5, split: str = 'train'):
        """Visualize sample multi-digit images."""
        if split == 'train':
            images = self.train_images[:num_samples]
            labels = self.train_labels[:num_samples]
        elif split == 'val':
            images = self.val_images[:num_samples]
            labels = self.val_labels[:num_samples]
        elif split == 'test':
            images = self.test_images[:num_samples]
            labels = self.test_labels[:num_samples]
        else:
            raise ValueError(f"Invalid split: {split}")
        
        fig, axes = plt.subplots(1, num_samples, figsize=(2*num_samples, 2))
        if num_samples == 1:
            axes = [axes]
        
        for i, (image, label) in enumerate(zip(images, labels)):
            axes[i].imshow(image.squeeze(), cmap='gray')
            axes[i].set_title(f'Label: {label}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()


class MultiDigitDataset(Dataset):
    """Custom dataset for multi-digit images."""
    
    def __init__(self, images: List[torch.Tensor], labels: List[int]):
        self.images = images
        self.labels = labels
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx] 