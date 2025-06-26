import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
import os
from typing import Tuple, Optional

class MNISTDataset:
    """
    Clean MNIST dataset wrapper for single digit recognition.
    Uses torchvision's MNIST dataset with proper train/val/test splits.
    """
    
    def __init__(self, 
                 data_dir: str = "data",
                 image_size: int = 28,
                 train_split: float = 0.8,
                 batch_size: int = 64,
                 num_workers: int = 4,
                 download: bool = True):
        """
        Initialize MNIST dataset.
        
        Args:
            data_dir: Directory to store/load MNIST data
            image_size: Size to resize images to (default 28 for MNIST)
            train_split: Fraction of training data to use for validation
            batch_size: Batch size for data loaders
            num_workers: Number of workers for data loading
            download: Whether to download MNIST if not present
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.train_split = train_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Define transformations
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
        ])
        
        # Load datasets
        self._load_datasets(download)
        
        # Create data loaders
        self._create_data_loaders()
    
    def _load_datasets(self, download: bool):
        """Load MNIST datasets from torchvision."""
        # Load full training dataset
        full_train_dataset = datasets.MNIST(
            self.data_dir, 
            train=True, 
            download=download, 
            transform=self.transform
        )
        
        # Load test dataset
        self.test_dataset = datasets.MNIST(
            self.data_dir, 
            train=False, 
            download=download, 
            transform=self.transform
        )
        
        # Split training data into train and validation
        train_size = int(self.train_split * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            full_train_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # For reproducibility
        )
        
        print(f"MNIST Dataset loaded:")
        print(f"  Training samples: {len(self.train_dataset)}")
        print(f"  Validation samples: {len(self.val_dataset)}")
        print(f"  Test samples: {len(self.test_dataset)}")
    
    def _create_data_loaders(self):
        """Create data loaders for train, validation, and test sets."""
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
    
    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get train, validation, and test data loaders."""
        return self.train_loader, self.val_loader, self.test_loader
    
    def get_dataset_sizes(self) -> Tuple[int, int, int]:
        """Get sizes of train, validation, and test datasets."""
        return len(self.train_dataset), len(self.val_dataset), len(self.test_dataset)
    
    def get_class_distribution(self, split: str = 'train') -> dict:
        """
        Get class distribution for a given split.
        
        Args:
            split: 'train', 'val', or 'test'
        
        Returns:
            Dictionary with class counts
        """
        if split == 'train':
            dataset = self.train_dataset
        elif split == 'val':
            dataset = self.val_dataset
        elif split == 'test':
            dataset = self.test_dataset
        else:
            raise ValueError(f"Invalid split: {split}")
        
        class_counts = {i: 0 for i in range(10)}
        
        for _, label in dataset:
            class_counts[label] += 1
        
        return class_counts
    
    def print_class_distribution(self, split: str = 'train'):
        """Print class distribution for a given split."""
        distribution = self.get_class_distribution(split)
        print(f"\nClass distribution for {split} set:")
        for digit, count in distribution.items():
            print(f"  Digit {digit}: {count} samples")
        
        total = sum(distribution.values())
        print(f"  Total: {total} samples") 