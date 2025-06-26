import mlx.core as mx
import mlx.nn as nn
import numpy as np
from typing import Tuple, List
import os
import urllib.request
import gzip
import pickle
from pathlib import Path

class MultiDigitMNISTDataset:
    """
    Dataset that creates multi-digit images from MNIST for training ViT models.
    Generates images with 1-3 digits that can be classified as numbers 0-999.
    """
    
    def __init__(self, 
                 data_dir: str = "data",
                 image_size: int = 80,
                 max_digits: int = 3,
                 train_split: float = 0.8):
        """
        Initialize the multi-digit MNIST dataset.
        
        Args:
            data_dir: Directory to store/load MNIST data
            image_size: Size of output images (should match ViT patch size)
            max_digits: Maximum number of digits per image (1-3)
            train_split: Fraction of data to use for training
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.max_digits = max_digits
        self.train_split = train_split
        
        # Create data directory
        self.data_dir.mkdir(exist_ok=True)
        
        # Download and load MNIST data
        self.mnist_data = self._download_mnist()
        
        # Generate multi-digit images
        self.train_images, self.train_labels = self._generate_multi_digit_data(split='train')
        self.val_images, self.val_labels = self._generate_multi_digit_data(split='val')
        
        print(f"Generated {len(self.train_images)} training images")
        print(f"Generated {len(self.val_images)} validation images")
        print(f"Label range: {self.train_labels.min()} - {self.train_labels.max()}")
    
    def _download_mnist(self) -> dict:
        """Download and load MNIST data."""
        # Updated URLs that should work
        mnist_urls = {
            'train_images': 'https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz',
            'train_labels': 'https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz',
            'test_images': 'https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz',
            'test_labels': 'https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz'
        }
        
        mnist_data = {}
        
        for name, url in mnist_urls.items():
            file_path = self.data_dir / f"{name}.gz"
            
            # Download if not exists
            if not file_path.exists():
                print(f"Downloading {name}...")
                try:
                    urllib.request.urlretrieve(url, file_path)
                    print(f"Successfully downloaded {name}")
                except Exception as e:
                    print(f"Failed to download {name} from {url}: {e}")
                    print("Trying alternative source...")
                    # Try alternative URL
                    alt_url = url.replace('storage.googleapis.com/cvdf-datasets', 'ossci-datasets.s3.amazonaws.com')
                    try:
                        urllib.request.urlretrieve(alt_url, file_path)
                        print(f"Successfully downloaded {name} from alternative source")
                    except Exception as e2:
                        print(f"Failed to download {name} from alternative source: {e2}")
                        raise RuntimeError(f"Could not download MNIST data for {name}")
            
            # Load data
            try:
                with gzip.open(file_path, 'rb') as f:
                    if 'images' in name:
                        # Images: [N, 28, 28]
                        data = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
                    else:
                        # Labels: [N]
                        data = np.frombuffer(f.read(), np.uint8, offset=8)
                    
                    mnist_data[name] = data
                    print(f"Loaded {name}: {data.shape}")
            except Exception as e:
                print(f"Failed to load {name}: {e}")
                raise
        
        return mnist_data
    
    def _create_multi_digit_image(self, digits: List[int], positions: List[Tuple[int, int]]) -> np.ndarray:
        """
        Create a multi-digit image by placing MNIST digits at specified positions.
        
        Args:
            digits: List of digit values (0-9)
            positions: List of (x, y) positions for each digit
            
        Returns:
            Multi-digit image of size image_size x image_size
        """
        # Create blank canvas
        canvas = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        
        for digit, (x, y) in zip(digits, positions):
            # Get random MNIST image for this digit
            digit_indices = np.where(self.mnist_data['train_labels'] == digit)[0]
            if len(digit_indices) == 0:
                continue
            
            # Pick random instance of this digit
            idx = np.random.choice(digit_indices)
            digit_img = self.mnist_data['train_images'][idx].astype(np.float32) / 255.0
            
            # Place digit on canvas
            h, w = digit_img.shape
            y_end = min(y + h, self.image_size)
            x_end = min(x + w, self.image_size)
            
            if y < self.image_size and x < self.image_size:
                canvas[y:y_end, x:x_end] = digit_img[:y_end-y, :x_end-x]
        
        return canvas
    
    def _generate_multi_digit_data(self, split: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate multi-digit images and labels for training/validation.
        
        Args:
            split: 'train' or 'val'
            
        Returns:
            Tuple of (images, labels)
        """
        # Determine number of samples based on split - INCREASED FOR BETTER PERFORMANCE
        if split == 'train':
            n_samples = int(15000 * self.train_split)  # Increased from 5000 to 15000
        else:
            n_samples = int(15000 * (1 - self.train_split))  # Increased from 5000 to 15000
        
        images = []
        labels = []
        
        for i in range(n_samples):
            # Randomly choose number of digits (1-3)
            num_digits = np.random.randint(1, self.max_digits + 1)
            
            # Generate random digits
            digits = np.random.randint(0, 10, num_digits)
            
            # Calculate label (e.g., [1, 2, 3] -> 123)
            label = int(''.join(map(str, digits)))
            
            # Generate positions for digits
            positions = []
            for j in range(num_digits):
                # Position digits horizontally with some spacing, centered for 80x80 image
                # For 80x80 image, center around 40, with spacing of ~20 pixels
                x = 15 + j * 20 + np.random.randint(-3, 4)  # Adjusted for 80x80
                y = 15 + np.random.randint(-3, 4)  # Adjusted for 80x80
                positions.append((x, y))
            
            # Create multi-digit image
            image = self._create_multi_digit_image(digits, positions)
            
            images.append(image)
            labels.append(label)
        
        return np.array(images), np.array(labels)
    
    def get_batch(self, batch_size: int, split: str = 'train') -> Tuple[mx.array, mx.array]:
        """
        Get a batch of data.
        
        Args:
            batch_size: Size of batch
            split: 'train' or 'val'
            
        Returns:
            Tuple of (images, labels) as MLX arrays
        """
        if split == 'train':
            images = self.train_images
            labels = self.train_labels
        else:
            images = self.val_images
            labels = self.val_labels
        
        # Random sampling
        indices = np.random.choice(len(images), batch_size, replace=False)
        batch_images = images[indices]
        batch_labels = labels[indices]
        
        # Convert to MLX arrays
        batch_images = mx.array(batch_images)
        batch_labels = mx.array(batch_labels)
        
        return batch_images, batch_labels
    
    def get_all_data(self, split: str = 'train') -> Tuple[mx.array, mx.array]:
        """
        Get all data for a split.
        
        Args:
            split: 'train' or 'val'
            
        Returns:
            Tuple of (images, labels) as MLX arrays
        """
        if split == 'train':
            images = self.train_images
            labels = self.train_labels
        else:
            images = self.val_images
            labels = self.val_labels
        
        return mx.array(images), mx.array(labels)

def create_sample_visualization(dataset: MultiDigitMNISTDataset, n_samples: int = 5):
    """
    Create a visualization of sample multi-digit images.
    
    Args:
        dataset: MultiDigitMNISTDataset instance
        n_samples: Number of samples to visualize
    """
    import matplotlib.pyplot as plt
    
    # Get sample images
    images, labels = dataset.get_batch(n_samples, 'train')
    images = np.array(images)
    labels = np.array(labels)
    
    # Create subplot
    fig, axes = plt.subplots(1, n_samples, figsize=(3*n_samples, 3))
    if n_samples == 1:
        axes = [axes]
    
    for i, (image, label) in enumerate(zip(images, labels)):
        axes[i].imshow(image, cmap='gray')
        axes[i].set_title(f'Label: {label}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_multi_digit_images.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved sample visualization to 'sample_multi_digit_images.png'")

# Removed the main block to prevent execution during Docker build 