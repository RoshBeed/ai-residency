# Datasets

This directory contains dataset implementations for training Vision Transformer models.

## Structure

```
datasets/
├── __init__.py              # Package initialization and exports
├── mnist_dataset.py         # Single digit MNIST dataset
├── multi_digit_dataset.py   # Multi-digit dataset for seq2seq training
└── README.md                # This file
```

## Dataset Implementations

### MNISTDataset (`mnist_dataset.py`)

A PyTorch dataset for single-digit MNIST images with preprocessing and augmentation.

**Features:**
- Standard MNIST digit classification
- Image preprocessing and normalization
- Data augmentation (rotation, brightness, etc.)
- Configurable image size and transformations
- Train/validation split support

**Usage:**
```python
from datasets import MNISTDataset

# Create dataset
dataset = MNISTDataset(
    data_dir="data",
    image_size=84,
    train=True,
    max_samples=10000
)

# Use with DataLoader
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### MultiDigitMNISTDataset (`multi_digit_dataset.py`)

A PyTorch dataset for multi-digit sequences with seq2seq training support.

**Features:**
- Multi-digit number generation (0-999)
- Sequence-to-sequence training format
- Configurable maximum number of digits
- Automatic sequence generation and padding
- Support for variable-length sequences
- Teacher forcing compatible

**Usage:**
```python
from datasets import MultiDigitMNISTDataset

# Create dataset
dataset = MultiDigitMNISTDataset(
    data_dir="data",
    image_size=84,
    max_digits=3,
    train=True,
    max_samples=100000
)

# Use with DataLoader
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

## Data Format

### Single Digit Dataset
- **Images**: Grayscale images of single digits (0-9)
- **Labels**: Integer labels (0-9)
- **Output**: `(image_tensor, label)`

### Multi-Digit Dataset
- **Images**: Grayscale images of multi-digit numbers
- **Labels**: Digit sequences with START/EOS tokens
- **Output**: `(image_tensor, sequence_tensor)`

## Configuration

Both datasets support configuration through:
- `data_dir`: Directory containing MNIST data
- `image_size`: Target image size for preprocessing
- `train`: Whether to use training or validation split
- `max_samples`: Maximum number of samples to load
- `transform`: Custom image transformations

## Data Preprocessing

### Image Processing
- Resize to target image size
- Convert to grayscale
- Normalize pixel values to [-1, 1]
- Add batch and channel dimensions

### Sequence Processing (Multi-digit)
- Convert numbers to digit sequences
- Add START token (10) at beginning
- Add EOS token (11) at end
- Pad sequences to maximum length
- Convert to tensor format

## Integration with Models

These datasets are designed to work seamlessly with:
- **VisionTransformer**: Uses MNISTDataset for classification
- **Seq2SeqVisionTransformer**: Uses MultiDigitMNISTDataset for sequence generation

## Data Sources

The datasets expect MNIST data in the following structure:
```
data/
├── train/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

## Performance

- **Memory Efficient**: Lazy loading of images
- **Fast Preprocessing**: Optimized transformations
- **Multi-processing**: Compatible with DataLoader workers
- **Caching**: Optional caching for faster training 