# High-Level Models

This directory contains high-level models that build on top of the transformer components in the `transformer/` directory.

## Structure

```
models/
├── __init__.py                    # Package initialization and exports
├── classifier/                    # Vision Transformer classification models
│   ├── __init__.py
│   ├── classifier.py              # Vision Transformer for classification
│   ├── trainer.py                 # Training script for classifier
│   └── Dockerfile                 # Dockerfile for classifier training
├── seq2seq/                       # Seq2Seq Vision Transformer models
│   ├── __init__.py
│   ├── seq2seq.py                 # Seq2Seq Vision Transformer
│   ├── multi_digit_trainer.py     # Training script for seq2seq
│   └── Dockerfile                 # Dockerfile for seq2seq training
└── README.md                      # This file
```

## Models

### VisionTransformer (`classifier/classifier.py`)
A standard Vision Transformer for image classification tasks. This model uses the transformer components from the `transformer/` package to build a complete classification model.

**Usage:**
```python
from models import VisionTransformer

config = {
    "img_size": 28,
    "patch_size": 4,
    "in_channels": 1,
    "num_classes": 10,
    "d_model": 128,
    "num_encoder_layers": 6,
    "heads": 8,
    "head_dim": 16,
    "ff_hidden_dim": 512,
    "dropout": 0.1
}

model = VisionTransformer(config)
```

### Seq2SeqVisionTransformer (`seq2seq/seq2seq.py`)
A sequence-to-sequence Vision Transformer for tasks like multi-digit recognition. This model combines an encoder (for images) with a decoder (for sequences).

**Usage:**
```python
from models import Seq2SeqVisionTransformer

config = {
    "img_size": 84,
    "patch_size": 4,
    "in_channels": 1,
    "d_model": 128,
    "num_encoder_layers": 6,
    "num_decoder_layers": 6,
    "heads": 8,
    "head_dim": 16,
    "ff_hidden_dim": 512,
    "dropout": 0.1,
    "vocab_size": 12,
    "max_seq_length": 5
}

model = Seq2SeqVisionTransformer(config)
```

## Training Scripts

### classifier/trainer.py
A comprehensive training script for the Vision Transformer classifier with features like:
- Configurable model architecture
- Training/validation loops
- Checkpoint saving and loading
- Learning rate scheduling
- Early stopping
- Training history visualization

**Usage:**
```bash
# Basic training
python models/classifier/trainer.py

# Custom parameters
python models/classifier/trainer.py --epochs 50 --batch_size 32 --learning_rate 5e-5
```

### seq2seq/multi_digit_trainer.py
A training script for the Seq2Seq Vision Transformer model with features like:
- Multi-digit sequence training
- Teacher forcing
- Beam search inference
- Sequence generation utilities

**Usage:**
```bash
# Basic training
python models/seq2seq/multi_digit_trainer.py

# Custom parameters
python models/seq2seq/multi_digit_trainer.py --epochs 50 --batch_size 32 --learning_rate 5e-5
```

## Dependencies

These models depend on the transformer components in the `transformer/` directory:
- `transformer.attention` - Multi-head attention mechanism
- `transformer.encoder` - Transformer encoder layers
- `transformer.decoder` - Transformer decoder layers
- `transformer.patch_embedding` - Image patch embedding
- `transformer.positional` - Positional encoding
- `transformer.feedforward` - Feed-forward networks

## Import Examples

```python
# Import specific models
from models.classifier.classifier import VisionTransformer
from models.seq2seq.seq2seq import Seq2SeqVisionTransformer

# Import from package
from models import VisionTransformer, Seq2SeqVisionTransformer

# Import training scripts
from models.classifier.trainer import Trainer
from models.seq2seq.multi_digit_trainer import train_seq2seq_model, evaluate_seq2seq_model
```

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Input Image   │    │   Input Image   │    │   Input Image   │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          ▼                      ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Patch Embed    │    │  Patch Embed    │    │  Patch Embed    │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          ▼                      ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Transformer     │    │ Transformer     │    │ Transformer     │
│ Encoder         │    │ Encoder         │    │ Encoder         │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          ▼                      ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Classification  │    │ Transformer     │    │ [CLS] Token     │
│ Head            │    │ Decoder         │    │ Classification  │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          ▼                      ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Class Logits  │    │  Sequence       │    │   Class Logits  │
│   (ViT)         │    │  Output         │    │   (Encoder)     │
│                 │    │  (Seq2Seq)      │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Configuration

All models use configuration dictionaries that specify:
- Image and patch sizes
- Model dimensions and architecture
- Number of layers and attention heads
- Dropout rates and regularization
- Vocabulary size (for seq2seq models)
- Training parameters

See individual model files for detailed configuration options. 