# Transformer Components

This directory contains the low-level transformer components that are used by the high-level models in the `models/` directory.

## Structure

```
transformer/
├── __init__.py              # Package initialization and exports
├── attention.py             # Multi-head attention mechanism
├── encoder.py               # Transformer encoder layer
├── decoder.py               # Transformer decoder layer
├── feedforward.py           # Feed-forward network
├── patch_embedding.py       # Image patch embedding
└── positional.py            # Positional encoding
```

## Components

### MultiHeadAttention (`attention.py`)
Multi-head self-attention mechanism with configurable heads and dimensions.

**Features:**
- Configurable number of attention heads
- Scaled dot-product attention
- Optional dropout for regularization
- Support for both self-attention and cross-attention

### Encoder (`encoder.py`)
Transformer encoder layer with self-attention and feed-forward networks.

**Features:**
- Multi-head self-attention
- Feed-forward network
- Layer normalization
- Residual connections
- Configurable dropout

### Decoder (`decoder.py`)
Transformer decoder layer with self-attention, cross-attention, and feed-forward networks.

**Features:**
- Multi-head self-attention (masked for autoregressive generation)
- Multi-head cross-attention to encoder
- Feed-forward network
- Layer normalization
- Residual connections

### PatchEmbedding (`patch_embedding.py`)
Converts input images into patch embeddings using convolutional layers.

**Features:**
- Convolutional patch extraction
- Linear projection to embedding dimension
- Configurable patch size and embedding dimension

### PositionalEncoding (`positional.py`)
Adds positional information to embeddings using learnable positional encodings.

**Features:**
- Learnable positional embeddings
- Configurable sequence length
- Dropout for regularization

### FeedForward (`feedforward.py`)
Standard feed-forward network with linear layers and activation functions.

**Features:**
- Two linear layers with activation
- Configurable hidden dimension
- Dropout for regularization

## Import Examples

```python
# Import individual components
from transformer.attention import MultiHeadAttention
from transformer.encoder import Encoder
from transformer.decoder import Decoder
from transformer.patch_embedding import PatchEmbedding
from transformer.positional import PositionalEncoding
from transformer.feedforward import FeedForward

# Import from package
from transformer import MultiHeadAttention, Encoder, Decoder, PatchEmbedding, PositionalEncoding, FeedForward
```

## Usage in High-Level Models

These components are used by the high-level models in the `models/` directory:

- **VisionTransformer** (`models/classifier/classifier.py`): Uses Encoder, PatchEmbedding, and PositionalEncoding
- **Seq2SeqVisionTransformer** (`models/seq2seq/seq2seq.py`): Uses Encoder, Decoder, PatchEmbedding, and PositionalEncoding

## Configuration

Each component accepts configuration parameters:

```python
# Example attention configuration
attention_config = {
    "d_model": 256,
    "heads": 8,
    "head_dim": 32,
    "dropout": 0.1
}

# Example encoder configuration
encoder_config = {
    "d_model": 256,
    "heads": 8,
    "head_dim": 32,
    "ff_hidden_dim": 1024,
    "dropout": 0.1
}
```

See individual component files for detailed configuration options. 