# Seq2Seq Vision Transformer for Multi-Digit Recognition

A Vision Transformer (ViT) sequence-to-sequence model for recognizing multi-digit handwritten numbers from MNIST images.

## Overview

This project implements a seq2seq Vision Transformer that can:
- Process handwritten digit images (80x84 pixels)
- Generate sequences of digits (0-9) 
- Handle multi-digit numbers (up to 3 digits)
- Use autoregressive generation for inference

## Architecture

- **Encoder**: Vision Transformer with patch embedding and self-attention
- **Decoder**: Transformer decoder with cross-attention to encoder
- **Vocabulary**: 12 tokens (0-9 digits + START token + EOS token)
- **Model Size**: ~1.9M parameters

## Project Structure

```
├── config.py                 # Configuration settings
├── transformer/              # Core transformer components
│   ├── seq2seq_vit.py       # Main seq2seq ViT model
│   ├── attention.py         # Multi-head attention
│   ├── encoder.py           # Transformer encoder
│   ├── decoder.py           # Transformer decoder
│   ├── patch_embedding.py   # Image patch embedding
│   └── positional.py        # Positional encoding
├── s2s_vit/                 # Training scripts
│   ├── train_seq2seq_vit.py # Main training script
│   └── train_single_digit.py # Single digit training
├── datasets/                # Dataset handling
│   └── multi_digit_dataset.py # Multi-digit MNIST dataset
├── vit_encoder/             # Alternative ViT encoder
└── streamlit/               # Web interface
```

## Features

- **Multi-digit Recognition**: Handles 1-3 digit numbers
- **Autoregressive Generation**: Proper seq2seq inference
- **MPS Support**: Optimized for Apple Silicon GPUs
- **Configurable**: Easy to adjust model size and training parameters
- **Modular Design**: Clean, reusable components

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install torch torchvision tqdm matplotlib streamlit
   ```

2. **Train the Model**:
   ```bash
   python -m s2s_vit.train_seq2seq_vit
   ```

3. **Run Web Interface** (optional):
   ```bash
   streamlit run streamlit/app.py
   ```

## Configuration

Key settings in `config.py`:

```python
MODEL_CONFIG = {
    "img_size": 84,           # Input image size
    "patch_size": 16,         # Patch size for embedding
    "d_model": 128,           # Model dimension
    "num_encoder_layers": 4,  # Number of encoder layers
    "num_decoder_layers": 4,  # Number of decoder layers
    "heads": 2,               # Number of attention heads
    "vocab_size": 12,         # Vocabulary size (0-9 + START + EOS)
}

TRAINING_CONFIG = {
    "epochs": 10,             # Training epochs
    "batch_size": 128,        # Batch size
    "learning_rate": 0.004,   # Learning rate
    "device": "mps",          # Device (mps/cpu/cuda)
    "max_digits": 3,          # Maximum digits per number
}
```

## Training Notes

- **Teacher Forcing**: Training uses ground truth sequences
- **Autoregressive Validation**: Validation uses generation for realistic accuracy
- **Expected Pattern**: Training loss drops quickly, validation accuracy improves gradually
- **MPS Optimization**: Uses Apple Silicon GPU for faster training

## Model Performance

- **Training**: Uses teacher forcing for stable learning
- **Validation**: Uses autoregressive generation for realistic evaluation
- **Expected Accuracy**: Improves from ~0% to 50-80% over 10-20 epochs
- **Memory Usage**: ~1.9M parameters, suitable for CPU/MPS training

## License

MIT License - feel free to use and modify for your projects! 