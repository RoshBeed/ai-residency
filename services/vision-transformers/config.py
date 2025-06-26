"""
Centralized configuration for the Vision Transformer model.
All components (training, serving, evaluation) should use this config.
"""

import os
import platform
import sys

# Vocabulary constants for better maintainability
VOCAB_CONSTANTS = {
    "DIGITS": list(range(10)),  # 0-9 digits
    "START_TOKEN": 10,          # START token for seq2seq
    "EOS_TOKEN": 11,            # EOS token for seq2seq
    "VOCAB_SIZE": 12,           # Total vocabulary size
}

# Import device utilities
try:
    from utils.device_utils import detect_device, get_device_config, print_device_info
except ImportError:
    # Fallback if utils not available
    def detect_device():
        """Fallback device detection."""
        if platform.system() == "Darwin":
            return "mps"
        return "cpu"
    
    def get_device_config(device):
        """Fallback device config."""
        return {"batch_size": 32, "learning_rate": 0.0001, "mixed_precision": False}
    
    def print_device_info(device):
        """Fallback device info."""
        print(f"Using device: {device}")

# Detect device automatically
DETECTED_DEVICE = detect_device()

# Print device information
print_device_info(DETECTED_DEVICE)

# Model Architecture Configuration
MODEL_CONFIG = {
    # Image and patch settings
    "img_size": 84,
    "patch_size": 16,
    "in_channels": 1,
    
    # Model dimensions - scaled up for better performance
    "d_model": 256,  # Increased from 128 for better accuracy
    "num_encoder_layers": 6,  # Increased from 4 for better accuracy
    "num_decoder_layers": 6,  # Increased from 4 for better accuracy
    "heads": 8,  # Increased from 2 for better accuracy
    "head_dim": 32,  # Keep same for compatibility
    "ff_hidden_dim": 1024,  # Increased from 512 for better accuracy
    
    # Seq2Seq settings - using constants
    "vocab_size": VOCAB_CONSTANTS["VOCAB_SIZE"],
    "start_token": VOCAB_CONSTANTS["START_TOKEN"],
    "eos_token": VOCAB_CONSTANTS["EOS_TOKEN"],
    
    # Training settings - increased regularization
    "dropout": 0.1,
}

# Get device-specific configuration
DEVICE_CONFIG = get_device_config(DETECTED_DEVICE)

# Training Configuration - optimized for speed
TRAINING_CONFIG = {
    "epochs": 15,  # Increased for better convergence with larger model
    "batch_size": DEVICE_CONFIG["batch_size"],  # Device-specific batch size
    "learning_rate": DEVICE_CONFIG["learning_rate"],  # Device-specific learning rate
    "weight_decay": 0.01,
    "train_split": 0.8,
    "device": DETECTED_DEVICE,  # Automatically detected device
    "max_digits": 3,
    "max_samples": 100000,  # Increased significantly to match paper ratios
    
    # Modern training techniques
    "warmup_steps": 4000,  # Learning rate warmup
    "label_smoothing": 0.1,  # Label smoothing for better generalization
    "gradient_clip_norm": 1.0,  # Gradient clipping
    "scheduler_patience": 3,  # Patience for learning rate scheduling
    "scheduler_factor": 0.5,  # Factor for learning rate reduction
    
    # Gradient accumulation for memory efficiency - device-specific
    "gradient_accumulation_steps": DEVICE_CONFIG.get("gradient_accumulation_steps", 1),
    
    # Device-specific settings
    "mixed_precision": DEVICE_CONFIG["mixed_precision"],
}

# Dataset Configuration
DATASET_CONFIG = {
    "data_dir": "data",
    "image_size": MODEL_CONFIG["img_size"],
    "max_digits": TRAINING_CONFIG["max_digits"],
    "train_split": TRAINING_CONFIG["train_split"],
}

# Model file paths
MODEL_PATHS = {
    "best_model": "model_data/best_multi_digit_vit.pth",
    "final_model": "model_data/vit_model_final.pth",
    "best_seq2seq_model": "model_data/best_multi_digit_vit_seq2seq.pth",
    "final_seq2seq_model": "model_data/vit_model_final_seq2seq.pth",
}

# API Configuration
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "log_level": "debug",
}

# Streamlit Configuration
STREAMLIT_CONFIG = {
    "host": "0.0.0.0",
    "port": 8501,
    "model_url": "http://model-serve:8000/predict",
}

# Environment Configuration
ENV_CONFIG = {
    "device": DETECTED_DEVICE,
    "in_docker": os.path.exists('/.dockerenv'),
    "platform": platform.system(),
    "python_version": sys.version,
} 