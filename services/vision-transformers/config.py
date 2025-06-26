"""
Centralized configuration for the Vision Transformer model.
All components (training, serving, evaluation) should use this config.
"""

# Model Architecture Configuration
MODEL_CONFIG = {
    # Image and patch settings
    "img_size": 84,
    "patch_size": 16,
    "in_channels": 1,
    
    # Model dimensions - much larger for better accuracy
    "d_model": 128,  # Increased from 96 for much better accuracy
    "num_encoder_layers": 4,  # Increased from 2 for better accuracy
    "num_decoder_layers": 4,  # Increased from 2 for better accuracy
    "heads": 2,  # Increased from 4 for better accuracy
    "head_dim": 32,  # Increased from 24 for better accuracy
    "ff_hidden_dim": 512,  # Increased from 192 for better accuracy
    
    # Seq2Seq settings
    "vocab_size": 12,  # 0-9 digits + START token (10) + EOS token (11) for seq2seq
    
    # Training settings - increased regularization
    "dropout": 0.1,
}

# Training Configuration - optimized for speed
TRAINING_CONFIG = {
    "epochs": 10,  # Increased from 15 for better accuracy
    "batch_size": 128,  # Increased from 32 for MPS
    "learning_rate": 0.004,  # Reduced from 5e-4 for stability with larger model
    "weight_decay": 0.01,
    "train_split": 0.8,
    "device": "mps",  # Use Apple Silicon GPU
    "max_digits": 3,
    "max_samples": 8000,  # Increased from 5000 for more data
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