# Utilities

This directory contains utility modules for device detection, Hugging Face integration, and other common functionality.

## Structure

```
utils/
├── __init__.py              # Package initialization and exports
├── device_utils.py          # Device detection and configuration
├── huggingface_utils.py     # Hugging Face model upload utilities
└── README.md                # This file
```

## Utility Modules

### Device Utils (`device_utils.py`)

Utilities for automatic device detection and configuration across different platforms.

**Features:**
- **Automatic Device Detection**: Detects CUDA, MPS, or CPU
- **Platform-Specific Optimization**: Different configs for different devices
- **Performance Monitoring**: Device information and capabilities
- **Fallback Handling**: Graceful degradation when preferred device unavailable

**Functions:**
- `detect_device()`: Automatically detect the best available device
- `get_device_config(device)`: Get device-specific configuration
- `print_device_info(device)`: Display device information
- `get_torch_device_object(device)`: Get PyTorch device object

**Usage:**
```python
from utils.device_utils import detect_device, get_device_config

# Detect best device
device = detect_device()  # Returns "cuda", "mps", or "cpu"

# Get device-specific config
config = get_device_config(device)
# Returns: {"batch_size": 128, "learning_rate": 0.0001, "mixed_precision": True}
```

### Hugging Face Utils (`huggingface_utils.py`)

Utilities for uploading trained models to Hugging Face Hub.

**Features:**
- **Model Upload**: Upload PyTorch models to Hugging Face
- **Metadata Management**: Add model cards and descriptions
- **Version Control**: Track model versions and changes
- **Token Management**: Secure authentication with Hugging Face

**Functions:**
- `upload_model_to_hf()`: Upload model with metadata
- `create_model_card()`: Generate model card content
- `validate_hf_token()`: Validate Hugging Face token
- `get_model_info()`: Extract model information

**Usage:**
```python
from utils.huggingface_utils import upload_model_to_hf

# Upload model to Hugging Face
upload_model_to_hf(
    model_path="model_data/best_model.pth",
    repo_name="your-username/transformer-models",
    model_name="vision-transformer-classifier",
    description="Vision Transformer for digit classification"
)
```

## Device Configurations

### CUDA (NVIDIA GPU)
- **Batch Size**: 128
- **Learning Rate**: 0.0001
- **Mixed Precision**: True
- **Gradient Accumulation**: 1

### MPS (Apple Silicon)
- **Batch Size**: 64
- **Learning Rate**: 0.0001
- **Mixed Precision**: False
- **Gradient Accumulation**: 1

### CPU
- **Batch Size**: 32
- **Learning Rate**: 0.00005
- **Mixed Precision**: False
- **Gradient Accumulation**: 2

## Integration

These utilities are used throughout the project:

- **`config.py`**: Uses device detection for automatic configuration
- **Training Scripts**: Use device-specific configurations
- **Model Serving**: Uses device detection for inference
- **Infrastructure**: Uses Hugging Face utils for model deployment

## Environment Variables

### Device Utils
- `CUDA_VISIBLE_DEVICES`: Control which CUDA devices are visible
- `PYTORCH_ENABLE_MPS_FALLBACK`: Enable MPS fallback to CPU

### Hugging Face Utils
- `HF_TOKEN`: Hugging Face authentication token
- `HF_REPO`: Default repository for model uploads
- `HF_USERNAME`: Hugging Face username

## Error Handling

Both modules include comprehensive error handling:
- **Device Detection**: Graceful fallback to CPU if GPU unavailable
- **Token Validation**: Clear error messages for authentication issues
- **Network Issues**: Retry logic for Hugging Face uploads
- **Platform Compatibility**: Cross-platform support with appropriate warnings 