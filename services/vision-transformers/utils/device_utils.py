"""
Device detection and management utilities for PyTorch.
Handles automatic device selection and provides device-specific configurations.
"""

import os
import platform
import subprocess
import sys
from typing import Dict, Any, Optional

def detect_device() -> str:
    """
    Automatically detect the best available device for PyTorch.
    Priority: CUDA (NVIDIA GPU) > MPS (Apple Silicon) > CPU
    
    Returns:
        str: Device name ('cuda', 'mps', or 'cpu')
    """
    # Check if we're in a Docker container
    in_docker = os.path.exists('/.dockerenv')
    
    # Check for NVIDIA GPU (CUDA)
    if in_docker or platform.system() == "Linux":
        if _check_cuda_available():
            print("🚀 Detected NVIDIA GPU - using CUDA")
            return "cuda"
    
    # Check for Apple Silicon (MPS)
    if platform.system() == "Darwin" and not in_docker:
        if _check_mps_available():
            print("🍎 Detected Apple Silicon - using MPS")
            return "mps"
    
    # Fallback to CPU
    print("💻 Using CPU (no CUDA/MPS detected)")
    return "cpu"

def _check_cuda_available() -> bool:
    """Check if CUDA is available via PyTorch."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        # Fallback to nvidia-smi if PyTorch not available
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            return False

def _check_mps_available() -> bool:
    """Check if MPS is available via PyTorch."""
    try:
        import torch
        return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    except ImportError:
        return False

def get_device_config(device: str) -> Dict[str, Any]:
    """
    Get device-specific configuration.
    
    Args:
        device: Device name ('cuda', 'mps', 'cpu')
        
    Returns:
        Dict containing device-specific settings
    """
    configs = {
        "cuda": {
            "batch_size": 128,
            "learning_rate": 0.0001,
            "mixed_precision": True,
            "gradient_accumulation_steps": 1,  # GPU can handle larger batches
            "description": "NVIDIA GPU with CUDA support"
        },
        "mps": {
            "batch_size": 64,
            "learning_rate": 0.0001,
            "mixed_precision": False,  # MPS doesn't support mixed precision yet
            "gradient_accumulation_steps": 2,  # Moderate accumulation for MPS
            "description": "Apple Silicon with Metal Performance Shaders"
        },
        "cpu": {
            "batch_size": 32,
            "learning_rate": 0.00005,  # Lower learning rate for stability
            "mixed_precision": False,
            "gradient_accumulation_steps": 4,  # More accumulation for CPU
            "description": "CPU fallback"
        }
    }
    
    return configs.get(device, configs["cpu"])

def get_torch_device(device: str) -> str:
    """
    Get the PyTorch device string for the given device.
    
    Args:
        device: Device name ('cuda', 'mps', 'cpu')
        
    Returns:
        PyTorch device string
    """
    device_map = {
        "cuda": "cuda",
        "mps": "mps", 
        "cpu": "cpu"
    }
    return device_map.get(device, "cpu")

def print_device_info(device: str):
    """Print detailed device information."""
    config = get_device_config(device)
    
    print(f"\n🔧 Device Configuration:")
    print(f"   Device: {device.upper()}")
    print(f"   Description: {config['description']}")
    print(f"   Batch Size: {config['batch_size']}")
    print(f"   Learning Rate: {config['learning_rate']}")
    print(f"   Mixed Precision: {config['mixed_precision']}")
    
    # Print environment info
    print(f"\n🌍 Environment:")
    print(f"   Platform: {platform.system()}")
    print(f"   Python: {sys.version.split()[0]}")
    print(f"   In Docker: {os.path.exists('/.dockerenv')}")
    
    # Print PyTorch info if available
    try:
        import torch
        print(f"   PyTorch Version: {torch.__version__}")
        if device == "cuda" and torch.cuda.is_available():
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   GPU Count: {torch.cuda.device_count()}")
            print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
        elif device == "mps" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print(f"   MPS Available: Yes")
    except ImportError:
        print("   PyTorch: Not installed")

def setup_device_for_training(device: str) -> Dict[str, Any]:
    """
    Setup device-specific configuration for training.
    
    Args:
        device: Device name
        
    Returns:
        Dict with training configuration
    """
    config = get_device_config(device)
    
    # Base training config
    training_config = {
        "device": device,
        "batch_size": config["batch_size"],
        "learning_rate": config["learning_rate"],
        "mixed_precision": config["mixed_precision"],
    }
    
    # Device-specific optimizations
    if device == "cuda":
        training_config.update({
            "gradient_accumulation_steps": 1,  # GPU can handle larger batches
            "num_workers": 4,  # More workers for GPU
        })
    elif device == "mps":
        training_config.update({
            "gradient_accumulation_steps": 2,  # Moderate accumulation for MPS
            "num_workers": 2,  # Fewer workers for MPS
        })
    else:  # CPU
        training_config.update({
            "gradient_accumulation_steps": 4,  # More accumulation for CPU
            "num_workers": 1,  # Single worker for CPU
        })
    
    return training_config

def get_torch_device_object(device: str):
    """
    Get a PyTorch device object.
    
    Args:
        device: Device name ('cuda', 'mps', 'cpu')
        
    Returns:
        PyTorch device object
    """
    try:
        import torch
        device_str = get_torch_device(device)
        return torch.device(device_str)
    except ImportError:
        raise ImportError("PyTorch is required but not installed")

if __name__ == "__main__":
    # Test device detection
    device = detect_device()
    print_device_info(device)
    
    # Show training config
    training_config = setup_device_for_training(device)
    print(f"\n📋 Training Configuration:")
    for key, value in training_config.items():
        print(f"   {key}: {value}")
    
    # Test PyTorch device object
    try:
        torch_device = get_torch_device_object(device)
        print(f"\n🔧 PyTorch Device Object: {torch_device}")
    except ImportError:
        print("\n❌ PyTorch not installed - cannot create device object") 