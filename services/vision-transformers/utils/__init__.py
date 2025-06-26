"""
Utility modules for the Vision Transformer project.
"""

from .device_utils import (
    detect_device,
    get_device_config,
    get_mlx_device,
    print_device_info,
    setup_device_for_training
)

__all__ = [
    'detect_device',
    'get_device_config', 
    'get_mlx_device',
    'print_device_info',
    'setup_device_for_training'
] 