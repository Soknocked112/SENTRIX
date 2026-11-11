"""
Engine utilities for platform detection and diagnostics.
"""

from engine.platform_utils import (
    is_kaggle,
    is_colab,
    get_device,
    print_platform_diagnostics,
    resolve_data_dir,
    resolve_raw_dirs,
    resolve_output_dir,
    get_profile_settings,
)

__all__ = [
    "is_kaggle",
    "is_colab",
    "get_device",
    "print_platform_diagnostics",
    "resolve_data_dir",
    "resolve_raw_dirs",
    "resolve_output_dir",
    "get_profile_settings",
]
