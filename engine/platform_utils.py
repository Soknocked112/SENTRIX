import glob
import logging
import os
import platform
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

logger = logging.getLogger("platform_utils")


def is_kaggle() -> bool:
    """Check if code is running inside a Kaggle notebook or environment."""
    return (
        os.path.exists("/kaggle")
        or "KAGGLE_KERNEL_RUN_TYPE" in os.environ
        or "KAGGLE_URL_BASE" in os.environ
    )


def is_colab() -> bool:
    """Check if code is running inside Google Colaboratory."""
    return "google.colab" in sys.modules or (
        os.path.exists("/content") and "COLAB_GPU" in os.environ
    )


def get_system_ram_gb() -> Optional[float]:
    """Retrieve total system RAM in gigabytes across Linux and other OSes."""
    try:
        if os.path.exists("/proc/meminfo"):
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = float(line.split()[1])
                        return round(kb / (1024.0 * 1024.0), 2)
    except Exception:
        pass

    try:
        import psutil
        return round(psutil.virtual_memory().total / (1024.0 ** 3), 2)
    except Exception:
        return None


def is_edge_pc() -> bool:
    """
    Detect whether the host is an edge device (e.g. Raspberry Pi, Jetson,
    ARM SBC, or low-RAM constrained PC).
    """
    arch = platform.machine().lower()
    is_arm = any(a in arch for a in ("arm", "aarch64", "armv7l", "armv8"))

    ram_gb = get_system_ram_gb()
    low_ram = ram_gb is not None and ram_gb <= 8.0

    return is_arm or low_ram


def get_device(requested_device: Optional[str] = None) -> torch.device:
    """
    Intelligently select PyTorch compute device ('cuda', 'mps', 'cpu', or 'auto').
    Gracefully falls back to CPU if requested accelerator is unavailable.
    """
    req = (requested_device or "auto").lower().strip()

    if req == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")

    if req == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        logger.warning("Apple MPS requested but not available. Falling back to CPU.")
        return torch.device("cpu")

    if req == "cpu":
        return torch.device("cpu")

    # Auto resolution
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def optimize_cpu_threads(device: torch.device, max_threads: Optional[int] = None) -> int:
    """
    Set optimal CPU thread allocation to prevent thread contention / thermal throttling
    on multi-core edge devices.
    """
    if device.type == "cpu":
        cpu_cnt = os.cpu_count() or 1
        target_threads = max_threads or min(4, cpu_cnt)
        torch.set_num_threads(max(1, target_threads))
        return target_threads
    return 1


def resolve_data_dir(
    data_dir: Optional[str] = None,
    cfg_data_dir: Optional[str] = None,
    default_dir: str = "data/processed",
) -> str:
    """
    Resolve preprocessed dataset path. If not found locally, automatically
    searches Kaggle input directory (/kaggle/input) or Colab (/content).
    """
    candidates = []
    if data_dir:
        candidates.append(data_dir)
    if cfg_data_dir:
        candidates.append(cfg_data_dir)
    candidates.append(default_dir)

    for c in candidates:
        if c and os.path.isdir(c) and os.path.exists(os.path.join(c, "train_data.npz")):
            return c

    # Search Kaggle input paths
    if is_kaggle() or os.path.exists("/kaggle/input"):
        logger.info("Scanning /kaggle/input for preprocessed datasets (train_data.npz)...")
        found = glob.glob("/kaggle/input/**/train_data.npz", recursive=True)
        if found:
            detected_dir = os.path.dirname(found[0])
            logger.info(f"Auto-detected preprocessed data at: {detected_dir}")
            return detected_dir

    # Search Colab paths
    if is_colab() or os.path.exists("/content"):
        found = glob.glob("/content/**/train_data.npz", recursive=True)
        if found:
            detected_dir = os.path.dirname(found[0])
            logger.info(f"Auto-detected preprocessed data at: {detected_dir}")
            return detected_dir

    # Fallback to candidate or default
    return data_dir or cfg_data_dir or default_dir


def resolve_raw_dirs(
    override_dirs: Optional[List[str]] = None,
    cfg_dirs: Optional[Union[str, List[str]]] = None,
) -> List[str]:
    """
    Resolve raw CSV directories. If not found locally, automatically
    scans Kaggle /kaggle/input for directories containing flow CSV files.
    """
    if override_dirs:
        return override_dirs

    raw_list: List[str] = []
    if isinstance(cfg_dirs, str):
        raw_list = [cfg_dirs]
    elif isinstance(cfg_dirs, list):
        raw_list = list(cfg_dirs)

    # Check if configured directories exist and contain CSV files
    valid_dirs = []
    for d in raw_list:
        if os.path.isdir(d) and len(glob.glob(os.path.join(d, "*.csv"))) > 0:
            valid_dirs.append(d)

    if valid_dirs:
        return valid_dirs

    # Scan Kaggle input directories
    if is_kaggle() or os.path.exists("/kaggle/input"):
        logger.info("Scanning /kaggle/input for raw CSV dataset directories...")
        kaggle_csvs = glob.glob("/kaggle/input/**/*.csv", recursive=True)
        if kaggle_csvs:
            discovered_dirs = sorted(list({os.path.dirname(f) for f in kaggle_csvs}))
            logger.info(f"Auto-detected {len(discovered_dirs)} dataset directories in /kaggle/input:")
            for dd in discovered_dirs:
                logger.info(f"  - {dd}")
            return discovered_dirs

    return raw_list


def resolve_output_dir(
    output_dir: Optional[str] = None,
    default_dir: str = "checkpoints",
) -> str:
    """
    Resolve a writable output directory. On Kaggle, defaults to /kaggle/working/<dir>.
    """
    if output_dir:
        target = output_dir
    elif is_kaggle() and not os.path.isabs(default_dir):
        target = os.path.join("/kaggle/working", default_dir)
    else:
        target = default_dir

    os.makedirs(target, exist_ok=True)
    return target


def get_profile_settings(profile_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Retrieve hyperparameter presets based on platform profile:
      - 'standard': Full cloud/GPU dataset & model settings
      - 'edge': Low-memory, low-latency preset for Edge PCs / low RAM
      - 'quick': Ultra-fast sanity check preset
    """
    name = (profile_name or "").lower().strip()
    if not name:
        name = "edge" if is_edge_pc() else "standard"

    if name in ("edge", "edge_pc", "iot", "rpi", "jetson"):
        return {
            "profile": "edge",
            "sample_per_class": 5000,
            "tcn_epochs": 8,
            "tcn_batch_size": 32,
            "drl_episodes": 60,
            "drl_steps_per_episode": 250,
            "drl_batch_size": 32,
            "max_samples": 20000,
            "num_workers": 0,
            "precompute_latents": True,
        }
    elif name in ("quick", "test", "ci"):
        return {
            "profile": "quick",
            "sample_per_class": 1000,
            "tcn_epochs": 2,
            "tcn_batch_size": 32,
            "drl_episodes": 10,
            "drl_steps_per_episode": 100,
            "drl_batch_size": 16,
            "max_samples": 5000,
            "num_workers": 0,
            "precompute_latents": True,
        }
    else:
        return {
            "profile": "standard",
            "sample_per_class": 30000,
            "tcn_epochs": 15,
            "tcn_batch_size": 128,
            "drl_episodes": 200,
            "drl_steps_per_episode": 500,
            "drl_batch_size": 64,
            "max_samples": None,
            "num_workers": 2 if torch.cuda.is_available() else 0,
            "precompute_latents": True,
        }


def print_platform_diagnostics():
    """Print hardware and runtime platform diagnostics."""
    ram = get_system_ram_gb()
    ram_str = f"{ram:.1f} GB" if ram else "Unknown"
    is_k = is_kaggle()
    is_c = is_colab()
    is_e = is_edge_pc()
    plat_str = "Kaggle" if is_k else ("Colab" if is_c else ("Edge PC" if is_e else "Standard Linux/Desktop"))

    device = get_device("auto")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"

    print("=" * 65)
    print(" SENTRIX IPS: PLATFORM & HARDWARE DIAGNOSTICS")
    print("=" * 65)
    print(f" Environment:        {plat_str}")
    print(f" Architecture:       {platform.machine()} ({platform.system()})")
    print(f" CPU Cores:          {os.cpu_count()}")
    print(f" System RAM:         {ram_str}")
    print(f" Compute Device:     {device.type.upper()} ({gpu_name})")
    print(f" PyTorch Version:    {torch.__version__}")
    print("=" * 65)
