import os
import tempfile
import numpy as np
import pytest
import torch

from engine.platform_utils import (
    get_device,
    get_profile_settings,
    is_colab,
    is_edge_pc,
    is_kaggle,
    optimize_cpu_threads,
    resolve_data_dir,
    resolve_output_dir,
    resolve_raw_dirs,
)
from environment.ips_env import IPSEnvironment
from data.dataset import CICIDSDataset


def test_device_selection():
    cpu_dev = get_device("cpu")
    assert cpu_dev.type == "cpu"

    auto_dev = get_device("auto")
    assert auto_dev.type in ("cuda", "mps", "cpu")

    # Graceful fallback for non-existent CUDA
    if not torch.cuda.is_available():
        fallback_dev = get_device("cuda")
        assert fallback_dev.type == "cpu"


def test_optimize_cpu_threads():
    threads = optimize_cpu_threads(torch.device("cpu"), max_threads=2)
    assert threads == 2
    assert torch.get_num_threads() == 2


def test_profile_settings():
    edge_cfg = get_profile_settings("edge")
    assert edge_cfg["profile"] == "edge"
    assert edge_cfg["sample_per_class"] == 5000
    assert edge_cfg["precompute_latents"] is True
    assert edge_cfg["tcn_batch_size"] == 32

    std_cfg = get_profile_settings("standard")
    assert std_cfg["profile"] == "standard"
    assert std_cfg["sample_per_class"] == 30000

    quick_cfg = get_profile_settings("quick")
    assert quick_cfg["profile"] == "quick"
    assert quick_cfg["tcn_epochs"] == 2


def test_resolve_directories():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a mock train_data.npz
        npz_file = os.path.join(tmpdir, "train_data.npz")
        np.savez(npz_file, X=np.zeros((10, 10, 78)), y=np.zeros(10), y_binary=np.zeros(10))

        resolved = resolve_data_dir(tmpdir)
        assert resolved == tmpdir

        # Test output dir creation
        out_target = os.path.join(tmpdir, "my_checkpoints")
        res_out = resolve_output_dir(out_target)
        assert os.path.isdir(res_out)


def test_precomputed_latents_in_environment():
    n_samples = 50
    X_seq = np.random.randn(n_samples, 10, 78).astype(np.float32)
    y_multi = np.random.choice([0, 1, 2], size=n_samples)
    y_binary = (y_multi > 0).astype(int)
    precomputed = np.ones((n_samples, 32), dtype=np.float32) * 42.0

    env = IPSEnvironment(
        X_seq=X_seq,
        y_multi=y_multi,
        y_binary=y_binary,
        precomputed_latents=precomputed,
        max_steps=10,
        state_dim=40,
    )

    obs, _ = env.reset()
    assert obs.shape == (40,)
    # The first 32 dimensions must match the precomputed latent
    assert np.allclose(obs[:32], 42.0)

    next_obs, reward, terminated, _, _ = env.step(0)
    assert next_obs.shape == (40,)
    assert np.allclose(next_obs[:32], 42.0)


def test_dataset_max_samples_subsampling():
    with tempfile.TemporaryDirectory() as tmpdir:
        npz_file = os.path.join(tmpdir, "train_data.npz")
        np.savez(
            npz_file,
            X=np.zeros((100, 10, 78), dtype=np.float32),
            y=np.zeros(100, dtype=np.int64),
            y_binary=np.zeros(100, dtype=np.int64),
        )

        full_ds = CICIDSDataset(npz_file)
        assert len(full_ds) == 100

        sub_ds = CICIDSDataset(npz_file, max_samples=25)
        assert len(sub_ds) == 25
