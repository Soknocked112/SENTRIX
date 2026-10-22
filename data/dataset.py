"""
dataset.py - PyTorch Dataset & DataLoader utilities for temporal flow sequences.
"""

import os
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class CICIDSDataset(Dataset):
    """
    PyTorch Dataset for sliding-window sequential flow records.
    Features: (batch, seq_len, num_features)
    Labels: multi-class (0..5) and binary (0 or 1)
    """

    def __init__(self, npz_path: str):
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Processed dataset file not found: {npz_path}")

        data = np.load(npz_path)
        self.X = torch.from_numpy(data["X"]).float()
        self.y = torch.from_numpy(data["y"]).long()
        self.y_binary = torch.from_numpy(data["y_binary"]).long()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx], self.y_binary[idx]


def get_dataloaders(
    data_dir: str = "data/processed",
    batch_size: int = 128,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Factory creating PyTorch DataLoaders for Train, Val, and Test splits."""
    train_path = os.path.join(data_dir, "train_data.npz")
    val_path = os.path.join(data_dir, "val_data.npz")
    test_path = os.path.join(data_dir, "test_data.npz")

    train_ds = CICIDSDataset(train_path)
    val_ds = CICIDSDataset(val_path)
    test_ds = CICIDSDataset(test_path)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
