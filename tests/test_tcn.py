"""
test_tcn.py - Unit tests for Temporal Convolutional Network (TCN)
"""

import pytest
import torch
from models.tcn import Chomp1d, TCNThreatDetector, TemporalBlock, TemporalConvNet


def test_chomp1d():
    chomp = Chomp1d(chomp_size=2)
    x = torch.randn(4, 8, 10)  # (B, C, T)
    out = chomp(x)
    assert out.shape == (4, 8, 8)
    assert torch.equal(out, x[:, :, :8])


def test_temporal_block():
    block = TemporalBlock(
        n_inputs=16,
        n_outputs=32,
        kernel_size=3,
        stride=1,
        dilation=2,
        padding=4,
        dropout=0.1,
    )
    x = torch.randn(4, 16, 12)
    out = block(x)
    assert out.shape == (4, 32, 12)


def test_tcn_forward_and_latent():
    model = TCNThreatDetector(
        input_features=20,
        num_channels=[32, 32],
        kernel_size=3,
        dropout=0.1,
        latent_dim=16,
        num_classes=6,
    )
    x = torch.randn(8, 10, 20)  # (B, T, F)
    logits, latent = model(x)

    assert logits.shape == (8, 6)
    assert latent.shape == (8, 16)


def test_tcn_causality():
    """Verify that future time-steps cannot leak backward to influence earlier outputs."""
    model = TCNThreatDetector(
        input_features=10,
        num_channels=[16, 16],
        kernel_size=3,
        dropout=0.0,
        latent_dim=8,
        num_classes=6,
    )
    model.eval()

    x1 = torch.randn(1, 10, 10)
    x2 = x1.clone()
    # Mutate the future (step 8 and 9)
    x2[:, 8:, :] += torch.randn(1, 2, 10)

    with torch.no_grad():
        x1_trans = x1.transpose(1, 2)
        x2_trans = x2.transpose(1, 2)
        out1 = model.tcn(x1_trans)
        out2 = model.tcn(x2_trans)

    # Time steps 0 through 7 must be strictly identical
    assert torch.allclose(out1[:, :, :8], out2[:, :, :8], atol=1e-5)
