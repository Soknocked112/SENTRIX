"""
tcn.py - Temporal Convolutional Network (TCN) for Network Intrusion Detection

Implements Dilated Causal 1D Convolutions with Residual Blocks.
Provides threat classification and extracts latent temporal feature embeddings
used as the state observation for the Deep Reinforcement Learning (DRL) agent.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Chomp1d(nn.Module):
    """
    Causal padding adjustment: removes the extra padded elements on the right
    to guarantee that convolutions do not leak future information.
    """

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size <= 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    Residual Block with two dilated causal convolution layers,
    batch normalization, ReLU activation, and dropout.
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.bn1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.bn2, self.relu2, self.dropout2
        )

        # 1x1 conv for residual dimension matching if input and output channels differ
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """
    Stack of TemporalBlocks with exponentially increasing dilations (1, 2, 4, 8, ...).
    """

    def __init__(
        self,
        num_inputs: int,
        num_channels: List[int],
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=padding,
                    dropout=dropout,
                )
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TCNThreatDetector(nn.Module):
    """
    Dual-head Temporal Convolutional Network:
    1. Threat Classification Head (multi-class intrusion detection logits)
    2. Latent Representation Head (temporal feature embedding fed to DRL)
    """

    def __init__(
        self,
        input_features: int = 78,
        num_channels: Optional[List[int]] = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
        latent_dim: int = 32,
        num_classes: int = 6,
    ):
        super().__init__()
        if num_channels is None:
            num_channels = [64, 64, 64, 64]

        self.input_features = input_features
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # Temporal Backbone
        self.tcn = TemporalConvNet(
            num_inputs=input_features,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        last_channel = num_channels[-1]

        # Latent Temporal Projection Head
        self.latent_proj = nn.Sequential(
            nn.Linear(last_channel, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
        )

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input shape: (batch_size, seq_len, input_features)
        Returns:
            logits: (batch_size, num_classes)
            latent: (batch_size, latent_dim)
        """
        # Transpose from (B, T, F) to (B, F, T) for 1D convolution
        x_trans = x.transpose(1, 2)

        # Pass through dilated causal TCN
        tcn_out = self.tcn(x_trans)  # (B, C_last, T)

        # Extract features at the most recent time step (t = -1)
        last_step_features = tcn_out[:, :, -1]  # (B, C_last)

        # Generate latent representation
        latent = self.latent_proj(last_step_features)  # (B, latent_dim)

        # Generate class logits
        logits = self.classifier(latent)  # (B, num_classes)

        return logits, latent

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract only the latent temporal embedding (for DRL observation)."""
        self.eval()
        with torch.no_grad():
            _, latent = self.forward(x)
        return latent
