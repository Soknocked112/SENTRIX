from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Chomp1d(nn.Module):
    """Prunes right-side padding so convolutions remain strictly causal."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size <= 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
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

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
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
    TCN for sequence classification and latent embedding extraction.
    Outputs:
      - logits: threat class predictions
      - latent: intermediate representation fed into the RL policy
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

        self.tcn = TemporalConvNet(
            num_inputs=input_features,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        last_channel = num_channels[-1]
        self.latent_proj = nn.Sequential(
            nn.Linear(last_channel, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # (B, T, F) -> (B, F, T) for 1D convolution
        x_trans = x.transpose(1, 2)
        tcn_out = self.tcn(x_trans)

        # Slice the last time step (t = -1)
        latent = self.latent_proj(tcn_out[:, :, -1])
        logits = self.classifier(latent)

        return logits, latent

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            _, latent = self.forward(x)
        return latent
