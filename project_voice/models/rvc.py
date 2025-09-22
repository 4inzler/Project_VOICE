"""Lightweight F0-guided RVC generator."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.activation(self.conv1(x))
        x = self.conv2(x)
        return self.activation(x + residual)


class F0Encoder(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, f0: torch.Tensor) -> torch.Tensor:
        if f0.dim() == 2:
            f0 = f0.unsqueeze(1)
        return self.conv(f0)


class RVCGenerator(nn.Module):
    """Simplified RVC generator with F0 guidance."""

    def __init__(
        self,
        content_dim: int,
        channels: int = 256,
        n_layers: int = 4,
        kernel_size: int = 7,
        strides: Tuple[int, ...] = (2, 2, 2, 2),
        upsample_initial_channel: int = 512,
        mel_channels: int = 80,
    ) -> None:
        super().__init__()
        self.pre = nn.Conv1d(content_dim, upsample_initial_channel, kernel_size=5, padding=2)
        self.f0_encoder = F0Encoder(upsample_initial_channel)

        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        in_channels = upsample_initial_channel
        for stride in strides:
            out_channels = in_channels // 2
            self.ups.append(
                nn.ConvTranspose1d(
                    in_channels,
                    out_channels,
                    kernel_size=stride * 2,
                    stride=stride,
                    padding=stride // 2,
                )
            )
            self.resblocks.append(
                nn.Sequential(
                    ResidualBlock(out_channels, kernel_size=kernel_size, dilation=1),
                    ResidualBlock(out_channels, kernel_size=kernel_size, dilation=3),
                )
            )
            in_channels = out_channels
        self.post = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(in_channels, mel_channels, kernel_size=5, padding=2),
            nn.ReLU(),
        )

    def forward(self, content: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
        x = self.pre(content)
        f0_encoded = self.f0_encoder(f0)
        if f0_encoded.shape[-1] != x.shape[-1]:
            f0_encoded = F.interpolate(f0_encoded, size=x.shape[-1], mode="linear", align_corners=True)
        x = x + f0_encoded

        for up, res in zip(self.ups, self.resblocks):
            x = up(x)
            x = res(x)
        return self.post(x)


__all__ = ["RVCGenerator"]
