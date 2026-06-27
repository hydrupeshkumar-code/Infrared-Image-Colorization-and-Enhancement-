"""Common CNN building blocks shared by SR backbones."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def default_conv(in_ch: int, out_ch: int, kernel_size: int, bias: bool = True) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2, bias=bias)


class MeanShift(nn.Conv2d):
    """Normalize/denormalize by a fixed per-channel mean (identity for 1-ch TIR
    when mean=0). Kept for parity with EDSR-style backbones."""

    def __init__(self, mean: float = 0.0, std: float = 1.0, n_channels: int = 1,
                 sign: int = -1):
        super().__init__(n_channels, n_channels, kernel_size=1)
        self.weight.data = torch.eye(n_channels).view(n_channels, n_channels, 1, 1) / std
        self.bias.data = sign * mean / std * torch.ones(n_channels)
        for p in self.parameters():
            p.requires_grad = False


class ResidualBlock(nn.Module):
    """EDSR residual block: conv-relu-conv with residual scaling (no BN)."""

    def __init__(self, n_feats: int, kernel_size: int = 3, res_scale: float = 1.0):
        super().__init__()
        self.body = nn.Sequential(
            default_conv(n_feats, n_feats, kernel_size),
            nn.ReLU(inplace=True),
            default_conv(n_feats, n_feats, kernel_size),
        )
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x) * self.res_scale


class Upsampler(nn.Sequential):
    """Pixel-shuffle upsampler for power-of-two or x3 scales."""

    def __init__(self, scale: int, n_feats: int):
        layers: list[nn.Module] = []
        if (scale & (scale - 1)) == 0:  # power of two
            for _ in range(int(math.log2(scale))):
                layers.append(default_conv(n_feats, 4 * n_feats, 3))
                layers.append(nn.PixelShuffle(2))
        elif scale == 3:
            layers.append(default_conv(n_feats, 9 * n_feats, 3))
            layers.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f"Unsupported scale {scale}")
        super().__init__(*layers)
