"""Super-resolution backbones (TIR 1-channel, 2x). Swappable via config.

* ``edsr``    — fast, robust CNN baseline (fidelity-first).
* ``swinir``  — compact SwinIR-style window-attention transformer.

Both map (B,1,h,w) -> (B,1,2h,2w). Use :func:`build_sr_model`.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tir.models.blocks import ResidualBlock, Upsampler, default_conv


# --------------------------------------------------------------------------- #
# EDSR
# --------------------------------------------------------------------------- #
class EDSR(nn.Module):
    def __init__(self, scale: int = 2, in_ch: int = 1, num_features: int = 64,
                 num_blocks: int = 8, res_scale: float = 1.0):
        super().__init__()
        self.head = default_conv(in_ch, num_features, 3)
        self.body = nn.Sequential(
            *[ResidualBlock(num_features, res_scale=res_scale) for _ in range(num_blocks)],
            default_conv(num_features, num_features, 3),
        )
        self.tail = nn.Sequential(
            Upsampler(scale, num_features),
            default_conv(num_features, in_ch, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.head(x)
        feat = feat + self.body(feat)
        return self.tail(feat)


# --------------------------------------------------------------------------- #
# Compact SwinIR
# --------------------------------------------------------------------------- #
def _window_partition(x: torch.Tensor, ws: int) -> torch.Tensor:
    b, h, w, c = x.shape
    x = x.view(b, h // ws, ws, w // ws, ws, c)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws * ws, c)


def _window_reverse(windows: torch.Tensor, ws: int, h: int, w: int) -> torch.Tensor:
    b = int(windows.shape[0] / (h * w / ws / ws))
    x = windows.view(b, h // ws, w // ws, ws, ws, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)


class WindowAttention(nn.Module):
    def __init__(self, dim: int, window_size: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        return self.proj(out)


class SwinBlock(nn.Module):
    def __init__(self, dim: int, window_size: int, num_heads: int, shift: int,
                 mlp_ratio: float = 2.0):
        super().__init__()
        self.window_size = window_size
        self.shift = shift
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, _, c = x.shape
        shortcut = x
        x = self.norm1(x).view(b, h, w, c)
        if self.shift > 0:
            x = torch.roll(x, shifts=(-self.shift, -self.shift), dims=(1, 2))
        windows = _window_partition(x, self.window_size)
        windows = self.attn(windows)
        x = _window_reverse(windows, self.window_size, h, w)
        if self.shift > 0:
            x = torch.roll(x, shifts=(self.shift, self.shift), dims=(1, 2))
        x = x.view(b, h * w, c)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class RSTB(nn.Module):
    """Residual Swin Transformer Block: a few Swin blocks + conv, with residual."""

    def __init__(self, dim: int, depth: int, window_size: int, num_heads: int):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinBlock(dim, window_size, num_heads,
                      shift=0 if (i % 2 == 0) else window_size // 2)
            for i in range(depth)
        ])
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        shortcut = x
        for blk in self.blocks:
            x = blk(x, h, w)
        b, _, c = x.shape
        feat = x.transpose(1, 2).view(b, c, h, w)
        feat = self.conv(feat).flatten(2).transpose(1, 2)
        return shortcut + feat


class SwinIR(nn.Module):
    def __init__(self, scale: int = 2, in_ch: int = 1, embed_dim: int = 60,
                 depths=(4, 4), num_heads: int = 4, window_size: int = 8):
        super().__init__()
        self.window_size = window_size
        self.conv_first = nn.Conv2d(in_ch, embed_dim, 3, padding=1)
        self.layers = nn.ModuleList([
            RSTB(embed_dim, depth, window_size, num_heads) for depth in depths
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.conv_after = nn.Conv2d(embed_dim, embed_dim, 3, padding=1)
        self.upsample = nn.Sequential(Upsampler(scale, embed_dim),
                                      nn.Conv2d(embed_dim, in_ch, 3, padding=1))

    def _pad(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        _, _, h, w = x.shape
        ws = self.window_size
        ph = (ws - h % ws) % ws
        pw = (ws - w % ws) % ws
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        return x, h, w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, h0, w0 = self._pad(x)
        feat = self.conv_first(x)
        b, c, h, w = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)
        for layer in self.layers:
            tokens = layer(tokens, h, w)
        tokens = self.norm(tokens)
        feat2 = tokens.transpose(1, 2).view(b, c, h, w)
        feat = feat + self.conv_after(feat2)
        out = self.upsample(feat)
        scale = out.shape[-1] // x.shape[-1]
        return out[:, :, : h0 * scale, : w0 * scale]


_REGISTRY = {"edsr": EDSR, "swinir": SwinIR}


def build_sr_model(cfg: dict) -> nn.Module:
    """Build an SR model from a config dict (``cfg['name']`` selects backbone)."""
    cfg = dict(cfg)
    name = cfg.pop("name", "edsr").lower()
    if name not in _REGISTRY:
        raise ValueError(f"Unknown SR model '{name}'. Options: {list(_REGISTRY)}")
    if name == "edsr":
        return EDSR(scale=cfg.get("scale", 2), num_features=cfg.get("num_features", 64),
                    num_blocks=cfg.get("num_blocks", 8),
                    res_scale=cfg.get("res_scale", 1.0))
    return SwinIR(scale=cfg.get("scale", 2), embed_dim=cfg.get("embed_dim", 60),
                  depths=tuple(cfg.get("depths", (4, 4))),
                  num_heads=cfg.get("num_heads", 4),
                  window_size=cfg.get("window_size", 8))
