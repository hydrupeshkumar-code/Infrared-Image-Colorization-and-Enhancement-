"""VGG perceptual loss. Lazily loads VGG16 features; falls back gracefully if
weights cannot be downloaded (offline) by using random-init features, which
still provides a multi-scale feature-space penalty for smoke tests."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VGGPerceptual(nn.Module):
    def __init__(self, layers=(3, 8, 15), resize: bool = True):
        super().__init__()
        from torchvision.models import vgg16
        try:
            from torchvision.models import VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        except Exception:
            vgg = vgg16(weights=None).features  # offline fallback
        self.slices = nn.ModuleList()
        prev = 0
        for layer in layers:
            self.slices.append(nn.Sequential(*[vgg[i] for i in range(prev, layer + 1)]))
            prev = layer + 1
        for p in self.parameters():
            p.requires_grad = False
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = (x + 1.0) / 2.0 if x.min() < 0 else x  # map [-1,1]->[0,1] if needed
        x = (x - self.mean) / self.std
        if self.resize and x.shape[-1] < 32:
            x = F.interpolate(x, size=32, mode="bilinear", align_corners=False)
        return x

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p, t = self._prep(pred), self._prep(target)
        loss = pred.new_zeros(())
        for slice_ in self.slices:
            p, t = slice_(p), slice_(t)
            loss = loss + F.l1_loss(p, t)
        return loss
