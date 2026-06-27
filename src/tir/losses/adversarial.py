"""Adversarial loss (LSGAN by default) for the colorization PatchGAN."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GANLoss(nn.Module):
    """Least-squares GAN loss. ``target_is_real`` picks the 1/0 target."""

    def __init__(self, mode: str = "lsgan"):
        super().__init__()
        self.mode = mode

    def __call__(self, pred: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
        if self.mode == "lsgan":
            return F.mse_loss(pred, target)
        return F.binary_cross_entropy_with_logits(pred, target)
