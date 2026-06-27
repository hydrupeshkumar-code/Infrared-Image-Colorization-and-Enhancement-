"""Physics-informed components (BONUS).

Two ideas, both aimed at reducing hallucination and keeping the SR output
radiometrically faithful:

1. ``DownsampleConsistencyLoss`` — downsampling the super-resolved output by
   the SR factor must reproduce the low-resolution input. This is a cycle /
   reconstruction constraint that strongly penalises invented structure that
   is not supported by the LR observation.

2. ``dn_to_brightness_temperature`` — convert Landsat L1 TIR digital numbers
   to at-sensor brightness temperature (Kelvin) via the Planck-based USGS
   relation. Operating SR in temperature units makes the fidelity metrics
   physically interpretable. Emissivity is assumed ~1 (at-sensor BT); land
   surface temperature would additionally require an emissivity correction.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DownsampleConsistencyLoss(nn.Module):
    """|down(SR) - LR|: SR must collapse back to the observed LR input."""

    def __init__(self, scale: int = 2, mode: str = "area"):
        super().__init__()
        self.scale = scale
        self.mode = mode

    def forward(self, sr: torch.Tensor, lr: torch.Tensor) -> torch.Tensor:
        down = F.interpolate(sr, scale_factor=1.0 / self.scale, mode=self.mode)
        if down.shape[-2:] != lr.shape[-2:]:
            down = F.interpolate(down, size=lr.shape[-2:], mode=self.mode)
        return F.l1_loss(down, lr)


# Landsat-9 TIRS Band 10 thermal constants (USGS Collection-2 metadata).
# K1, K2 come from the MTL file (RADIANCE/THERMAL constants); defaults below
# are the published nominal B10 values.
B10_K1 = 799.0329
B10_K2 = 1329.2405
B10_ML = 3.8e-4   # RADIANCE_MULT_BAND_10 (nominal)
B10_AL = 0.1      # RADIANCE_ADD_BAND_10 (nominal)


def dn_to_brightness_temperature(dn, ml: float = B10_ML, al: float = B10_AL,
                                 k1: float = B10_K1, k2: float = B10_K2):
    """Landsat L1 DN -> at-sensor brightness temperature (Kelvin).

    Radiance L = ml * DN + al;  BT = K2 / ln(K1 / L + 1).
    Works on numpy arrays or torch tensors.
    """
    radiance = ml * dn + al
    if isinstance(dn, torch.Tensor):
        return k2 / torch.log(k1 / radiance.clamp_min(1e-6) + 1.0)
    import numpy as np
    return k2 / np.log(k1 / np.clip(radiance, 1e-6, None) + 1.0)
