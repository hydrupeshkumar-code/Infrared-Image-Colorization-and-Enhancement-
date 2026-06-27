"""Landsat-9 Collection-2 radiometric conversion to physical units.

Why this matters: the Kelvin metrics (sr_mean_bias_k / sr_rmse_k) are only
physically meaningful if B10 is in brightness-temperature Kelvin. Collection-2
products ship scaled integers, so convert before building the dataset.

Levels
------
* ``none`` : assume bands are already in physical units (the offline synthetic
  sample, or data you pre-scaled). Default — keeps existing behaviour.
* ``L2``   : Collection-2 Level-2. Optical SR_B* -> surface reflectance;
  ST_B10 -> surface temperature (Kelvin) via the published scale/offset.
* ``L1``   : Collection-2 Level-1. B10 DN -> at-sensor brightness temperature
  (Planck relation, see :mod:`tir.losses.physics`). Optical L1 DN needs the MTL
  sun-elevation correction for true TOA reflectance, which requires the scene's
  metadata; here L1 optical is passed through (use L2 for colorization-grade RGB).
"""
from __future__ import annotations

import numpy as np

from tir.losses.physics import dn_to_brightness_temperature

# USGS Landsat Collection-2 Level-2 scaling (constant across scenes).
L2_REFLECTANCE_SCALE = 0.0000275
L2_REFLECTANCE_OFFSET = -0.2
L2_ST_SCALE = 0.00341802
L2_ST_OFFSET = 149.0

VALID_LEVELS = ("none", "L1", "L2")


def apply_optical(data: np.ndarray, level: str) -> np.ndarray:
    """Convert optical band(s) (B2/B3/B4) to reflectance for the given level."""
    if level == "L2":
        out = data * L2_REFLECTANCE_SCALE + L2_REFLECTANCE_OFFSET
        return np.clip(out, 0.0, 1.0).astype(np.float32)
    return data.astype(np.float32)  # none / L1 (pass-through)


def apply_thermal(data: np.ndarray, level: str) -> np.ndarray:
    """Convert thermal band B10 to brightness/surface temperature (Kelvin)."""
    if level == "L2":
        return (data * L2_ST_SCALE + L2_ST_OFFSET).astype(np.float32)
    if level == "L1":
        return dn_to_brightness_temperature(data.astype(np.float32)).astype(np.float32)
    return data.astype(np.float32)  # none (already physical)


def normalize_level(level: str | None) -> str:
    level = (level or "none").strip()
    if level not in VALID_LEVELS:
        raise ValueError(f"radiometric.level must be one of {VALID_LEVELS}, got '{level}'")
    return level
