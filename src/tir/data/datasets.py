"""PyTorch datasets for the two stages, driven by ``manifest.csv``.

* :class:`SRDataset`        LR_TIR_200m -> HR_TIR_100m  (single channel)
* :class:`ColorizeDataset`  HR_TIR_100m -> RGB_100m     (1 -> 3 channels)

Normalization keeps radiometry meaningful: TIR is standardized with dataset
mean/std (brightness-temperature Kelvin), RGB is assumed already in [0,1]
(stretched) and mapped to [-1, 1] for the tanh generator. Augmentations are
restricted to flips / 90-degree rotations so radiometry is never altered.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from tir.utils.geo import read_raster


def _augment(*arrays: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    """Apply the same flip/rot90 to all arrays (shape C,H,W)."""
    k = int(rng.integers(0, 4))
    do_h = bool(rng.integers(0, 2))
    do_v = bool(rng.integers(0, 2))
    out = []
    for a in arrays:
        a = np.rot90(a, k=k, axes=(1, 2))
        if do_h:
            a = a[:, :, ::-1]
        if do_v:
            a = a[:, ::-1, :]
        out.append(np.ascontiguousarray(a))
    return tuple(out)


class _BaseManifestDataset(Dataset):
    def __init__(self, manifest: str | Path, split: str = "train",
                 augment: bool = False, seed: int = 42):
        df = pd.read_csv(manifest)
        if split != "all":
            df = df[df["split"] == split].reset_index(drop=True)
        self.df = df
        self.split = split
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.df)


class SRDataset(_BaseManifestDataset):
    """LR_TIR (1,h,w) -> HR_TIR (1,H,W). Standardized with tir_mean/tir_std."""

    def __init__(self, manifest, split="train", augment=False,
                 tir_mean: float = 300.0, tir_std: float = 15.0, seed: int = 42):
        super().__init__(manifest, split, augment, seed)
        self.tir_mean = tir_mean
        self.tir_std = tir_std

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.tir_mean) / self.tir_std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.tir_std + self.tir_mean

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        lr = read_raster(row["lr_tir"]).data.astype(np.float32)
        hr = read_raster(row["hr_tir"]).data.astype(np.float32)
        lr = self.normalize(lr)
        hr = self.normalize(hr)
        if self.augment:
            lr, hr = _augment(lr, hr, rng=self.rng)
        return {"lr": torch.from_numpy(lr), "hr": torch.from_numpy(hr),
                "patch": row["patch"]}


class ColorizeDataset(_BaseManifestDataset):
    """TIR (1,H,W) -> RGB (3,H,W). RGB mapped to [-1,1]; TIR standardized.

    ``tir_source`` selects the conditioning input: 'hr' uses the real 100 m
    HR TIR (clean ablation); 'lr_up' uses the LR TIR bilinearly upsampled to
    HR size (closer to the composed-pipeline distribution).
    """

    def __init__(self, manifest, split="train", augment=False,
                 tir_mean: float = 300.0, tir_std: float = 15.0,
                 tir_source: str = "hr", seed: int = 42):
        super().__init__(manifest, split, augment, seed)
        self.tir_mean = tir_mean
        self.tir_std = tir_std
        assert tir_source in ("hr", "lr_up")
        self.tir_source = tir_source

    def normalize_tir(self, x: np.ndarray) -> np.ndarray:
        return (x - self.tir_mean) / self.tir_std

    @staticmethod
    def normalize_rgb(x: np.ndarray) -> np.ndarray:
        return x * 2.0 - 1.0  # [0,1] -> [-1,1]

    @staticmethod
    def denormalize_rgb(x: torch.Tensor) -> torch.Tensor:
        return ((x + 1.0) / 2.0).clamp(0.0, 1.0)

    def _load_tir(self, row) -> np.ndarray:
        if self.tir_source == "hr":
            return read_raster(row["hr_tir"]).data.astype(np.float32)
        lr = read_raster(row["lr_tir"]).data.astype(np.float32)
        t = torch.from_numpy(lr)[None]
        up = torch.nn.functional.interpolate(
            t, size=read_raster(row["hr_tir"]).data.shape[1:],
            mode="bilinear", align_corners=False)
        return up[0].numpy()

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        tir = self.normalize_tir(self._load_tir(row))
        rgb = read_raster(row["rgb"]).data.astype(np.float32)
        rgb = self.normalize_rgb(np.clip(rgb, 0.0, 1.0))
        if self.augment:
            tir, rgb = _augment(tir, rgb, rng=self.rng)
        return {"tir": torch.from_numpy(tir), "rgb": torch.from_numpy(rgb),
                "patch": row["patch"]}


def compute_tir_stats(manifest: str | Path, split: str = "train",
                      max_patches: int = 200) -> tuple[float, float]:
    """Estimate TIR mean/std (Kelvin) over the train split for normalization."""
    df = pd.read_csv(manifest)
    df = df[df["split"] == split]
    vals = []
    for _, row in df.head(max_patches).iterrows():
        vals.append(read_raster(row["hr_tir"]).data.ravel())
    if not vals:
        return 300.0, 15.0
    arr = np.concatenate(vals)
    return float(arr.mean()), float(arr.std() + 1e-6)
