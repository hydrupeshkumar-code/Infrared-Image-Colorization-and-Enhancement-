"""Qualitative visualization: side-by-side panels and residual maps."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402


def _to_display(img: np.ndarray) -> np.ndarray:
    """Normalize a (H,W) or (C,H,W) array to [0,1] for display."""
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    finite = arr[np.isfinite(arr)]
    if finite.size:
        lo, hi = float(finite.min()), float(finite.max())
        if hi > lo:
            arr = (arr - lo) / (hi - lo)
    return np.clip(arr, 0.0, 1.0)


def save_panel(images: Sequence[np.ndarray], titles: Sequence[str],
               path: str | Path) -> None:
    """Save a 1-row panel of images with titles (for hallucination audits)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.2))
    if n == 1:
        axes = [axes]
    for ax, img, title in zip(axes, images, titles):
        disp = _to_display(img)
        ax.imshow(disp, cmap=None if disp.ndim == 3 else "inferno")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_residual(pred: np.ndarray, target: np.ndarray, path: str | Path,
                  title: str = "residual |pred-target|") -> None:
    """Save an absolute-error residual map to flag invented structure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    p = _to_display(pred)
    t = _to_display(target)
    if p.ndim == 3:
        p = p.mean(-1)
    if t.ndim == 3:
        t = t.mean(-1)
    resid = np.abs(p - t)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(resid, cmap="magma", vmin=0.0, vmax=max(1e-6, resid.max()))
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
