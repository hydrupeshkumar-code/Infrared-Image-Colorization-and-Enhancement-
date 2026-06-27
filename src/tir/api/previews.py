"""Render the four audit previews and compute the SR-vs-LR residual.

The residual = SR_TIR(100m) - LR_TIR(200m) bilinearly upsampled to 100m. Its
mean / RMS (in Kelvin) are honest radiometric-consistency metrics available at
inference without any HR ground truth: a faithful SR collapses back to the
observed LR, so a near-zero residual is the anti-hallucination signal.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def upsample_bilinear(arr: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """Bilinearly resize a (1,h,w) array to (1,H,W) without torch import cost."""
    import torch
    t = torch.from_numpy(np.ascontiguousarray(arr))[None].float()
    up = torch.nn.functional.interpolate(t, size=out_hw, mode="bilinear",
                                         align_corners=False)
    return up[0].numpy()


def compute_residual(sr_tir: np.ndarray, lr_tir: np.ndarray) -> np.ndarray:
    """SR(100m) minus LR(200m) upsampled to the SR grid. Shapes (1,H,W)."""
    lr_up = upsample_bilinear(lr_tir, sr_tir.shape[-2:])
    return sr_tir - lr_up


def _save_single(fig_arr: np.ndarray, path: Path, cmap: str | None,
                 vmin=None, vmax=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = fig_arr.shape[-2:]
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    if fig_arr.ndim == 3 and fig_arr.shape[0] == 3:      # RGB
        ax.imshow(np.clip(np.transpose(fig_arr, (1, 2, 0)), 0, 1))
    else:                                                 # single band
        ax.imshow(fig_arr[0] if fig_arr.ndim == 3 else fig_arr,
                  cmap=cmap, vmin=vmin, vmax=vmax)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def render_previews(lr_tir: np.ndarray, sr_tir: np.ndarray, rgb: np.ndarray,
                    residual: np.ndarray, out_dir: Path,
                    thermal_cmap: str = "inferno") -> dict[str, str]:
    """Write the four PNG previews. Returns {name: filename}.

    Input and SR share vmin/vmax (2-98 percentile of the LR input) for an
    honest side-by-side; the residual uses a symmetric diverging map at 0.
    """
    out_dir = Path(out_dir)
    finite = lr_tir[np.isfinite(lr_tir)]
    vmin, vmax = np.percentile(finite, [2, 98]) if finite.size else (0.0, 1.0)

    _save_single(lr_tir, out_dir / "input_preview.png", thermal_cmap, vmin, vmax)
    _save_single(sr_tir, out_dir / "sr_preview.png", thermal_cmap, vmin, vmax)
    _save_single(np.clip(rgb, 0, 1), out_dir / "rgb_preview.png", None)

    rabs = float(np.percentile(np.abs(residual), 99)) or 1e-6
    _save_single(residual, out_dir / "residual_preview.png", "RdBu_r",
                 vmin=-rabs, vmax=rabs)

    return {
        "input_preview_png": "input_preview.png",
        "sr_preview_png": "sr_preview.png",
        "rgb_preview_png": "rgb_preview.png",
        "residual_preview_png": "residual_preview.png",
    }


def residual_metrics_k(residual: np.ndarray) -> tuple[float, float]:
    """Mean bias and RMSE of the residual, in Kelvin (TIR is brightness temp)."""
    r = residual[np.isfinite(residual)]
    if r.size == 0:
        return 0.0, 0.0
    return float(r.mean()), float(np.sqrt(np.mean(r ** 2)))
