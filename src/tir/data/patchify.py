"""Cut co-registered rasters into aligned patches and write a manifest.

The three products per scene share the same top-left origin and differ only
in resolution:
  * LR_TIR_200m  : factor 6.67 from 30 m
  * HR_TIR_100m  : factor 3.33 from 30 m   (2x of LR)
  * RGB_100m     : factor 3.33 from 30 m

Patches are indexed on the 100 m grid (HR/RGB) of size ``hr_patch``; the
matching LR patch is ``hr_patch // sr_factor``. A patch is kept only if its
valid (non-nodata) fraction exceeds ``min_valid_frac``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from affine import Affine

from tir.utils.geo import GeoRaster, write_raster
from tir.utils.logging import get_logger

LOG = get_logger("patchify")


def _window(raster: GeoRaster, row: int, col: int, size: int) -> GeoRaster:
    """Extract a (row,col)->(row+size,col+size) window keeping georeferencing."""
    sub = raster.data[:, row:row + size, col:col + size]
    # transform of the sub-window: shift origin by (col, row) pixels
    transform = raster.transform * Affine.translation(col, row)
    return GeoRaster(sub, transform, raster.crs, raster.nodata)


def patchify_scene(
    scene_id: str,
    lr_tir: GeoRaster,
    hr_tir: GeoRaster,
    rgb: GeoRaster,
    out_dir: Path,
    hr_patch: int = 128,
    sr_factor: int = 2,
    stride: int | None = None,
    min_valid_frac: float = 0.5,
) -> list[dict]:
    """Write aligned LR/HR/RGB patches for one scene; return manifest rows."""
    lr_patch = hr_patch // sr_factor
    stride = stride or hr_patch
    rows: list[dict] = []

    h = min(hr_tir.height, rgb.height)
    w = min(hr_tir.width, rgb.width)

    patch_idx = 0
    for r in range(0, h - hr_patch + 1, stride):
        for c in range(0, w - hr_patch + 1, stride):
            lr_r, lr_c = r // sr_factor, c // sr_factor
            hr_win = _window(hr_tir, r, c, hr_patch)
            rgb_win = _window(rgb, r, c, hr_patch)
            lr_win = _window(lr_tir, lr_r, lr_c, lr_patch)
            if lr_win.width != lr_patch or lr_win.height != lr_patch:
                continue

            valid = hr_win.nodata_mask().mean()
            if valid < min_valid_frac:
                continue

            stem = f"{scene_id}_p{patch_idx:04d}"
            paths = {
                "lr_tir": out_dir / "lr_tir" / f"{stem}.tif",
                "hr_tir": out_dir / "hr_tir" / f"{stem}.tif",
                "rgb": out_dir / "rgb" / f"{stem}.tif",
            }
            write_raster(paths["lr_tir"], lr_win)
            write_raster(paths["hr_tir"], hr_win)
            write_raster(paths["rgb"], rgb_win)

            t = hr_win.transform
            rows.append({
                "scene": scene_id,
                "patch": stem,
                "lr_tir": str(paths["lr_tir"]),
                "hr_tir": str(paths["hr_tir"]),
                "rgb": str(paths["rgb"]),
                "row": r, "col": c,
                "origin_x": t.c, "origin_y": t.f,
                "hr_res": abs(t.a), "valid_frac": round(float(valid), 4),
            })
            patch_idx += 1

    LOG.info("scene %s -> %d patches", scene_id, len(rows))
    return rows
