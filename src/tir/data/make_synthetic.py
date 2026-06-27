"""Generate tiny synthetic Landsat-9-like scenes so the whole pipeline runs
offline (no USGS download required).

Each "scene" is a set of single-band GeoTIFFs (B2, B3, B4, B10) at a 30 m
grid with a valid CRS and geotransform. The thermal band (B10) is built to be
spatially correlated with the optical bands so colorization is learnable on
the toy data. Scenes are placed at distinct geographic origins so the
geographically-disjoint split is meaningful.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tir.utils.geo import GeoRaster, write_raster
from tir.utils.logging import get_logger
from tir.utils.seed import seed_everything

try:
    from rasterio.crs import CRS
    from rasterio.transform import from_origin
except Exception:  # pragma: no cover
    CRS = None
    from_origin = None

LOG = get_logger("make_synthetic")

# Landsat band names used throughout the project.
OPTICAL_BANDS = ["B2", "B3", "B4"]  # blue, green, red
THERMAL_BAND = "B10"


def _smooth_field(rng: np.random.Generator, h: int, w: int, scale: int) -> np.ndarray:
    """Low-frequency random field via upsampling a coarse grid (no SciPy)."""
    ch = max(2, h // scale)
    cw = max(2, w // scale)
    coarse = rng.standard_normal((ch, cw)).astype(np.float32)
    # bilinear upsample with numpy
    ys = np.linspace(0, ch - 1, h)
    xs = np.linspace(0, cw - 1, w)
    y0 = np.floor(ys).astype(int); y1 = np.minimum(y0 + 1, ch - 1)
    x0 = np.floor(xs).astype(int); x1 = np.minimum(x0 + 1, cw - 1)
    wy = (ys - y0)[:, None]; wx = (xs - x0)[None, :]
    top = coarse[y0][:, x0] * (1 - wx) + coarse[y0][:, x1] * wx
    bot = coarse[y1][:, x0] * (1 - wx) + coarse[y1][:, x1] * wx
    return top * (1 - wy) + bot * wy


def make_scene(rng: np.random.Generator, size: int, origin_x: float,
               origin_y: float, res: float = 30.0) -> dict[str, GeoRaster]:
    """Build one synthetic scene as a dict of band -> GeoRaster (30 m)."""
    if from_origin is None:
        raise ImportError("rasterio is required to write synthetic scenes.")
    transform = from_origin(origin_x, origin_y, res, res)
    crs = CRS.from_epsg(32643)  # UTM 43N (typical for India / ISRO AOIs)

    base = _smooth_field(rng, size, size, scale=8)
    veg = _smooth_field(rng, size, size, scale=5)
    water = (_smooth_field(rng, size, size, scale=10) > 1.0).astype(np.float32)

    # Optical reflectance-like bands in [0, 1].
    blue = np.clip(0.25 + 0.15 * base - 0.1 * veg + 0.2 * water, 0, 1)
    green = np.clip(0.30 + 0.10 * base + 0.25 * veg + 0.05 * water, 0, 1)
    red = np.clip(0.28 + 0.12 * base - 0.05 * veg - 0.1 * water, 0, 1)

    # Thermal brightness temperature (K): water cooler, bare soil warmer,
    # correlated with optical structure -> colorization is learnable.
    bt = 300.0 + 12.0 * base - 6.0 * veg - 10.0 * water
    bt += rng.standard_normal((size, size)).astype(np.float32) * 0.3

    def gr(arr: np.ndarray) -> GeoRaster:
        return GeoRaster(arr[None].astype(np.float32), transform, crs, nodata=None)

    return {"B2": gr(blue), "B3": gr(green), "B4": gr(red), "B10": gr(bt)}


def generate(out_dir: str | Path, n_scenes: int = 4, size: int = 384,
             seed: int = 42) -> Path:
    """Write ``n_scenes`` synthetic scenes under ``out_dir/scene_XX/``."""
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Space scenes far apart geographically (disjoint regions).
    for i in range(n_scenes):
        origin_x = 600000.0 + i * 100000.0
        origin_y = 2000000.0 + i * 100000.0
        scene = make_scene(rng, size, origin_x, origin_y)
        scene_dir = out_dir / f"scene_{i:02d}"
        scene_dir.mkdir(parents=True, exist_ok=True)
        for band, raster in scene.items():
            write_raster(scene_dir / f"{band}.tif", raster)
        LOG.info("wrote synthetic %s (%dx%d) at origin (%.0f, %.0f)",
                 scene_dir.name, size, size, origin_x, origin_y)
    return out_dir


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic Landsat-like scenes.")
    ap.add_argument("--out-dir", default="data/sample/raw")
    ap.add_argument("--n-scenes", type=int, default=4)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    generate(args.out_dir, args.n_scenes, args.size, args.seed)


if __name__ == "__main__":
    main()
