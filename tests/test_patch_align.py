"""LR/HR/RGB patches must be spatially co-registered (shared top-left origin)
and sized by the SR factor."""
import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.crs import CRS
from rasterio.transform import from_origin

from tir.data.patchify import patchify_scene
from tir.utils.geo import GeoRaster


def _raster(size, res, origin=(600000.0, 2000000.0), bands=1, seed=0):
    rng = np.random.default_rng(seed)
    transform = from_origin(origin[0], origin[1], res, res)
    data = rng.random((bands, size, size)).astype(np.float32)
    return GeoRaster(data, transform, CRS.from_epsg(32643), nodata=None)


def test_patches_aligned_and_sized(tmp_path):
    sr_factor, hr_patch = 2, 32
    hr = _raster(96, 100.0)          # 100 m HR/RGB grid
    rgb = _raster(96, 100.0, bands=3)
    lr = _raster(48, 200.0)          # 200 m LR grid (same origin)

    rows = patchify_scene("scene_t", lr, hr, rgb, tmp_path,
                          hr_patch=hr_patch, sr_factor=sr_factor,
                          stride=hr_patch, min_valid_frac=0.0)
    assert rows, "expected at least one patch"
    for row in rows:
        from tir.utils.geo import read_raster
        lr_p = read_raster(row["lr_tir"]); hr_p = read_raster(row["hr_tir"])
        rgb_p = read_raster(row["rgb"])
        # sizes follow the SR factor
        assert hr_p.width == hr_patch and rgb_p.width == hr_patch
        assert lr_p.width == hr_patch // sr_factor
        # co-registered: HR and RGB share origin; LR origin matches too
        assert hr_p.transform.c == rgb_p.transform.c
        assert hr_p.transform.f == rgb_p.transform.f
        assert lr_p.transform.c == pytest.approx(hr_p.transform.c)
        assert lr_p.transform.f == pytest.approx(hr_p.transform.f)
