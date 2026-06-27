"""Raster I/O must preserve CRS, geotransform and nodata; resampling must
scale the geotransform correctly while keeping the same origin."""
import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.crs import CRS
from rasterio.transform import from_origin

from tir.utils.geo import (GeoRaster, read_raster, resample_to_resolution,
                           scaled_transform, write_raster)


def _make(tmp_path):
    transform = from_origin(600000.0, 2000000.0, 30.0, 30.0)
    crs = CRS.from_epsg(32643)
    data = np.random.rand(1, 99, 99).astype(np.float32)
    return GeoRaster(data, transform, crs, nodata=-9999.0)


def test_write_read_roundtrip(tmp_path):
    r = _make(tmp_path)
    path = tmp_path / "r.tif"
    write_raster(path, r)
    back = read_raster(path)
    assert back.crs == r.crs
    assert back.transform.almost_equals(r.transform)
    assert back.nodata == r.nodata
    np.testing.assert_allclose(back.data, r.data, atol=1e-5)


def test_scaled_transform_preserves_origin(tmp_path):
    r = _make(tmp_path)
    t2 = scaled_transform(r.transform, 3.33)
    assert t2.c == r.transform.c and t2.f == r.transform.f  # same origin
    assert abs(t2.a) == pytest.approx(30.0 * 3.33, rel=1e-6)


def test_resample_resolution_and_grid(tmp_path):
    r = _make(tmp_path)
    out = resample_to_resolution(r, 3.33, "average")
    assert out.crs == r.crs
    assert abs(out.res[0]) == pytest.approx(99.9, rel=1e-3)
    assert out.transform.c == r.transform.c  # top-left origin unchanged
    assert out.height == round(99 / 3.33)
