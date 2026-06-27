"""Geospatial raster I/O that preserves CRS / geotransform / nodata.

All raster outputs in this project must keep a valid CRS and an affine
geotransform that is *correctly scaled* for the output resolution. These
helpers centralise that so every writer behaves consistently.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import rasterio
    from affine import Affine
    from rasterio.enums import Resampling
    from rasterio.warp import reproject
    _HAS_RASTERIO = True
except Exception:  # pragma: no cover - exercised only when rasterio missing
    _HAS_RASTERIO = False
    Affine = None  # type: ignore
    Resampling = None  # type: ignore


_RESAMPLING_BY_NAME = {
    "nearest": "nearest",
    "bilinear": "bilinear",
    "cubic": "cubic",
    "average": "average",
    "area": "average",  # alias: area-weighted == average for downsampling
    "lanczos": "lanczos",
}


def _require_rasterio() -> None:
    if not _HAS_RASTERIO:
        raise ImportError(
            "rasterio is required for geospatial I/O. Install with `pip install rasterio`."
        )


def gdal_version() -> str:
    """Return the GDAL version that backs rasterio.

    Rasterio embeds GDAL (it is GDAL's Python interface for this project), so
    the PS10-required GDAL engine is always present wherever rasterio is.
    """
    if not _HAS_RASTERIO:
        return "unavailable"
    try:
        from osgeo import gdal  # standalone bindings, if installed
        return gdal.__version__
    except Exception:
        return getattr(rasterio, "__gdal_version__", "unknown")


@dataclass
class GeoRaster:
    """A raster + its georeferencing metadata.

    ``data`` is shaped (bands, height, width). ``transform`` is the affine
    geotransform, ``crs`` the coordinate reference system, ``nodata`` the
    nodata value (or ``None``).
    """

    data: np.ndarray
    transform: "Affine"
    crs: object
    nodata: Optional[float] = None

    @property
    def count(self) -> int:
        return self.data.shape[0]

    @property
    def height(self) -> int:
        return self.data.shape[1]

    @property
    def width(self) -> int:
        return self.data.shape[2]

    @property
    def res(self) -> tuple[float, float]:
        """(x_res, y_res) pixel size in CRS units (y_res positive)."""
        return (abs(self.transform.a), abs(self.transform.e))

    def nodata_mask(self) -> np.ndarray:
        """Boolean mask (H, W) where True == valid (not nodata, finite)."""
        valid = np.isfinite(self.data).all(axis=0)
        if self.nodata is not None:
            valid &= (self.data != self.nodata).all(axis=0)
        return valid


def read_raster(path: str | Path) -> GeoRaster:
    """Read a GeoTIFF into a :class:`GeoRaster` (bands, H, W)."""
    _require_rasterio()
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
        return GeoRaster(data=data, transform=src.transform, crs=src.crs,
                         nodata=src.nodata)


def write_raster(path: str | Path, raster: GeoRaster, dtype: str = "float32") -> None:
    """Write a :class:`GeoRaster` to a GeoTIFF, preserving georeferencing."""
    _require_rasterio()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = raster.data.astype(dtype)
    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=raster.height, width=raster.width, count=raster.count,
        dtype=dtype, crs=raster.crs, transform=raster.transform,
        nodata=raster.nodata, compress="deflate",
    ) as dst:
        dst.write(data)


def scaled_transform(transform: "Affine", scale: float) -> "Affine":
    """Return a transform whose pixel size is multiplied by ``scale``.

    ``scale > 1`` => coarser pixels (e.g. 30m -> 100m uses scale 3.33).
    The top-left origin is preserved so grids stay co-registered.
    """
    _require_rasterio()
    return transform * Affine.scale(scale)


def resample_to_resolution(
    raster: GeoRaster,
    scale: float,
    resampling: str = "average",
) -> GeoRaster:
    """Resample a raster by a linear ``scale`` factor on pixel size.

    ``scale = 3.33`` turns 30m pixels into ~100m pixels (downsample).
    Output keeps the same CRS and a correctly scaled geotransform with the
    same top-left origin, so all products remain pixel-aligned.
    """
    _require_rasterio()
    name = _RESAMPLING_BY_NAME.get(resampling.lower())
    if name is None:
        raise ValueError(f"Unknown resampling '{resampling}'")
    resampling_enum = getattr(Resampling, name)

    dst_height = max(1, int(round(raster.height / scale)))
    dst_width = max(1, int(round(raster.width / scale)))
    dst_transform = scaled_transform(raster.transform, scale)

    dst = np.zeros((raster.count, dst_height, dst_width), dtype=np.float32)
    for b in range(raster.count):
        reproject(
            source=raster.data[b],
            destination=dst[b],
            src_transform=raster.transform, src_crs=raster.crs,
            dst_transform=dst_transform, dst_crs=raster.crs,
            resampling=resampling_enum,
            src_nodata=raster.nodata, dst_nodata=raster.nodata,
        )
    return GeoRaster(data=dst, transform=dst_transform, crs=raster.crs,
                     nodata=raster.nodata)


def percentile_stretch(
    data: np.ndarray, low: float = 2.0, high: float = 98.0,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Per-band 2-98 percentile stretch to [0, 1] for visualization-grade RGB.

    Radiometric data should be kept separately; this is for display only.
    """
    out = np.empty_like(data, dtype=np.float32)
    for b in range(data.shape[0]):
        band = data[b]
        vals = band[mask] if mask is not None else band[np.isfinite(band)]
        if vals.size == 0:
            out[b] = 0.0
            continue
        lo, hi = np.percentile(vals, [low, high])
        if hi <= lo:
            hi = lo + 1e-6
        out[b] = np.clip((band - lo) / (hi - lo), 0.0, 1.0)
    return out
