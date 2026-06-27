"""Report the PS10 geospatial / CV tech stack so reviewers can confirm the
required libraries (GDAL, Rasterio, tifffile, OpenCV) are present.

Run with:  tir-stack-check
"""
from __future__ import annotations

from tir.utils.geo import gdal_version


def stack_versions() -> dict[str, str]:
    versions: dict[str, str] = {}

    import rasterio
    versions["Rasterio"] = rasterio.__version__
    versions["GDAL (via rasterio)"] = gdal_version()

    import tifffile
    versions["tifffile"] = tifffile.__version__

    import cv2
    versions["OpenCV"] = cv2.__version__

    import numpy
    versions["NumPy"] = numpy.__version__
    try:
        import torch
        versions["PyTorch"] = torch.__version__
        versions["CUDA available"] = str(torch.cuda.is_available())
    except Exception:
        versions["PyTorch"] = "not installed"
    return versions


def main(argv=None) -> None:  # pragma: no cover - thin CLI
    print("PS10 tech stack:")
    for name, ver in stack_versions().items():
        print(f"  {name:22s} {ver}")


if __name__ == "__main__":
    main()
