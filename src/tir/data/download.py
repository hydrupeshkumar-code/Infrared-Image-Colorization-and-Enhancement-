"""How to obtain real Landsat-9 Collection-2 scenes (B2, B3, B4, B10).

This module is documentation-first: real acquisition needs USGS credentials,
so we describe the two supported routes rather than shipping secrets. After
downloading, place each scene's bands as::

    data/raw/scene_<id>/B2.tif
    data/raw/scene_<id>/B3.tif
    data/raw/scene_<id>/B4.tif
    data/raw/scene_<id>/B10.tif

then run ``tir-prepare-data``. If ``data/raw`` is empty, the pipeline falls
back to a synthetic sample so everything still runs offline.

Routes
------
1. EarthExplorer (manual): https://earthexplorer.usgs.gov
   - Dataset: "Landsat 9 OLI/TIRS C2 L1" (or L2 surface reflectance).
   - Download bands B2, B3, B4 (30 m) and B10 (TIRS, delivered at 30 m).

2. M2M API (scriptable): https://m2m.cr.usgs.gov/
   - Register, request access, then use the JSON API or the ``landsatxplore``
     / ``usgs`` Python packages to search + download by path/row + date.

L1 vs L2
--------
* L1 (DN): convert to TOA reflectance / at-sensor brightness temperature using
  the MTL metadata (see ``tir.losses.physics.dn_to_brightness_temperature``).
* L2: bands are already surface reflectance / surface temperature — preferred.
"""
from __future__ import annotations

INSTRUCTIONS = __doc__


def main(argv=None) -> None:  # pragma: no cover - informational entry point
    print(INSTRUCTIONS)


if __name__ == "__main__":
    main()
