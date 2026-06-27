"""Optional helper to fetch Landsat-9 Collection-2 scenes and lay them out as
``data/raw/scene_<id>/{B2,B3,B4,B10}.tif`` for ``tir-prepare-data``.

This needs USGS M2M API access (free, register at https://ers.cr.usgs.gov and
request M2M access). It uses the ``landsatxplore`` package if available; if not,
it prints manual EarthExplorer instructions and exits cleanly (no hard dep).

Usage:
    pip install landsatxplore
    export USGS_USERNAME=...   USGS_TOKEN=...
    python scripts/download_landsat.py --scenes LC09_L2SP_146039_20240115 ...

Without credentials the pipeline still runs end-to-end on the synthetic sample.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

MANUAL = """\
Manual route (no extra packages needed):
  1. Register at https://earthexplorer.usgs.gov
  2. Search dataset 'Landsat 9 OLI/TIRS C2 L2' over your regions/path-rows.
  3. Download bands SR_B2, SR_B3, SR_B4, ST_B10 for each scene.
  4. Save as data/raw/scene_00/{B2,B3,B4,B10}.tif  (rename SR_B4->B4, ST_B10->B10).
  5. Set `radiometric.level: L2` in configs/data.yaml, then run `tir-prepare-data`.
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Download Landsat-9 scenes (optional).")
    ap.add_argument("--scenes", nargs="*", default=[],
                    help="Landsat product IDs to download")
    ap.add_argument("--out-dir", default="data/raw")
    args = ap.parse_args(argv)

    try:
        from landsatxplore.api import API           # type: ignore
        from landsatxplore.earthexplorer import EarthExplorer  # type: ignore
    except Exception:
        print("landsatxplore not installed — showing manual instructions.\n")
        print(MANUAL)
        return 0

    user = os.environ.get("USGS_USERNAME")
    token = os.environ.get("USGS_TOKEN") or os.environ.get("USGS_PASSWORD")
    if not (user and token):
        print("Set USGS_USERNAME and USGS_TOKEN (M2M) env vars.\n")
        print(MANUAL)
        return 1
    if not args.scenes:
        print("No --scenes given. Example: --scenes LC09_L2SP_146039_20240115")
        return 1

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ee = EarthExplorer(user, token)
    try:
        for i, scene in enumerate(args.scenes):
            dest = out / f"scene_{i:02d}"
            dest.mkdir(parents=True, exist_ok=True)
            print(f"downloading {scene} -> {dest} …")
            ee.download(scene, output_dir=str(dest))
        print("\nDone. Unpack bands to {B2,B3,B4,B10}.tif, set radiometric.level "
              "in configs/data.yaml, then run `tir-prepare-data`.")
    finally:
        ee.logout()
    return 0


if __name__ == "__main__":
    sys.exit(main())
