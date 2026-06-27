"""Phase 1 orchestrator: raw Landsat-like scenes -> aligned patch pairs.

Steps (all driven by ``configs/data.yaml``):
  1. For each scene, read B2/B3/B4/B10 (30 m).
  2. Merge B2/B3/B4 -> 3-band RGB; apply optional 2-98% stretch for the
     visualization-grade target (radiometric kept if stretch disabled).
  3. Resample with documented factors:
        RGB 30m  -> 100m   (x3.33, average)
        B10 30m  -> 100m   (x3.33, average)  [HR target]
        B10 30m  -> 200m   (x6.67, average)  [LR input]
  4. Patchify into aligned LR/HR/RGB tiles, skipping mostly-nodata patches.
  5. Write ``manifest.csv`` and a geographically-disjoint scene-level split.

This generates the synthetic sample first if no raw scenes are present, so the
pipeline is runnable end-to-end without any manual download.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tir.data.make_synthetic import OPTICAL_BANDS, THERMAL_BAND, generate
from tir.data.patchify import patchify_scene
from tir.data.radiometric import apply_optical, apply_thermal, normalize_level
from tir.utils.config import load_config
from tir.utils.geo import (GeoRaster, percentile_stretch, read_raster,
                           resample_to_resolution, write_raster)
from tir.utils.logging import get_logger
from tir.utils.seed import seed_everything

LOG = get_logger("preprocess")


def build_rgb(scene_dir: Path, stretch: bool, low: float, high: float,
              level: str = "none") -> GeoRaster:
    """Merge B2/B3/B4 -> 3-band RGB (R,G,B order for display).

    Optical bands are converted to reflectance per ``level`` before merging.
    """
    bands = {b: read_raster(scene_dir / f"{b}.tif") for b in OPTICAL_BANDS}
    ref = bands["B4"]
    # stack as (R=B4, G=B3, B=B2), each converted to reflectance for the level
    rgb = np.concatenate([apply_optical(bands["B4"].data, level),
                          apply_optical(bands["B3"].data, level),
                          apply_optical(bands["B2"].data, level)], axis=0)
    if stretch:
        rgb = percentile_stretch(rgb, low, high)
    return GeoRaster(rgb.astype(np.float32), ref.transform, ref.crs, ref.nodata)


def split_scenes(scene_ids: list[str], ratios: tuple[float, float, float],
                 seed: int) -> dict[str, str]:
    """Assign whole scenes to train/val/test (no patch-level leakage)."""
    rng = np.random.default_rng(seed)
    ids = list(scene_ids)
    rng.shuffle(ids)
    n = len(ids)
    n_train = max(1, int(round(ratios[0] * n)))
    n_val = max(1, int(round(ratios[1] * n))) if n - n_train > 1 else 0
    assign = {}
    for i, sid in enumerate(ids):
        if i < n_train:
            assign[sid] = "train"
        elif i < n_train + n_val:
            assign[sid] = "val"
        else:
            assign[sid] = "test"
    return assign


def prepare(config_path: str | Path) -> Path:
    cfg = load_config(config_path)
    seed_everything(int(cfg.get("seed", 42)))

    raw_dir = Path(cfg["raw_dir"])
    interim_dir = Path(cfg["interim_dir"])
    out_dir = Path(cfg["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    sr_factor = int(cfg.get("sr_factor", 2))
    f_hr = float(cfg.get("downscale_hr", 3.33))   # 30m -> 100m
    f_lr = float(cfg.get("downscale_lr", 6.67))   # 30m -> 200m
    resampling = cfg.get("resampling", "average")
    hr_patch = int(cfg.get("hr_patch", 128))
    stride = cfg.get("stride")
    min_valid = float(cfg.get("min_valid_frac", 0.5))
    stretch = bool(cfg.get("rgb_stretch", True))
    low, high = cfg.get("stretch_low", 2.0), cfg.get("stretch_high", 98.0)
    level = normalize_level(cfg.get("radiometric", {}).get("level", "none")
                            if isinstance(cfg.get("radiometric"), dict) else "none")
    if level != "none":
        LOG.info("radiometric conversion: Collection-2 %s -> physical units", level)

    # Bootstrap synthetic data if no scenes present.
    scene_dirs = sorted([d for d in raw_dir.glob("scene_*") if d.is_dir()])
    if not scene_dirs:
        LOG.info("no raw scenes in %s -> generating synthetic sample", raw_dir)
        generate(raw_dir, n_scenes=int(cfg.get("synthetic_scenes", 4)),
                 size=int(cfg.get("synthetic_size", 384)),
                 seed=int(cfg.get("seed", 42)))
        scene_dirs = sorted([d for d in raw_dir.glob("scene_*") if d.is_dir()])

    all_rows: list[dict] = []
    for scene_dir in scene_dirs:
        sid = scene_dir.name
        rgb30 = build_rgb(scene_dir, stretch, low, high, level)
        tir30 = read_raster(scene_dir / f"{THERMAL_BAND}.tif")
        tir30.data = apply_thermal(tir30.data, level)  # -> Kelvin for L1/L2

        rgb100 = resample_to_resolution(rgb30, f_hr, resampling)
        hr_tir100 = resample_to_resolution(tir30, f_hr, resampling)
        lr_tir200 = resample_to_resolution(tir30, f_lr, resampling)

        # persist interim products for inspection / reuse
        write_raster(interim_dir / sid / "rgb_100m.tif", rgb100)
        write_raster(interim_dir / sid / "hr_tir_100m.tif", hr_tir100)
        write_raster(interim_dir / sid / "lr_tir_200m.tif", lr_tir200)

        rows = patchify_scene(sid, lr_tir200, hr_tir100, rgb100, out_dir,
                              hr_patch=hr_patch, sr_factor=sr_factor,
                              stride=stride, min_valid_frac=min_valid)
        all_rows.extend(rows)

    if not all_rows:
        raise RuntimeError("No patches produced. Check patch size vs scene size.")

    assign = split_scenes([d.name for d in scene_dirs],
                          tuple(cfg.get("split", [0.6, 0.2, 0.2])),
                          int(cfg.get("seed", 42)))
    for row in all_rows:
        row["split"] = assign[row["scene"]]

    manifest = pd.DataFrame(all_rows)
    manifest_path = out_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    LOG.info("wrote %d patches across %d scenes -> %s",
             len(manifest), len(scene_dirs), manifest_path)
    LOG.info("split counts: %s", manifest["split"].value_counts().to_dict())
    return manifest_path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Prepare TIR SR/colorization patch dataset.")
    ap.add_argument("--config", default="configs/data.yaml")
    args = ap.parse_args(argv)
    prepare(args.config)


if __name__ == "__main__":
    main()
