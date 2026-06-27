"""End-to-end: synthetic scenes -> patches -> manifest with a disjoint split."""
import pandas as pd
import pytest
import yaml

pytest.importorskip("rasterio")

from tir.data.preprocess import prepare


def test_prepare_synthetic_end_to_end(tmp_path):
    cfg = {
        "seed": 0,
        "raw_dir": str(tmp_path / "raw"),
        "interim_dir": str(tmp_path / "interim"),
        "processed_dir": str(tmp_path / "processed"),
        "sr_factor": 2, "downscale_hr": 3.33, "downscale_lr": 6.67,
        "resampling": "average", "rgb_stretch": True,
        "hr_patch": 32, "stride": 32, "min_valid_frac": 0.0,
        "split": [0.5, 0.25, 0.25],
        "synthetic_scenes": 4, "synthetic_size": 400,
    }
    cfg_path = tmp_path / "data.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    manifest_path = prepare(cfg_path)
    df = pd.read_csv(manifest_path)
    assert len(df) > 0
    # every patch references three existing rasters
    for col in ("lr_tir", "hr_tir", "rgb"):
        assert df[col].map(lambda p: __import__("os").path.exists(p)).all()
    # geographically-disjoint split: no scene appears in two splits
    per_scene_splits = df.groupby("scene")["split"].nunique()
    assert (per_scene_splits == 1).all()
