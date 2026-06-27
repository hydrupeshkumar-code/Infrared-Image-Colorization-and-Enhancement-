"""API tests: validation + 404s always run; the full inference happy-path runs
only when trained checkpoints are present."""
import io
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
rasterio = pytest.importorskip("rasterio")
from fastapi.testclient import TestClient
from rasterio.crs import CRS
from rasterio.transform import from_origin

from app import app

client = TestClient(app)


def _geotiff_bytes(count=1, georef=True, size=16):
    buf = io.BytesIO()
    kw = dict(driver="GTiff", height=size, width=size, count=count, dtype="float32")
    if georef:
        kw.update(crs=CRS.from_epsg(32643), transform=from_origin(6e5, 2e6, 200, 200))
    with rasterio.open(buf, "w", **kw) as ds:
        ds.write((np.random.rand(count, size, size) * 10 + 300).astype("float32"))
    return buf.getvalue()


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_reject_non_georeferenced():
    r = client.post("/infer", files={"file": ("bad.tif", _geotiff_bytes(georef=False),
                                              "image/tiff")})
    assert r.status_code == 422 and "error" in r.json()


def test_reject_multiband():
    r = client.post("/infer", files={"file": ("rgb.tif", _geotiff_bytes(count=3),
                                              "image/tiff")})
    assert r.status_code == 422
    assert "band" in r.json()["error"].lower()


def test_unknown_job_404():
    assert client.get("/jobs/nope").status_code == 404


def test_unknown_artifact_404():
    assert client.get("/results/nope/sr_tif").status_code == 404


@pytest.mark.skipif(not Path("checkpoints/sr/best.pth").exists()
                    or not Path("checkpoints/colorize/best.pth").exists(),
                    reason="trained checkpoints not present")
def test_full_inference_happy_path():
    data = _geotiff_bytes(count=1, georef=True, size=64)
    r = client.post("/infer", files={"file": ("input.tif", data, "image/tiff")})
    assert r.status_code == 200
    job_id = r.json()["job_id"]

    rec = None
    for _ in range(120):
        rec = client.get(f"/jobs/{job_id}").json()
        if rec["status"] in ("done", "failed"):
            break
        time.sleep(0.5)
    assert rec["status"] == "done", rec

    m = rec["metrics"]
    # PSNR/SSIM need HR ground truth -> null at inference; never fabricated.
    assert m["psnr_sr"] is None and m["ssim_rgb"] is None
    # Residual-based Kelvin metrics are always available.
    assert isinstance(m["sr_rmse_k"], (int, float))

    for key in ["input_preview_png", "sr_preview_png", "rgb_preview_png",
                "residual_preview_png", "sr_tif", "rgb_tif"]:
        rr = client.get(f"/results/{job_id}/{key}")
        assert rr.status_code == 200
        assert rr.headers["content-type"] in ("image/png", "image/tiff")
