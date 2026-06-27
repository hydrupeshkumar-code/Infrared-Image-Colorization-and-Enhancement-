"""FastAPI app wrapping the TIR SR + colorization pipeline.

Run with:  uvicorn app:app --reload
Config (checkpoint paths) comes from ``configs/infer.yaml`` (overridable via the
``TIR_INFER_CONFIG`` env var) — paths are never hardcoded here.
"""
from __future__ import annotations

import os
from pathlib import Path

import rasterio
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from tir.api.jobs import ARTIFACT_FILES, JobStore
from tir.api.schemas import ErrorResponse, InferResponse, JobRecord
from tir.utils.logging import get_logger

LOG = get_logger("api.server")

CONFIG_PATH = os.environ.get("TIR_INFER_CONFIG", "configs/infer.yaml")
JOBS_DIR = Path(os.environ.get("TIR_JOBS_DIR", "out/api_jobs"))
SEED = int(os.environ.get("TIR_SEED", "42"))
ALLOWED_ORIGINS = ["http://localhost:5173"]

app = FastAPI(title="ChaturVyuha TIR SR+Colorization API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

store = JobStore(JOBS_DIR, CONFIG_PATH, seed=SEED)

_missing = store.missing_checkpoints()
if _missing:
    LOG.warning("checkpoints not found: %s — train first (`make smoke` or "
                "tir-train-sr / tir-train-colorize) before running inference.",
                ", ".join(_missing))


def _validate_geotiff(data: bytes) -> str | None:
    """Return an error string if the upload is not a single-band georeferenced
    raster; ``None`` if valid."""
    try:
        with rasterio.io.MemoryFile(data) as mem, mem.open() as ds:
            if ds.count != 1:
                return (f"Expected a single-band TIR raster, got {ds.count} bands. "
                        "Upload the 200m thermal band only.")
            if ds.crs is None:
                return "Raster has no CRS — a georeferenced GeoTIFF is required."
            if ds.transform is None or ds.transform.is_identity:
                return "Raster has no valid geotransform (it is not georeferenced)."
    except rasterio.errors.RasterioIOError:
        return "Uploaded file is not a readable GeoTIFF/raster."
    return None


@app.get("/health")
def health() -> dict:
    missing = store.missing_checkpoints()
    return {"status": "ok", "checkpoints_ready": not missing,
            "missing_checkpoints": missing}


@app.post("/infer", responses={200: {"model": InferResponse},
                               422: {"model": ErrorResponse}})
async def infer(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        return JSONResponse(status_code=422, content={"error": "Empty upload."})
    error = _validate_geotiff(data)
    if error is not None:
        return JSONResponse(status_code=422, content={"error": error})
    job_id = store.create(data, file.filename or "input.tif")
    LOG.info("queued job %s (%s)", job_id, file.filename)
    return InferResponse(job_id=job_id)


@app.get("/jobs/{job_id}", response_model=JobRecord,
         responses={404: {"model": ErrorResponse}})
def get_job(job_id: str):
    job = store.get(job_id)
    if job is None:
        return JSONResponse(status_code=404, content={"error": "Unknown job_id."})
    return store.to_record(job)


@app.get("/results/{job_id}/{artifact}",
         responses={404: {"model": ErrorResponse}})
def get_result(job_id: str, artifact: str):
    if artifact not in ARTIFACT_FILES:
        return JSONResponse(status_code=404,
                            content={"error": f"Unknown artifact '{artifact}'."})
    path = store.artifact_path(job_id, artifact)
    if path is None:
        return JSONResponse(status_code=404,
                            content={"error": "Artifact not found (job missing or not ready)."})
    media = "image/png" if path.suffix == ".png" else "image/tiff"
    return FileResponse(path, media_type=media, filename=path.name)
