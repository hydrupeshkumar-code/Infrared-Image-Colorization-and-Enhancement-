"""In-memory job store + background worker.

Inference is CPU/GPU-bound, so it runs in a ThreadPoolExecutor off the event
loop. The store is a plain dict {job_id: JobState}; it is intentionally simple
for a demo and does NOT survive a process restart.
"""
from __future__ import annotations

import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from tir.api.previews import (compute_residual, render_previews,
                              residual_metrics_k)
from tir.api.schemas import Artifacts, JobRecord, Metrics
from tir.infer.pipeline import run as run_pipeline
from tir.utils.geo import read_raster
from tir.utils.logging import get_logger
from tir.utils.seed import seed_everything

LOG = get_logger("api.jobs")

# artifact key -> filename on disk in the job dir
ARTIFACT_FILES = {
    "input_preview_png": "input_preview.png",
    "sr_preview_png": "sr_preview.png",
    "rgb_preview_png": "rgb_preview.png",
    "residual_preview_png": "residual_preview.png",
    "sr_tif": "HR_TIR_100m.tif",
    "rgb_tif": "RGB_100m.tif",
}


@dataclass
class JobState:
    job_id: str
    job_dir: Path
    status: str = "queued"
    error: Optional[str] = None
    metrics: Optional[Metrics] = None
    artifacts: Optional[Artifacts] = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


class JobStore:
    def __init__(self, base_dir: Path, config_path: str, seed: int = 42,
                 max_workers: int = 2):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.config_path = config_path
        self.seed = seed
        self._jobs: dict[str, JobState] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=max_workers)

    # -- accessors -------------------------------------------------------- #
    def get(self, job_id: str) -> Optional[JobState]:
        with self._lock:
            return self._jobs.get(job_id)

    def artifact_path(self, job_id: str, artifact: str) -> Optional[Path]:
        if artifact not in ARTIFACT_FILES:
            return None
        job = self.get(job_id)
        if job is None:
            return None
        path = job.job_dir / ARTIFACT_FILES[artifact]
        return path if path.exists() else None

    def to_record(self, job: JobState) -> JobRecord:
        return JobRecord(job_id=job.job_id, status=job.status, error=job.error,
                         metrics=job.metrics, artifacts=job.artifacts)

    # -- lifecycle -------------------------------------------------------- #
    def create(self, upload_bytes: bytes, filename: str) -> str:
        job_id = uuid.uuid4().hex
        job_dir = self.base_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(filename).suffix or ".tif"
        input_path = job_dir / f"input{suffix}"
        input_path.write_bytes(upload_bytes)

        job = JobState(job_id=job_id, job_dir=job_dir)
        with self._lock:
            self._jobs[job_id] = job
        self._pool.submit(self._run, job_id, input_path)
        return job_id

    def _artifacts_for(self, job_id: str) -> Artifacts:
        base = f"/results/{job_id}"
        return Artifacts(**{key: f"{base}/{key}" for key in ARTIFACT_FILES})

    def _run(self, job_id: str, input_path: Path) -> None:
        job = self.get(job_id)
        if job is None:
            return
        try:
            with job._lock:
                job.status = "running"
            seed_everything(self.seed)

            # Reuse the pipeline's existing tiled + feathered-blend inference.
            result = run_pipeline(str(input_path), str(job.job_dir),
                                  self.config_path, stages="both")
            outs = result["outputs"]

            lr = read_raster(str(input_path)).data.astype(np.float32)
            sr = read_raster(outs["hr_tir"]).data.astype(np.float32)
            rgb = read_raster(outs["rgb"]).data.astype(np.float32)

            residual = compute_residual(sr, lr)
            render_previews(lr, sr, rgb, residual, job.job_dir)
            bias_k, rmse_k = residual_metrics_k(residual)

            # psnr/ssim need HR ground truth, unavailable at inference -> None.
            metrics = Metrics(
                psnr_sr=_maybe(result, "psnr_sr"),
                ssim_sr=_maybe(result, "ssim_sr"),
                psnr_rgb=_maybe(result, "psnr_rgb"),
                ssim_rgb=_maybe(result, "ssim_rgb"),
                sr_mean_bias_k=round(bias_k, 4),
                sr_rmse_k=round(rmse_k, 4),
            )
            with job._lock:
                job.metrics = metrics
                job.artifacts = self._artifacts_for(job_id)
                job.status = "done"
            LOG.info("job %s done | bias %.3fK rmse %.3fK", job_id, bias_k, rmse_k)
        except Exception as exc:  # noqa: BLE001 - report any failure to client
            LOG.error("job %s failed: %s", job_id, exc)
            LOG.debug(traceback.format_exc())
            with job._lock:
                job.status = "failed"
                job.error = f"{type(exc).__name__}: {exc}"


def _maybe(result: dict, key: str) -> Optional[float]:
    """Pull a metric from the pipeline result if present, else None."""
    val = result.get("metrics", {}).get(key) if isinstance(result, dict) else None
    return float(val) if isinstance(val, (int, float)) else None
