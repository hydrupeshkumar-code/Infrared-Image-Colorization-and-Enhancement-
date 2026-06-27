"""Pydantic response models for the inference API (mirrors the API contract)."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

JobStatus = Literal["queued", "running", "done", "failed"]


class InferResponse(BaseModel):
    job_id: str


class ErrorResponse(BaseModel):
    error: str


class Metrics(BaseModel):
    """All fields nullable: metrics needing HR ground truth are ``None`` at
    inference (we never fabricate a number). ``sr_mean_bias_k`` / ``sr_rmse_k``
    are derived from the SR-vs-LR residual in Kelvin and are always available."""

    psnr_sr: Optional[float] = None
    ssim_sr: Optional[float] = None
    psnr_rgb: Optional[float] = None
    ssim_rgb: Optional[float] = None
    sr_mean_bias_k: Optional[float] = None
    sr_rmse_k: Optional[float] = None


class Artifacts(BaseModel):
    input_preview_png: str
    sr_preview_png: str
    rgb_preview_png: str
    residual_preview_png: str
    sr_tif: str
    rgb_tif: str


class JobRecord(BaseModel):
    job_id: str
    status: JobStatus
    error: Optional[str] = None
    metrics: Optional[Metrics] = None
    artifacts: Optional[Artifacts] = None
