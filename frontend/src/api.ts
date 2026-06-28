// Single place to point the frontend at the backend. Swap this to redeploy.
export const API_BASE = "http://localhost:8000";

export type JobStatus = "queued" | "running" | "done" | "failed";

export interface Metrics {
  psnr_sr: number | null;
  ssim_sr: number | null;
  psnr_rgb: number | null;
  ssim_rgb: number | null;
  sr_mean_bias_k: number | null;
  sr_rmse_k: number | null;
}

export interface Artifacts {
  input_preview_png: string;
  sr_preview_png: string;
  rgb_preview_png: string;
  residual_preview_png: string;
  sr_tif: string;
  rgb_tif: string;
}

export interface JobRecord {
  job_id: string;
  status: JobStatus;
  error: string | null;
  metrics: Metrics | null;
  artifacts: Artifacts | null;
}

/** Resolve a backend-relative artifact URL (e.g. "/results/..") to an absolute one. */
export const asset = (url: string): string =>
  url.startsWith("http") ? url : `${API_BASE}${url}`;

export interface HealthStatus {
  status: string;
  checkpoints_ready: boolean;
  missing_checkpoints: string[];
}

/** GET /health — reports whether the backend's model checkpoints are present.
 * Lets the UI warn upfront instead of failing a job after upload. */
export async function getHealth(): Promise<HealthStatus> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error(`Health check failed (HTTP ${res.status}).`);
  return (await res.json()) as HealthStatus;
}

export class InferError extends Error {}

/** POST a single-band TIR GeoTIFF; returns the job_id. Throws InferError on 422. */
export async function postInfer(file: File): Promise<string> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/infer`, { method: "POST", body: form });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new InferError(data?.error || `Upload failed (HTTP ${res.status}).`);
  }
  return data.job_id as string;
}

export async function getJob(jobId: string): Promise<JobRecord> {
  const res = await fetch(`${API_BASE}/jobs/${jobId}`);
  if (!res.ok) throw new Error(`Job lookup failed (HTTP ${res.status}).`);
  return (await res.json()) as JobRecord;
}
