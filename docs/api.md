# API reference

FastAPI backend wrapping the TIR SR + colorization pipeline. Implementation in
`src/tir/api/`, exposed as `app` in `app.py`.

```bash
pip install -e ".[api]"
uvicorn app:app --reload          # http://localhost:8000
```

Interactive OpenAPI docs are auto-generated at **`http://localhost:8000/docs`**
(and the raw schema at `/openapi.json`).

## Configuration (env vars — no hardcoded paths)

| Variable | Default | Meaning |
|----------|---------|---------|
| `TIR_INFER_CONFIG` | `configs/infer.yaml` | YAML with `sr_checkpoint` / `colorize_checkpoint` paths and tile settings |
| `TIR_JOBS_DIR` | `out/api_jobs` | per-job working directories |
| `TIR_SEED` | `42` | seed applied before each job for reproducibility |

CORS is restricted to `http://localhost:5173` (the Vite dev server). The job
store is in-memory and **does not survive a backend restart**.

---

## `POST /infer`

Upload a single-band, georeferenced 200 m TIR GeoTIFF. Inference runs off the
event loop (ThreadPoolExecutor); the call returns immediately.

- **Request:** `multipart/form-data`, field `file`.
- **200:** `{ "job_id": "<hex>" }`
- **422:** `{ "error": "<reason>" }` — not single-band, missing CRS, no valid
  geotransform, empty, or unreadable.

```bash
curl -F "file=@scene_lr_tir_200m.tif;type=image/tiff" http://localhost:8000/infer
# {"job_id":"2e9b26790c124beebdb1da7c598a3d49"}
```

## `GET /jobs/{job_id}`

Poll the job (the frontend polls every ~1.5 s). **404** if `job_id` is unknown.

```jsonc
{
  "job_id": "2e9b…",
  "status": "queued | running | done | failed",
  "error": null,                       // string when status == "failed"
  "metrics": {                         // null until status == "done"
    "psnr_sr": null, "ssim_sr": null,  // null at inference: no HR ground truth
    "psnr_rgb": null, "ssim_rgb": null,
    "sr_mean_bias_k": -0.0444,         // residual (SR − upsampled LR), Kelvin
    "sr_rmse_k": 3.2698
  },
  "artifacts": {                       // null until done; backend-relative URLs
    "input_preview_png": "/results/2e9b…/input_preview_png",
    "sr_preview_png":    "/results/2e9b…/sr_preview_png",
    "rgb_preview_png":   "/results/2e9b…/rgb_preview_png",
    "residual_preview_png": "/results/2e9b…/residual_preview_png",
    "sr_tif":  "/results/2e9b…/sr_tif",
    "rgb_tif": "/results/2e9b…/rgb_tif"
  }
}
```

> Metric policy: PSNR/SSIM require an HR reference and are returned as `null` at
> inference (never fabricated). `sr_mean_bias_k` / `sr_rmse_k` come from the
> SR-vs-LR residual in Kelvin and are always populated.

## `GET /results/{job_id}/{artifact}`

Stream one artifact. `artifact` ∈ `input_preview_png`, `sr_preview_png`,
`rgb_preview_png`, `residual_preview_png`, `sr_tif`, `rgb_tif`. Returns
`image/png` or `image/tiff`; **404** if the artifact or job is missing.

```bash
curl -o sr.tif http://localhost:8000/results/2e9b…/sr_tif
```

## Previews (rendered by the backend)

| Artifact | Colormap | Notes |
|----------|----------|-------|
| `input_preview_png` | `inferno` | LR TIR; vmin/vmax = 2–98 percentile of input |
| `sr_preview_png` | `inferno` | **same vmin/vmax as input** for an honest comparison |
| `rgb_preview_png` | as-is | colorized RGB in [0,1] |
| `residual_preview_png` | `RdBu_r` | symmetric, centered at 0 (blue = SR below input, red = above) |

## Errors

All error responses are structured JSON `{ "error": "<message>" }` with the
appropriate status code (422 validation, 404 not found). Job failures surface
as `status: "failed"` with the exception summary in `error`.
