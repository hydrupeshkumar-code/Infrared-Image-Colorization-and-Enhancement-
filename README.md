# TIR Super-Resolution + Colorization (BAH-2026 PS10, SAC-ISRO)

End-to-end, trainable deep-learning pipeline that takes a **single-channel
Thermal-IR (TIR) image at 200 m** and produces:

1. a **super-resolved TIR image at 100 m** (2× spatial SR), and
2. a **realistically colorized RGB image at 100 m** (TIR → RGB translation),

while preserving structural and spectral integrity and **avoiding
hallucination** of features not supported by the input.

> Problem statement: BAH-2026 **PS10** (SAC-ISRO).
> Data-prep mirrors [jugal-sac/IR-colorization-BAH2026](https://github.com/jugal-sac/IR-colorization-BAH2026).

---

## Why this matters

TIR bands (land-surface temperature, urban heat, water/vegetation thermal
signatures) are vital for Earth observation but are low-resolution and hard to
interpret as grayscale. This pipeline makes them **sharper** and **visually
interpretable as RGB**, with guardrails so the outputs stay faithful to the
physics of the input.

## Resolution math (defines the training pairs)

Source: Landsat-9 Collection-2 (B2/B3/B4 optical @30 m, B10 thermal delivered
at 30 m). Per co-registered tile:

| Product            | Built from           | Factor   | Role                |
|--------------------|----------------------|----------|---------------------|
| `RGB_100m`         | merge B2/B3/B4 → ×3.33| 30→100 m | colorization target |
| `HR_TIR_100m`      | B10 → ×3.33          | 30→100 m | SR target (HR)      |
| `LR_TIR_200m`      | B10 → ×6.67          | 30→200 m | SR input (LR)       |

Net SR factor: **200 m → 100 m = 2×**. All three products share a top-left
origin and CRS, so patches line up pixel-for-pixel.

---

## Install

```bash
pip install -e .          # or: pip install -r requirements.txt
tir-stack-check           # confirm the PS10 stack: GDAL / Rasterio / tifffile / OpenCV
```

PyTorch uses CUDA automatically when available (`device: auto` in the configs);
otherwise everything runs on CPU.

> **Conformance to the official PS10 problem statement** is mapped slide-by-slide
> in [`docs/ps10-conformance.md`](docs/ps10-conformance.md).

## Quickstart (runs offline on a synthetic sample)

No Landsat download is required to exercise the pipeline: if `data/raw` has no
scenes, a small **synthetic Landsat-like sample** is generated automatically.

```bash
# 1. Prepare data (synthetic fallback if data/raw is empty) -> patches + manifest.csv
tir-prepare-data   --config configs/data.yaml

# 2. Train the super-resolution stage (TIR 200m -> 100m)
tir-train-sr       --config configs/sr.yaml          # add --max-steps 20 for a smoke run

# 3. Train the colorization stage (TIR 100m -> RGB 100m)
tir-train-colorize --config configs/colorize.yaml    # add --max-steps 20 for a smoke run

# 4. Run the end-to-end pipeline on a 200m TIR GeoTIFF
tir-infer --input data/interim/scene_00/lr_tir_200m.tif --out-dir out/scene_00

# 5. Evaluate on the held-out test split (metrics + qualitative panels)
tir-evaluate --config configs/infer.yaml

# tests
pytest
```

## Using real Landsat-9 data

See `src/tir/data/download.py` (or `python scripts/download_landsat.py` /
`make download`) for the two supported routes (USGS EarthExplorer manual
download, or the scriptable M2M API). Place each scene as
`data/raw/scene_<id>/{B2,B3,B4,B10}.tif`.

For **Collection-2 Level-2** products, set `radiometric.level: L2` in
`configs/data.yaml` so the bands are converted to physical units automatically
(SR_B* → reflectance, ST_B10 → Kelvin) — this is what makes the `sr_mean_bias_k`
/ `sr_rmse_k` metrics true Kelvin. Use `L1` for Level-1 (B10 → brightness
temperature), or `none` (default) if your bands are already physical. Then run
`tir-prepare-data`.

---

## Architecture & loss rationale

**Super-resolution** (`src/tir/models/sr_model.py`, swappable via `sr.yaml`):
- `edsr` — fast residual CNN baseline (default).
- `swinir` — compact SwinIR-style window-attention transformer.
- Losses: **Charbonnier** (robust pixel fidelity) + optional VGG perceptual +
  **physics downsample-consistency**. Adversarial weight is kept **low/zero**:
  for TIR we prioritise radiometric fidelity over invented texture.

**Colorization** (`src/tir/models/colorize_model.py`, swappable via
`colorize.yaml`):
- `pix2pix` — U-Net generator + PatchGAN discriminator (conditional GAN).
- Losses: **L1 content** + **adversarial** (realism) + optional perceptual /
  SSIM. Can train on the real 100 m HR TIR (`tir_source: hr`, clean ablation)
  or on the upsampled LR TIR (`lr_up`, closer to the composed pipeline).

**Inference** (`src/tir/infer/`): tiled with overlap + **feathered (cosine)
blending** to remove seams; streamed tile-by-tile to bound memory; AMP/half
option on CUDA; SR and colorization can run jointly or independently. Outputs
are georeferenced GeoTIFFs with a correctly **scaled geotransform** for 100 m.

## Physics-informed modeling (bonus)

`src/tir/losses/physics.py`:
- **Brightness temperature** — convert B10 DN → at-sensor BT (Kelvin) via the
  Planck-based USGS relation, so SR fidelity is reported in physical units.
- **Downsample-consistency loss** — downsampling the SR output by the SR factor
  must reproduce the LR input (a cycle constraint that penalises invented
  structure). Enabled by `physics_consistency` in `sr.yaml`.
- Emissivity is assumed ≈1 (at-sensor BT); land-surface temperature would add
  an emissivity correction. These constraints are where physics most directly
  reduces hallucination.

## Anti-hallucination guardrails

- Fidelity-first losses; moderate/low adversarial weights.
- Physics downsample-consistency constraint on the SR stage.
- Every evaluation ships **side-by-side panels** (LR | SR | HR-GT | pred RGB |
  ref RGB) and **residual maps** so reviewers can audit invented structure.
- Valid CRS / scaled geotransform / nodata preserved on every raster (tested).
- **Geographically-disjoint** train/val/test split (by scene) — no leakage.
- Seeded, config-driven, pinned dependencies for reproducibility.

### Visual-inspection checklist
1. Does the SR TIR add edges/structures absent from the LR input? (check residual map)
2. Is the SR mean-bias / RMSE small in Kelvin? (reported by `train-sr` / `evaluate`)
3. Does the predicted RGB invent objects not implied by the thermal field?
4. Are land/water/vegetation thermal–color relationships plausible and consistent?
5. Any tiling seams in full-scene outputs? (should be removed by feather blending)

---

## Results

Metrics on the held-out **test** split (`tir-evaluate` writes
`out/eval/metrics.json` + panels). Reported on the synthetic sample with short
smoke training; **re-run on real Landsat scenes with full training** to fill in
production numbers and your hardware.

| Metric                     | SR (TIR) | Colorization (RGB) |
|----------------------------|----------|--------------------|
| PSNR ↑                     | _run_    | _run_              |
| SSIM ↑                     | _run_    | _run_              |
| FID ↓                      | —        | _run_              |
| SR mean-bias / RMSE (K)    | _run_    | —                  |
| Inference time per tile    | mean ± std (stated hardware) |          |

> FID needs Inception weights; it is skipped gracefully (reported as `nan`) in
> offline environments.

## Repo layout

```
app.py       FastAPI entry point (uvicorn app:app)
Makefile     install / smoke / test / backend / frontend / demo targets
configs/     data | sr | colorize | infer  (YAML, everything config-driven)
docs/        architecture.md (data flow) · api.md (API reference)
src/tir/
  data/      download, preprocess (band merge/resample), patchify, datasets, make_synthetic
  models/    sr_model (edsr/swinir), colorize_model (pix2pix), blocks/
  losses/    pixel, perceptual, adversarial, physics
  train/     train_sr, train_colorize
  eval/      metrics (PSNR/SSIM/FID), evaluate (panels + residuals)
  infer/     pipeline, tile_io (overlap + feather blending)
  api/       schemas, jobs (ThreadPoolExecutor store), previews (residual + PNGs), server
  utils/     geo (raster I/O), seed, logging, config, viz
frontend/    ChaturVyuha standalone landing site, served by Vite (see frontend/README.md)
tests/       geo I/O, patch alignment, metric/model shapes, e2e synthetic, API
```

## CLI reference

| Command             | Purpose                                              |
|---------------------|------------------------------------------------------|
| `tir-prepare-data`  | raw/synthetic scenes → aligned patch pairs + manifest|
| `tir-train-sr`      | train TIR super-resolution                           |
| `tir-train-colorize`| train TIR→RGB colorization                           |
| `tir-infer`         | 200 m TIR GeoTIFF → HR TIR + RGB GeoTIFFs            |
| `tir-evaluate`      | PSNR/SSIM/FID + per-tile time + qualitative panels   |

## Web demo — "ChaturVyuha" (FastAPI + React)

A full-stack demo wraps the pipeline: a FastAPI backend (`app.py` →
`src/tir/api/`) and a dark-themed frontend (`frontend/`) — the ChaturVyuha
standalone landing site, served by Vite — built around a
faithfulness/anti-hallucination story. See
[`docs/architecture.md`](docs/architecture.md) for the data flow and
[`docs/api.md`](docs/api.md) for the full API reference.

```bash
# 1. Backend — one command from a fresh clone (auto-bootstraps checkpoints via
#    the synthetic smoke run if none exist), then serves on :8000 (docs at /docs)
pip install -e ".[api]"
make serve                                # or: uvicorn app:app --reload (needs checkpoints)

# 2. Frontend (separate terminal). Serves the landing on :5173.
cd frontend && npm install && npm run dev # http://localhost:5173
```

The backend's CORS allow-list covers localhost/127.0.0.1 on :5173 (Vite), :4173
(`vite preview`), and :5500 (VS Code Live Server on `frontend/dist/`). For any
other origin, set `TIR_ALLOWED_ORIGINS` (comma-separated) on the backend.

The dev server serves the **ChaturVyuha standalone landing** at `/` (cinematic
hero + a working upload wired to the backend). See
[`frontend/README.md`](frontend/README.md).

`GET /health` reports `checkpoints_ready` — the landing probes it and warns up
front if checkpoints are missing, instead of failing only after an upload. (A job
that runs without checkpoints still fails with a clear "train first" message
rather than a stack trace.)

Shortcuts via the `Makefile`: `make smoke`, `make backend`, `make frontend`,
`make test`, `make demo`.

**API contract**

| Endpoint | Behaviour |
|----------|-----------|
| `POST /infer` | multipart `file` = single-band georeferenced TIR GeoTIFF. Validates `count==1` + valid CRS/transform (rasterio) → **422 `{error}`** otherwise. Saves to a per-job temp dir, runs inference **off the event loop** (ThreadPoolExecutor), returns `{job_id}` immediately. |
| `GET /jobs/{job_id}` | `{job_id, status, error, metrics, artifacts}`; **404** if unknown. `status` ∈ queued/running/done/failed. |
| `GET /results/{job_id}/{artifact}` | streams a PNG/GeoTIFF (`image/png`/`image/tiff`); **404** if missing. |

**Faithfulness by construction**
- After SR + RGB are written, the backend computes **residual = SR − bilinear-upsampled LR** and renders four previews: input (inferno), SR (inferno, *same vmin/vmax as input*), RGB (as-is), residual (`RdBu_r`, symmetric, centered at 0).
- `metrics`: `psnr_sr/ssim_sr/psnr_rgb/ssim_rgb` are **`null`** at inference (no HR ground truth — never fabricated); `sr_mean_bias_k`/`sr_rmse_k` are the residual's Kelvin mean/RMS, always available.
- Outputs preserve CRS / scaled 100 m geotransform / nodata; inference reuses the pipeline's tiled + feathered-blend path (no reimplementation).
- The frontend's residual-audit panel + Kelvin-metric cards exist to let a reviewer verify the model **sharpens what's there and doesn't invent**. Backend base URL lives in one constant (the `API_BASE` variable in the `TIR-BACKEND-WIRING` script in `frontend/public/chaturvyuha-site/ChaturVyuha (standalone).html`).

> The in-memory job store is fine for a demo but does **not** survive a backend restart.

## Limitations

- The shipped **synthetic** sample makes the pipeline runnable offline but is
  not a substitute for real Landsat radiometry — colorization realism and FID
  are only meaningful after training on real scenes.
- Colorizing thermal → RGB is inherently ill-posed; the adversarial term can
  still introduce plausible-but-unobserved color. Keep adversarial weight
  moderate and always review residual maps / panels.
- B10 is natively ~100 m (USGS resamples to 30 m); true sub-100 m thermal
  detail is limited by the sensor, so SR sharpens rather than recovers new
  thermal information.
