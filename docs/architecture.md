# Architecture & data flow

How the three layers of this project connect: the **pipeline** (training +
inference), the **FastAPI backend**, and the **React frontend**.

```
 Landsat-9 scenes                 ┌──────────────────────── tir (Python package) ────────────────────────┐
 B2 B3 B4 (30m) ─┐                │                                                                       │
 B10  (30m)  ────┤   prepare-data │  data/  ── preprocess ── patchify ── manifest.csv ── datasets         │
                 └──────────────► │     (merge RGB, resample 30→100/200m, geo-disjoint split)             │
                                  │                                                                       │
                                  │  models/ EDSR · SwinIR (SR)   pix2pix UNet+PatchGAN (colorize)        │
                                  │  losses/ Charbonnier · perceptual · adversarial · physics             │
                                  │  train/  train_sr ──► checkpoints/sr/best.pth                          │
                                  │          train_colorize ──► checkpoints/colorize/best.pth             │
                                  │                                                                       │
                LR TIR 200m ────► │  infer/  pipeline.run()  ── tiled + feathered blend ──► HR TIR 100m   │
                  (GeoTIFF)       │                                              └────────► RGB 100m       │
                                  │  eval/   PSNR · SSIM · FID · residual panels                           │
                                  └───────────────────────────────────┬───────────────────────────────────┘
                                                                       │  imported by
                                  ┌────────────────────────────────────▼──────────────────────────────────┐
   browser  ──upload .tif──►  app.py → tir.api  (FastAPI, :8000)                                            │
   (:5173)  ◄──job JSON────   POST /infer ─► validate (1 band + CRS) ─► ThreadPoolExecutor ─► pipeline.run  │
            ◄──PNG/TIFF────   GET /jobs/{id} (poll)   GET /results/{id}/{artifact}                          │
                                  │  computes residual = SR − bilinear-up(LR); renders 4 previews;          │
                                  │  metrics: psnr/ssim = null (no GT), sr_*_k from residual (Kelvin)       │
                                  └───────────────────────────────────────────────────────────────────────┘
                                                                       ▲
   React (Vite + Tailwind, frontend/) ───────────────────────────────┘
     api.ts (one base-URL const) · Hero + BeforeAfterSlider · ResultsDashboard (residual audit + Kelvin cards)
```

## Resolution math (the contract that links every stage)

Landsat-9 B10 is delivered resampled to 30 m. From the 30 m grid:

| Product       | Factor    | Resolution | Role                          |
|---------------|-----------|------------|-------------------------------|
| `RGB_100m`    | ×3.33     | 100 m      | colorization target           |
| `HR_TIR_100m` | ×3.33     | 100 m      | SR target (HR)                |
| `LR_TIR_200m` | ×6.67     | 200 m      | SR input (LR)                 |

Net SR factor **200 m → 100 m = 2×**. All products share a top-left origin and
CRS so patches and outputs stay pixel-aligned; output GeoTIFFs carry a
geotransform scaled for 100 m.

## Faithfulness / anti-hallucination chain

1. **Training** favours fidelity: Charbonnier pixel loss + a physics
   downsample-consistency term; adversarial weight kept low for SR.
2. **Inference** reuses the pipeline's tiled + feathered-blend path (no
   reimplementation) and the backend computes the **residual = SR − upsampled
   LR**; its Kelvin mean/RMS are the consistency metrics surfaced to the user.
3. **UI** leads with a before/after slider and a residual-audit panel + Kelvin
   metric cards so a reviewer can confirm the model sharpens what is present
   and does not invent structure. PSNR/SSIM need HR ground truth and show "—"
   at inference rather than a fabricated number.

## Where things live

| Path | Responsibility |
|------|----------------|
| `src/tir/{data,models,losses,train,eval,infer,utils}` | the ML pipeline |
| `src/tir/api/` + `app.py` | FastAPI backend (`schemas`, `jobs`, `previews`, `server`) |
| `frontend/src/` | React app (`api.ts`, `components/`, `App.tsx`) |
| `configs/` | `data` · `sr` · `colorize` · `infer` (checkpoint paths) |
| `tests/` | pipeline + API tests |
| `docs/` | this file + `api.md` |
