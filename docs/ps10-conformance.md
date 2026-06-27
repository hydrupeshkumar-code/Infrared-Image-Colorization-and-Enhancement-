# BAH-2026 PS10 conformance

How this repo maps to the official SAC-ISRO PS10 problem-statement slides.

## Slide 2 — Dataset and Workflow

| PS10 step | Where it's implemented |
|-----------|------------------------|
| STEP 1 · INPUT — TIR @200 m | `LR_TIR_200m` product (`src/tir/data/preprocess.py`); API accepts a single-band 200 m TIR GeoTIFF |
| STEP 2 · SUPER RESOLVE — @100 m | `src/tir/models/sr_model.py` (EDSR / SwinIR, 2×), `train_sr.py`, `infer/pipeline.py` |
| STEP 3 · COLORIZE — @100 m | `src/tir/models/colorize_model.py` (pix2pix), `train_colorize.py` |
| STEP 4 · COMPARE — original TIR @100 m + RGB reference | `src/tir/eval/evaluate.py` panels `LR | SR | HR-GT TIR | pred RGB | ref RGB` + metrics SR-vs-HR and RGB-vs-ref |
| Dataset-prep instructions (jugal-sac/IR-colorization-BAH2026) | `preprocess.py` mirrors them; documented in README + `download.py` |

## Slide 4 — Workflow Summary

| PS10 node | Implementation |
|-----------|----------------|
| Download Landsat-9 B2, B3, B4, B10 | `src/tir/data/download.py` (EarthExplorer / M2M); place under `data/raw/scene_*/` |
| Merge B2,B3,B4 → RGB | `preprocess.build_rgb()` |
| Downscale RGB ×3.3 → 100 m | `configs/data.yaml: downscale_hr: 3.33` |
| Downscale TIR B10 ×3.3 → 100 m | same factor → `HR_TIR_100m` (SR target) |
| Downscale TIR B10 ×6.7 → 200 m | `downscale_lr: 6.67` → `LR_TIR_200m` (SR input) |
| Create image patches | `src/tir/data/patchify.py` (aligned LR/HR/RGB tiles, geo-disjoint split, `manifest.csv`) |
| Super-resolution model → back to 100 m | SR stage (2×) |
| IR image colorization → colorized IR | colorization stage |

> Net SR factor 200 m → 100 m = **2×**, exactly as in the workflow.

## Slide 5 — Expected Output, Dataset & Tech stack

| PS10 requirement | Implementation |
|------------------|----------------|
| End-to-end **trained** DL pipeline | `tir-prepare-data → tir-train-sr → tir-train-colorize → tir-evaluate → tir-infer` |
| Input: single-channel TIR @200 m | enforced by API validation (`count == 1` + valid CRS) and `SRDataset` |
| Output 1: HR TIR @100 m | `HR_TIR_100m.tif` (georeferenced, scaled 100 m geotransform) |
| Output 2: colorized RGB @100 m | `RGB_100m.tif` |
| Framework: Python end-to-end | yes |
| Models: GANs / Diffusion / SoTA I2I | **pix2pix GAN** (colorize) + **SwinIR transformer** / EDSR (SR); swappable via config |
| Libraries: GDAL, Rasterio, tifffile, OpenCV | Rasterio (geo I/O) embeds **GDAL 3.10**; **OpenCV** does array resampling (`api/previews.py`); tifffile available. Verify with `tir-stack-check` |

## Slide 3 — Evaluation Parameters

| PS10 metric | Implementation |
|-------------|----------------|
| PSNR | `src/tir/eval/metrics.py::psnr` (SR-vs-HR, RGB-vs-ref) |
| SSIM | `metrics.py::ssim` |
| FID | `metrics.py::compute_fid` (pytorch-fid; skipped gracefully offline) |
| Qualitative — prevent hallucination | residual maps + 5-panel comparisons (`eval/evaluate.py`, `utils/viz.py`); UI residual-audit panel |
| Preferred — **low inference time per tile** | reported by `infer/pipeline.py` (`sec_per_tile`) and `evaluate.py` (mean ± std); optimizations: tiled batching, AMP/half, GeoTIFF streaming |
| **Bonus — physics-informed modeling** | `src/tir/losses/physics.py`: Planck brightness-temperature (B10 DN→BT) + downsample-consistency cycle loss; surfaced in the UI as Kelvin mean-bias / RMSE |

## Verifying conformance

```bash
tir-stack-check          # confirms GDAL / Rasterio / tifffile / OpenCV present
make test                # 16 tests (pipeline + API)
make smoke               # full STEP1→STEP4 run on a synthetic sample
```

## Notes for the real-data run

- For real Landsat scenes, raise the patch size in `configs/data.yaml` to
  `hr_patch: 256` (LR 128) — close to the reference repo's 256→512 SR tiles.
  The default `64` is sized for the small offline synthetic sample.
- Use Level-2 products (or convert L1 B10 to brightness temperature via
  `physics.dn_to_brightness_temperature`) so the Kelvin metrics are physical.
