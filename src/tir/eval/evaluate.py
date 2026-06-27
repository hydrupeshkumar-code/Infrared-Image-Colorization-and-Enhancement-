"""Phase 5 evaluation on the held-out test split.

Computes PSNR/SSIM (SR TIR vs HR-GT; predicted RGB vs reference RGB), FID for
the RGB distribution, and per-tile inference time (mean +/- std). Produces
qualitative panels (LR | SR | HR-GT | pred RGB | ref RGB) and residual maps so
reviewers can audit for hallucination, and writes a metrics summary.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tir.eval.metrics import compute_fid, psnr, ssim
from tir.infer.pipeline import TIRPipeline
from tir.train.train_sr import resolve_device
from tir.utils.config import load_config
from tir.utils.geo import read_raster
from tir.utils.logging import get_logger
from tir.utils.viz import save_panel, save_residual

LOG = get_logger("evaluate")


def evaluate(cfg_path: str) -> dict:
    cfg = load_config(cfg_path)
    device = resolve_device(cfg.get("device", "auto"))
    ecfg = cfg["eval"]; tcfg = cfg["tile"]
    out_dir = Path(ecfg["out_dir"]); (out_dir / "panels").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(cfg["manifest"])
    df = df[df["split"] == ecfg.get("split", "test")].reset_index(drop=True)
    if df.empty:
        raise RuntimeError("Test split is empty.")

    pipe = TIRPipeline(cfg.get("sr_checkpoint"), cfg.get("colorize_checkpoint"),
                       device, amp=tcfg.get("amp", False))

    sr_psnr, sr_ssim, rgb_psnr, rgb_ssim = [], [], [], []
    tile_times: list[float] = []
    fake_rgbs, real_rgbs = [], []
    n_panels = int(ecfg.get("num_panels", 8))

    for i, row in df.iterrows():
        lr = read_raster(row["lr_tir"]).data.astype(np.float32)
        hr_gt = read_raster(row["hr_tir"]).data.astype(np.float32)
        rgb_gt = np.clip(read_raster(row["rgb"]).data.astype(np.float32), 0, 1)

        t0 = time.perf_counter()
        sr, _ = pipe.super_resolve(lr, tcfg["size"], tcfg["overlap"],
                                   tcfg.get("batch_size", 4))
        rgb, _ = pipe.colorize(sr, tcfg["size"] * pipe.scale,
                               tcfg["overlap"] * pipe.scale, tcfg.get("batch_size", 4))
        tile_times.append(time.perf_counter() - t0)

        sr = sr[:, : hr_gt.shape[1], : hr_gt.shape[2]]
        rgb = rgb[:, : rgb_gt.shape[1], : rgb_gt.shape[2]]

        def T(a):
            return torch.from_numpy(a)[None]
        dr = float(hr_gt.max() - hr_gt.min()) or 1.0
        sr_psnr.append(psnr(T(sr), T(hr_gt), dr)); sr_ssim.append(ssim(T(sr), T(hr_gt), dr))
        rgb_psnr.append(psnr(T(rgb), T(rgb_gt), 1.0)); rgb_ssim.append(ssim(T(rgb), T(rgb_gt), 1.0))
        fake_rgbs.append(rgb); real_rgbs.append(rgb_gt)

        if i < n_panels:
            stem = row["patch"]
            save_panel([lr, sr, hr_gt, rgb, rgb_gt],
                       ["LR TIR 200m", "SR TIR 100m", "HR-GT TIR 100m",
                        "pred RGB 100m", "ref RGB 100m"],
                       out_dir / "panels" / f"{stem}_panel.png")
            save_residual(sr, hr_gt, out_dir / "panels" / f"{stem}_resid_tir.png",
                          "SR TIR residual")

    fid = compute_fid(fake_rgbs, real_rgbs, device="cpu") if ecfg.get("fid", True) else float("nan")
    summary = {
        "n_test": int(len(df)),
        "sr_psnr": float(np.mean(sr_psnr)), "sr_ssim": float(np.mean(sr_ssim)),
        "rgb_psnr": float(np.mean(rgb_psnr)), "rgb_ssim": float(np.mean(rgb_ssim)),
        "rgb_fid": fid,
        "tile_time_mean_s": float(np.mean(tile_times)),
        "tile_time_std_s": float(np.std(tile_times)),
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    LOG.info("=== Evaluation (test, n=%d) ===", summary["n_test"])
    LOG.info("SR  TIR : PSNR %.2f  SSIM %.3f", summary["sr_psnr"], summary["sr_ssim"])
    LOG.info("RGB     : PSNR %.2f  SSIM %.3f  FID %.2f",
             summary["rgb_psnr"], summary["rgb_ssim"], summary["rgb_fid"])
    LOG.info("inference: %.3f +/- %.3f s/tile", summary["tile_time_mean_s"],
             summary["tile_time_std_s"])
    LOG.info("panels + residual maps -> %s", out_dir / "panels")
    return summary


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Evaluate the TIR SR+colorization pipeline.")
    ap.add_argument("--config", default="configs/infer.yaml")
    args = ap.parse_args(argv)
    evaluate(args.config)


if __name__ == "__main__":
    main()
