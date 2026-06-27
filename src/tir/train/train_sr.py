"""Phase 2 training: TIR super-resolution (200 m -> 100 m, 2x).

Fidelity-first: Charbonnier pixel loss (+ optional VGG perceptual and a
physics downsample-consistency term). Logs PSNR/SSIM each epoch, checkpoints
the best model, and reports mean-bias / RMSE in original Kelvin units.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tir.data.datasets import SRDataset, compute_tir_stats
from tir.losses.physics import DownsampleConsistencyLoss
from tir.losses.pixel import CharbonnierLoss
from tir.models.sr_model import build_sr_model
from tir.utils.config import load_config
from tir.utils.logging import CSVMetricLogger, get_logger
from tir.utils.seed import seed_everything

LOG = get_logger("train_sr")


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@torch.no_grad()
def evaluate_sr(model, loader, device, ds: SRDataset):
    from torchmetrics.functional import (peak_signal_noise_ratio as psnr,
                                         structural_similarity_index_measure as ssim)
    model.eval()
    psnr_sum = ssim_sum = bias_sum = sq_sum = n = 0.0
    for batch in loader:
        lr, hr = batch["lr"].to(device), batch["hr"].to(device)
        sr = model(lr)
        sr = sr[..., : hr.shape[-2], : hr.shape[-1]]
        # metrics in normalized space (data_range ~ standardized)
        dr = float(hr.max() - hr.min()) or 1.0
        psnr_sum += float(psnr(sr, hr, data_range=dr)) * lr.size(0)
        ssim_sum += float(ssim(sr, hr, data_range=dr)) * lr.size(0)
        # radiometric error in Kelvin
        sr_k = ds.denormalize(sr); hr_k = ds.denormalize(hr)
        bias_sum += float((sr_k - hr_k).mean()) * lr.size(0)
        sq_sum += float(((sr_k - hr_k) ** 2).mean()) * lr.size(0)
        n += lr.size(0)
    return {"psnr": psnr_sum / n, "ssim": ssim_sum / n,
            "mean_bias_K": bias_sum / n, "rmse_K": (sq_sum / n) ** 0.5}


def train(config_path: str, max_steps: int | None = None) -> Path:
    cfg = load_config(config_path)
    seed_everything(int(cfg.get("seed", 42)))
    device = resolve_device(cfg.get("device", "auto"))
    out_dir = Path(cfg["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)

    mean = cfg.get("tir_mean", "auto"); std = cfg.get("tir_std", "auto")
    if mean == "auto" or std == "auto":
        mean, std = compute_tir_stats(cfg["manifest"], "train")
        LOG.info("TIR stats: mean=%.2f std=%.2f K", mean, std)

    tcfg = cfg["train"]
    train_ds = SRDataset(cfg["manifest"], "train", augment=tcfg.get("augment", True),
                         tir_mean=mean, tir_std=std)
    val_ds = SRDataset(cfg["manifest"], "val", augment=False, tir_mean=mean, tir_std=std)
    train_ld = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True,
                          num_workers=tcfg.get("num_workers", 0), drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=tcfg["batch_size"], shuffle=False)

    model = build_sr_model(dict(cfg["model"])).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=tcfg["lr"])

    charb = CharbonnierLoss()
    lcfg = cfg["loss"]
    use_perc = lcfg.get("perceptual", 0.0) > 0
    perceptual = None
    if use_perc:
        from tir.losses.perceptual import VGGPerceptual
        perceptual = VGGPerceptual().to(device)
    phys = DownsampleConsistencyLoss(scale=cfg["model"].get("scale", 2)) \
        if lcfg.get("physics_consistency", 0.0) > 0 else None

    metric_log = CSVMetricLogger(out_dir / "metrics.csv")
    best_psnr = -1.0; best_path = out_dir / "best.pth"
    step = 0; max_steps = max_steps if max_steps is not None else tcfg.get("max_steps")

    for epoch in range(tcfg["epochs"]):
        model.train()
        for batch in train_ld:
            lr, hr = batch["lr"].to(device), batch["hr"].to(device)
            sr = model(lr)[..., : hr.shape[-2], : hr.shape[-1]]
            loss = lcfg.get("charbonnier", 1.0) * charb(sr, hr)
            if perceptual is not None:
                loss = loss + lcfg["perceptual"] * perceptual(sr, hr)
            if phys is not None:
                loss = loss + lcfg["physics_consistency"] * phys(sr, lr)
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1
            if max_steps and step >= max_steps:
                break
        metrics = evaluate_sr(model, val_ld, device, val_ds)
        metrics.update({"epoch": epoch, "step": step, "loss": float(loss)})
        metric_log.log(metrics)
        LOG.info("epoch %d | loss %.4f | PSNR %.2f SSIM %.3f | bias %.2fK RMSE %.2fK",
                 epoch, float(loss), metrics["psnr"], metrics["ssim"],
                 metrics["mean_bias_K"], metrics["rmse_K"])
        if metrics["psnr"] > best_psnr:
            best_psnr = metrics["psnr"]
            torch.save({"model": model.state_dict(), "cfg": dict(cfg["model"]),
                        "tir_mean": mean, "tir_std": std, "metrics": metrics},
                       best_path)
        if max_steps and step >= max_steps:
            LOG.info("hit max_steps=%d; stopping", max_steps)
            break

    LOG.info("best PSNR %.2f -> %s", best_psnr, best_path)
    return best_path


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Train TIR super-resolution model.")
    ap.add_argument("--config", default="configs/sr.yaml")
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args(argv)
    train(args.config, args.max_steps)


if __name__ == "__main__":
    main()
