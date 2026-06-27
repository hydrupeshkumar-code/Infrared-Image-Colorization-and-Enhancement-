"""Phase 3 training: TIR 100 m -> RGB 100 m (pix2pix conditional GAN).

Generator loss = L1 content + adversarial (+ optional perceptual / SSIM).
Discriminator is a PatchGAN. Validates PSNR/SSIM (and FID via evaluate.py).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tir.data.datasets import ColorizeDataset, compute_tir_stats
from tir.losses.adversarial import GANLoss
from tir.models.colorize_model import build_colorize_model
from tir.train.train_sr import resolve_device
from tir.utils.config import load_config
from tir.utils.logging import CSVMetricLogger, get_logger
from tir.utils.seed import seed_everything

LOG = get_logger("train_colorize")


@torch.no_grad()
def evaluate_colorize(gen, loader, device):
    from torchmetrics.functional import (peak_signal_noise_ratio as psnr,
                                         structural_similarity_index_measure as ssim)
    gen.eval()
    psnr_sum = ssim_sum = n = 0.0
    for batch in loader:
        tir, rgb = batch["tir"].to(device), batch["rgb"].to(device)
        fake = gen(tir)[..., : rgb.shape[-2], : rgb.shape[-1]]
        fake01 = ColorizeDataset.denormalize_rgb(fake)
        rgb01 = ColorizeDataset.denormalize_rgb(rgb)
        psnr_sum += float(psnr(fake01, rgb01, data_range=1.0)) * tir.size(0)
        ssim_sum += float(ssim(fake01, rgb01, data_range=1.0)) * tir.size(0)
        n += tir.size(0)
    return {"psnr": psnr_sum / n, "ssim": ssim_sum / n}


def train(config_path: str, max_steps: int | None = None) -> Path:
    cfg = load_config(config_path)
    seed_everything(int(cfg.get("seed", 42)))
    device = resolve_device(cfg.get("device", "auto"))
    out_dir = Path(cfg["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)

    mean = cfg.get("tir_mean", "auto"); std = cfg.get("tir_std", "auto")
    if mean == "auto" or std == "auto":
        mean, std = compute_tir_stats(cfg["manifest"], "train")
        LOG.info("TIR stats: mean=%.2f std=%.2f K", mean, std)

    tcfg = cfg["train"]; src = cfg.get("tir_source", "hr")
    train_ds = ColorizeDataset(cfg["manifest"], "train", augment=tcfg.get("augment", True),
                               tir_mean=mean, tir_std=std, tir_source=src)
    val_ds = ColorizeDataset(cfg["manifest"], "val", augment=False,
                             tir_mean=mean, tir_std=std, tir_source=src)
    train_ld = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True,
                          num_workers=tcfg.get("num_workers", 0), drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=tcfg["batch_size"], shuffle=False)

    gen, disc = build_colorize_model(dict(cfg["model"]))
    gen, disc = gen.to(device), disc.to(device)
    opt_g = torch.optim.Adam(gen.parameters(), lr=tcfg["lr"],
                             betas=(tcfg.get("beta1", 0.5), 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=tcfg["lr"],
                             betas=(tcfg.get("beta1", 0.5), 0.999))

    gan = GANLoss("lsgan")
    lcfg = cfg["loss"]
    perceptual = None
    if lcfg.get("perceptual", 0.0) > 0:
        from tir.losses.perceptual import VGGPerceptual
        perceptual = VGGPerceptual().to(device)

    metric_log = CSVMetricLogger(out_dir / "metrics.csv")
    best = -1.0; best_path = out_dir / "best.pth"
    step = 0; max_steps = max_steps if max_steps is not None else tcfg.get("max_steps")

    for epoch in range(tcfg["epochs"]):
        gen.train(); disc.train()
        for batch in train_ld:
            tir, real = batch["tir"].to(device), batch["rgb"].to(device)
            fake = gen(tir)[..., : real.shape[-2], : real.shape[-1]]

            # --- discriminator ---
            opt_d.zero_grad()
            loss_d = 0.5 * (gan(disc(tir, real), True) +
                            gan(disc(tir, fake.detach()), False))
            loss_d.backward(); opt_d.step()

            # --- generator ---
            opt_g.zero_grad()
            loss_g = lcfg.get("adversarial", 1.0) * gan(disc(tir, fake), True)
            loss_g = loss_g + lcfg.get("l1", 100.0) * torch.nn.functional.l1_loss(fake, real)
            if perceptual is not None:
                loss_g = loss_g + lcfg["perceptual"] * perceptual(fake, real)
            if lcfg.get("ssim", 0.0) > 0:
                from torchmetrics.functional import structural_similarity_index_measure as ssim
                f01 = ColorizeDataset.denormalize_rgb(fake)
                r01 = ColorizeDataset.denormalize_rgb(real)
                loss_g = loss_g + lcfg["ssim"] * (1.0 - ssim(f01, r01, data_range=1.0))
            loss_g.backward(); opt_g.step()
            step += 1
            if max_steps and step >= max_steps:
                break

        metrics = evaluate_colorize(gen, val_ld, device)
        metrics.update({"epoch": epoch, "step": step,
                        "loss_g": float(loss_g), "loss_d": float(loss_d)})
        metric_log.log(metrics)
        LOG.info("epoch %d | G %.3f D %.3f | PSNR %.2f SSIM %.3f", epoch,
                 float(loss_g), float(loss_d), metrics["psnr"], metrics["ssim"])
        if metrics["psnr"] > best:
            best = metrics["psnr"]
            torch.save({"model": gen.state_dict(), "cfg": dict(cfg["model"]),
                        "tir_mean": mean, "tir_std": std, "metrics": metrics},
                       best_path)
        if max_steps and step >= max_steps:
            LOG.info("hit max_steps=%d; stopping", max_steps)
            break

    LOG.info("best PSNR %.2f -> %s", best, best_path)
    return best_path


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Train TIR->RGB colorization model.")
    ap.add_argument("--config", default="configs/colorize.yaml")
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args(argv)
    train(args.config, args.max_steps)


if __name__ == "__main__":
    main()
