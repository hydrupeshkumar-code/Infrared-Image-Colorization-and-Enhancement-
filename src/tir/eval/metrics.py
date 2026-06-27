"""Evaluation metrics: PSNR, SSIM (torchmetrics) and FID (pytorch-fid)."""
from __future__ import annotations

import numpy as np
import torch


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    from torchmetrics.functional import peak_signal_noise_ratio
    return float(peak_signal_noise_ratio(pred, target, data_range=data_range))


def ssim(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    from torchmetrics.functional import structural_similarity_index_measure
    return float(structural_similarity_index_measure(pred, target, data_range=data_range))


def compute_fid(fake_rgb: list[np.ndarray], real_rgb: list[np.ndarray],
                device: str = "cpu", dims: int = 2048) -> float:
    """FID between two sets of RGB images (each (3,H,W) in [0,1]).

    Returns ``nan`` if there are too few samples for the chosen feature dim.
    """
    if len(fake_rgb) < 2 or len(real_rgb) < 2:
        return float("nan")
    try:
        from pytorch_fid.fid_score import calculate_frechet_distance
        from pytorch_fid.inception import InceptionV3
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        model = InceptionV3([block_idx]).to(device).eval()
    except Exception:
        # Inception weights unavailable (e.g. offline) -> skip FID gracefully.
        return float("nan")

    def activations(images: list[np.ndarray]) -> np.ndarray:
        feats = []
        with torch.no_grad():
            for img in images:
                x = torch.from_numpy(np.clip(img, 0, 1)).float()[None].to(device)
                if x.shape[1] == 1:
                    x = x.repeat(1, 3, 1, 1)
                if x.shape[-1] < 75:  # inception needs >=75px
                    x = torch.nn.functional.interpolate(x, size=75, mode="bilinear",
                                                        align_corners=False)
                pred = model(x)[0].squeeze(-1).squeeze(-1)
                feats.append(pred.cpu().numpy())
        return np.concatenate(feats, axis=0)

    act_fake, act_real = activations(fake_rgb), activations(real_rgb)
    mu1, sig1 = act_fake.mean(0), np.cov(act_fake, rowvar=False)
    mu2, sig2 = act_real.mean(0), np.cov(act_real, rowvar=False)
    return float(calculate_frechet_distance(mu1, sig1, mu2, sig2))
