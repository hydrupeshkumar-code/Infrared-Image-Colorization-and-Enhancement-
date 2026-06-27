"""Colorization models (TIR 1-ch -> RGB 3-ch). Swappable via config.

Default: pix2pix — a U-Net generator with skip connections + a PatchGAN
discriminator (Isola et al. 2017). ``build_colorize_model`` returns
``(generator, discriminator)``.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class UNetDown(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, normalize: bool = True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers = [nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
                  nn.InstanceNorm2d(out_ch), nn.ReLU(inplace=True)]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = self.model(x)
        return torch.cat([x, skip], dim=1)


class UNetGenerator(nn.Module):
    """U-Net with ``n_down`` encoder steps; symmetric decoder with skips."""

    def __init__(self, in_ch: int = 1, out_ch: int = 3, ngf: int = 64,
                 n_down: int = 6):
        super().__init__()
        assert n_down >= 2
        self.n_down = n_down
        widths = [min(ngf * (2 ** i), ngf * 8) for i in range(n_down)]

        self.downs = nn.ModuleList()
        self.downs.append(UNetDown(in_ch, widths[0], normalize=False))
        for i in range(1, n_down):
            # innermost layer carries no norm (pix2pix convention; also avoids
            # InstanceNorm on a 1x1 bottleneck)
            self.downs.append(UNetDown(widths[i - 1], widths[i],
                                       normalize=(i != n_down - 1)))

        self.ups = nn.ModuleList()
        for i in range(n_down - 1, 0, -1):
            in_w = widths[i] if i == n_down - 1 else widths[i] * 2
            self.ups.append(UNetUp(in_w, widths[i - 1]))
        self.final = nn.Sequential(
            nn.ConvTranspose2d(widths[0] * 2, out_ch, 4, 2, 1), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
        # bottleneck is the last skip; decode upward
        x = skips[-1]
        for idx, up in enumerate(self.ups):
            skip = skips[-(idx + 2)]
            x = up(x, skip)
        return self.final(x)


class PatchGAN(nn.Module):
    """70x70-style PatchGAN discriminator over concatenated (cond, rgb)."""

    def __init__(self, in_ch: int = 4, ndf: int = 64, n_layers: int = 3):
        super().__init__()
        layers = [nn.Conv2d(in_ch, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True)]
        nf = ndf
        for i in range(1, n_layers):
            nf_prev, nf = nf, min(ndf * (2 ** i), ndf * 8)
            layers += [nn.Conv2d(nf_prev, nf, 4, 2, 1, bias=False),
                       nn.InstanceNorm2d(nf), nn.LeakyReLU(0.2, True)]
        nf_prev, nf = nf, min(nf * 2, ndf * 8)
        layers += [nn.Conv2d(nf_prev, nf, 4, 1, 1, bias=False),
                   nn.InstanceNorm2d(nf), nn.LeakyReLU(0.2, True)]
        layers += [nn.Conv2d(nf, 1, 4, 1, 1)]
        self.model = nn.Sequential(*layers)

    def forward(self, cond: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        return self.model(torch.cat([cond, rgb], dim=1))


def build_colorize_model(cfg: dict) -> tuple[nn.Module, nn.Module]:
    cfg = dict(cfg)
    name = cfg.get("name", "pix2pix").lower()
    if name != "pix2pix":
        raise ValueError(f"Unknown colorize model '{name}'. Options: ['pix2pix']")
    gen = UNetGenerator(in_ch=1, out_ch=3, ngf=cfg.get("ngf", 64),
                        n_down=cfg.get("n_down", 6))
    disc = PatchGAN(in_ch=1 + 3, ndf=cfg.get("ndf", 64))
    return gen, disc
