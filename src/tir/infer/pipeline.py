"""Phase 4 end-to-end inference.

Takes a 200 m single-channel TIR GeoTIFF and produces georeferenced
``HR_TIR_100m`` and ``RGB_100m`` GeoTIFFs (CRS preserved, geotransform scaled
for 100 m). Arbitrary-size scenes are handled by tiled inference with
overlap + feather blending, streamed tile-by-tile to bound memory. Stages can
be run jointly (``both``) or independently (``sr`` / ``colorize``).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from tir.infer.tile_io import TileStitcher, iter_tiles
from tir.models.colorize_model import build_colorize_model
from tir.models.sr_model import build_sr_model
from tir.train.train_sr import resolve_device
from tir.utils.config import load_config
from tir.utils.geo import GeoRaster, read_raster, scaled_transform, write_raster
from tir.utils.logging import get_logger

LOG = get_logger("pipeline")


class TIRPipeline:
    """Loads trained SR + colorization models and runs tiled inference."""

    def __init__(self, sr_ckpt: str | None, colorize_ckpt: str | None,
                 device: torch.device, amp: bool = False):
        self.device = device
        self.amp = amp and device.type == "cuda"
        self.sr_model = self.sr_mean = self.sr_std = None
        self.gen = self.col_mean = self.col_std = None
        self.scale = 2
        if sr_ckpt:
            ck = torch.load(sr_ckpt, map_location=device, weights_only=False)
            self.sr_model = build_sr_model(dict(ck["cfg"])).to(device).eval()
            self.sr_model.load_state_dict(ck["model"])
            self.sr_mean, self.sr_std = ck["tir_mean"], ck["tir_std"]
            self.scale = ck["cfg"].get("scale", 2)
        if colorize_ckpt:
            ck = torch.load(colorize_ckpt, map_location=device, weights_only=False)
            self.gen, _ = build_colorize_model(dict(ck["cfg"]))
            self.gen = self.gen.to(device).eval()
            self.gen.load_state_dict(ck["model"])
            self.col_mean, self.col_std = ck["tir_mean"], ck["tir_std"]

    @torch.no_grad()
    def _run(self, model, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", enabled=self.amp):
            return model(x).float()

    def super_resolve(self, lr_tir: np.ndarray, size: int, overlap: int,
                      batch_size: int = 4) -> tuple[np.ndarray, float]:
        """LR TIR (1,h,w) -> HR TIR (1, h*scale, w*scale). Returns (out, sec/tile)."""
        _, h, w = lr_tir.shape
        norm = (lr_tir - self.sr_mean) / self.sr_std
        stitch = TileStitcher(1, h * self.scale, w * self.scale,
                              overlap * self.scale)
        tiles = list(iter_tiles(norm, size, overlap))
        t0 = time.perf_counter()
        for i in range(0, len(tiles), batch_size):
            chunk = tiles[i:i + batch_size]
            batch = torch.from_numpy(np.stack([t.data for t in chunk])).to(self.device)
            out = self._run(self.sr_model, batch).cpu().numpy()
            for t, o in zip(chunk, out):
                o = o * self.sr_std + self.sr_mean
                stitch.add(o, t.row * self.scale, t.col * self.scale)
        sec_per_tile = (time.perf_counter() - t0) / max(1, len(tiles))
        return stitch.result(), sec_per_tile

    def colorize(self, tir_100m: np.ndarray, size: int, overlap: int,
                 batch_size: int = 4) -> tuple[np.ndarray, float]:
        """TIR (1,H,W) @100m -> RGB (3,H,W) in [0,1]. Returns (out, sec/tile)."""
        _, h, w = tir_100m.shape
        norm = (tir_100m - self.col_mean) / self.col_std
        stitch = TileStitcher(3, h, w, overlap)
        tiles = list(iter_tiles(norm, size, overlap))
        t0 = time.perf_counter()
        for i in range(0, len(tiles), batch_size):
            chunk = tiles[i:i + batch_size]
            batch = torch.from_numpy(np.stack([t.data for t in chunk])).to(self.device)
            out = self._run(self.gen, batch).cpu().numpy()
            out = (out + 1.0) / 2.0  # [-1,1] -> [0,1]
            for t, o in zip(chunk, out):
                stitch.add(np.clip(o, 0, 1), t.row, t.col)
        sec_per_tile = (time.perf_counter() - t0) / max(1, len(tiles))
        return stitch.result(), sec_per_tile


def run(input_path: str, out_dir: str, cfg_path: str = "configs/infer.yaml",
        stages: str | None = None) -> dict:
    cfg = load_config(cfg_path)
    device = resolve_device(cfg.get("device", "auto"))
    tcfg = cfg["tile"]
    stages = stages or cfg.get("stages", "both")
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    src = read_raster(input_path)
    pipe = TIRPipeline(
        cfg.get("sr_checkpoint") if stages in ("both", "sr") else None,
        cfg.get("colorize_checkpoint") if stages in ("both", "colorize") else None,
        device, amp=tcfg.get("amp", False))

    outputs: dict[str, str] = {}
    timings: dict[str, float] = {}

    # --- SR ---
    if stages in ("both", "sr"):
        hr, spt = pipe.super_resolve(src.data, tcfg["size"], tcfg["overlap"],
                                     tcfg.get("batch_size", 4))
        timings["sr_sec_per_tile"] = spt
        hr_transform = scaled_transform(src.transform, 1.0 / pipe.scale)
        hr_raster = GeoRaster(hr, hr_transform, src.crs, src.nodata)
        p = out_dir / "HR_TIR_100m.tif"; write_raster(p, hr_raster)
        outputs["hr_tir"] = str(p)
        tir_for_color = hr
        color_transform = hr_transform
    else:
        tir_for_color = src.data
        color_transform = src.transform

    # --- Colorize ---
    if stages in ("both", "colorize"):
        rgb, spt = pipe.colorize(tir_for_color, tcfg["size"] * pipe.scale,
                                 tcfg["overlap"] * pipe.scale,
                                 tcfg.get("batch_size", 4))
        timings["colorize_sec_per_tile"] = spt
        rgb_raster = GeoRaster(rgb.astype(np.float32), color_transform, src.crs, None)
        p = out_dir / "RGB_100m.tif"; write_raster(p, rgb_raster)
        outputs["rgb"] = str(p)

    LOG.info("wrote %s | timings %s", outputs, {k: round(v, 4) for k, v in timings.items()})
    return {"outputs": outputs, "timings": timings}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Run TIR SR + colorization pipeline.")
    ap.add_argument("--input", required=True, help="200 m single-channel TIR GeoTIFF")
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--config", default="configs/infer.yaml")
    ap.add_argument("--stages", choices=["both", "sr", "colorize"], default=None)
    args = ap.parse_args(argv)
    run(args.input, args.out_dir, args.config, args.stages)


if __name__ == "__main__":
    main()
