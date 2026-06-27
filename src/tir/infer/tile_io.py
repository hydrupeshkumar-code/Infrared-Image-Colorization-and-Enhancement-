"""Tiled inference helpers: split a large array into overlapping tiles and
stitch model outputs back with feathered (cosine) blending to remove seams.

Works on (C, H, W) numpy arrays. The output grid is the input grid scaled by
``scale`` (1 for colorization at same resolution, 2 for SR).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass
class Tile:
    row: int          # top in the (LR) input grid
    col: int
    data: np.ndarray  # (C, th, tw)


def iter_tiles(arr: np.ndarray, size: int, overlap: int) -> Iterator[Tile]:
    """Yield overlapping tiles covering the whole array (edge-aligned)."""
    _, h, w = arr.shape
    step = max(1, size - overlap)
    rows = list(range(0, max(1, h - size + 1), step))
    cols = list(range(0, max(1, w - size + 1), step))
    if rows[-1] != h - size and h > size:
        rows.append(h - size)
    if cols[-1] != w - size and w > size:
        cols.append(w - size)
    for r in rows:
        for c in cols:
            r0, c0 = min(r, max(0, h - size)), min(c, max(0, w - size))
            yield Tile(r0, c0, arr[:, r0:r0 + size, c0:c0 + size])


def _feather_window(th: int, tw: int, overlap: int) -> np.ndarray:
    """2D cosine taper that is ~1 in the centre and ->0 at overlapped edges."""
    def ramp(n: int) -> np.ndarray:
        w = np.ones(n, dtype=np.float32)
        o = min(overlap, n // 2)
        if o > 0:
            t = np.linspace(0, np.pi / 2, o, dtype=np.float32)
            w[:o] = np.sin(t) ** 2
            w[-o:] = np.sin(t[::-1]) ** 2
        return w
    # Small floor so image-border pixels (covered by a single tapered tile)
    # still receive non-zero weight and are reconstructed correctly.
    return np.clip(np.outer(ramp(th), ramp(tw)), 1e-3, None).astype(np.float32)


class TileStitcher:
    """Accumulate scaled tile outputs into a seamless full canvas."""

    def __init__(self, out_channels: int, out_h: int, out_w: int,
                 overlap_out: int):
        self.acc = np.zeros((out_channels, out_h, out_w), dtype=np.float32)
        self.wsum = np.zeros((1, out_h, out_w), dtype=np.float32)
        self.overlap_out = overlap_out

    def add(self, out_tile: np.ndarray, row_out: int, col_out: int) -> None:
        c, th, tw = out_tile.shape
        win = _feather_window(th, tw, self.overlap_out)[None]
        self.acc[:, row_out:row_out + th, col_out:col_out + tw] += out_tile * win
        self.wsum[:, row_out:row_out + th, col_out:col_out + tw] += win

    def result(self) -> np.ndarray:
        return self.acc / np.clip(self.wsum, 1e-6, None)
