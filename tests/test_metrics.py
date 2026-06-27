"""Metric sanity + model forward-shape contracts."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tir.eval.metrics import psnr, ssim
from tir.infer.tile_io import TileStitcher, iter_tiles
from tir.models.colorize_model import build_colorize_model
from tir.models.sr_model import build_sr_model


def test_psnr_ssim_identity():
    x = torch.rand(1, 1, 32, 32)
    assert psnr(x, x, data_range=1.0) > 60          # ~inf for identical
    assert ssim(x, x, data_range=1.0) == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize("name", ["edsr", "swinir"])
def test_sr_forward_doubles_resolution(name):
    model = build_sr_model({"name": name, "scale": 2})
    out = model(torch.rand(2, 1, 16, 16))
    assert out.shape[-2:] == (32, 32) and out.shape[1] == 1


def test_colorize_forward_shapes():
    gen, disc = build_colorize_model({"name": "pix2pix", "n_down": 5})
    tir = torch.rand(2, 1, 64, 64)
    rgb = gen(tir)
    assert rgb.shape == (2, 3, 64, 64)
    assert disc(tir, rgb).ndim == 4  # PatchGAN map


def test_tile_stitch_reconstructs_constant():
    arr = np.ones((1, 40, 40), dtype=np.float32) * 0.5
    stitch = TileStitcher(1, 40, 40, overlap_out=4)
    for t in iter_tiles(arr, size=16, overlap=4):
        stitch.add(t.data, t.row, t.col)
    out = stitch.result()
    np.testing.assert_allclose(out, 0.5, atol=1e-5)
