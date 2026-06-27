"""Landsat Collection-2 radiometric conversions."""
import numpy as np
import pytest

from tir.data.radiometric import (apply_optical, apply_thermal,
                                   normalize_level)


def test_l2_surface_temperature_to_kelvin():
    # ST_B10 DN ~44000 -> ~299 K (typical land surface temperature)
    dn = np.full((1, 4, 4), 44000.0, dtype=np.float32)
    k = apply_thermal(dn, "L2")
    assert 280.0 < float(k.mean()) < 330.0


def test_l2_reflectance_in_unit_range():
    dn = np.array([[[8000, 20000], [40000, 60000]]], dtype=np.float32)
    refl = apply_optical(dn, "L2")
    assert refl.min() >= 0.0 and refl.max() <= 1.0


def test_l1_brightness_temperature_is_kelvin():
    dn = np.full((1, 4, 4), 30000.0, dtype=np.float32)
    k = apply_thermal(dn, "L1")
    assert 200.0 < float(k.mean()) < 400.0


def test_none_is_passthrough():
    arr = np.array([[[300.0, 305.0]]], dtype=np.float32)
    np.testing.assert_allclose(apply_thermal(arr, "none"), arr)
    np.testing.assert_allclose(apply_optical(arr, "none"), arr)


def test_invalid_level_raises():
    with pytest.raises(ValueError):
        normalize_level("L3")
