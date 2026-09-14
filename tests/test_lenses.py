import math

import pytest
import torch

from wavesim import Grid, hex_mla_phase, spherical_lens_phase


def _small_grid(n=101, extent=6e-3):
    return Grid.from_extent(extent, n, device=torch.device("cpu"), precision="double")


def test_spherical_lens_on_axis_phase_is_zero():
    grid = _small_grid()
    phase_map = spherical_lens_phase(grid, wavelength=520e-9, focal_length=11e-3, diameter=4e-3)

    c = grid.ny // 2  # near-center pixel, where x=y~0
    assert torch.angle(phase_map[c, c]).abs().item() < 1e-3


def test_spherical_lens_aperture_clips_transmission():
    grid = _small_grid()
    diameter = 2e-3
    phase_map = spherical_lens_phase(grid, wavelength=520e-9, focal_length=11e-3, diameter=diameter)

    r2 = grid.X**2 + grid.Y**2
    outside = r2 > (diameter / 2) ** 2
    inside = ~outside

    assert torch.all(phase_map[outside] == 0)
    assert torch.all(phase_map[inside].abs() > 0.99)  # unit magnitude inside aperture


def test_hex_mla_produces_three_lenslets_with_unit_magnitude_transmission():
    grid = _small_grid(n=401, extent=4e-3)
    pitch = 1.0e-3
    phase_map = hex_mla_phase(grid, wavelength=520e-9, focal_length=9.5e-3, pitch=pitch, mode=1, level=1)

    covered = phase_map.abs() > 0.99
    assert covered.any()
    # transmission is either ~0 (outside all lenslets) or ~1 in magnitude (inside one)
    mags = phase_map.abs()
    assert torch.all((mags < 1e-6) | (mags > 0.99))

    # roughly 3 hexagonal cells of area (sqrt(3)/2)*pitch^2 each
    covered_area = covered.sum().item() * grid.dx * grid.dy
    expected_area = 3 * (math.sqrt(3) / 2) * pitch**2
    assert covered_area == pytest.approx(expected_area, rel=0.15)


def test_hex_mla_raises_when_cluster_not_found():
    grid = _small_grid()
    # level=0 collapses the cluster-selection radius to 0; with mode=1 no
    # lattice point sits exactly at the origin, so no lenslet qualifies.
    with pytest.raises(ValueError):
        hex_mla_phase(grid, wavelength=520e-9, focal_length=9.5e-3, pitch=1.0e-3, mode=1, level=0.0)
