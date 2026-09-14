import math

import pytest
import torch

from wavesim import AngularSpectrumMethod, Grid


def _small_grid(n=64, extent=200e-6):
    return Grid.from_extent(extent, n, device=torch.device("cpu"), precision="double")


def test_identity_at_zero_distance():
    grid = _small_grid()
    asm = AngularSpectrumMethod(grid, wavelength=520e-9)

    torch.manual_seed(0)
    field = torch.randn(grid.ny, grid.nx, dtype=grid.complex_dtype)

    out = asm.propagate(field, 0.0)
    assert torch.allclose(out, field, atol=1e-9, rtol=1e-6)


def test_energy_conservation_for_propagating_field():
    # A smooth, band-limited field (Gaussian) carries negligible energy in
    # the evanescent region, so |H|=1 everywhere it matters and free-space
    # ASM propagation should be unitary (Parseval): total intensity is
    # conserved even though the field itself spreads out spatially.
    grid = _small_grid(n=128, extent=500e-6)
    asm = AngularSpectrumMethod(grid, wavelength=520e-9)

    r2 = grid.X**2 + grid.Y**2
    w0 = 20e-6
    field = torch.exp(-r2 / w0**2).to(grid.complex_dtype)

    e0 = (field.abs() ** 2).sum().item()
    out = asm.propagate(field, 5e-3)
    e1 = (out.abs() ** 2).sum().item()

    assert e1 == pytest.approx(e0, rel=1e-3)


def test_batched_z_matches_loop():
    grid = _small_grid()
    asm = AngularSpectrumMethod(grid, wavelength=520e-9)

    r2 = grid.X**2 + grid.Y**2
    field = torch.exp(-r2 / (30e-6) ** 2).to(grid.complex_dtype)

    zs = torch.tensor([1e-3, 2e-3, 3e-3], dtype=grid.real_dtype)
    batched = asm.propagate(field, zs)

    for i, z in enumerate(zs.tolist()):
        single = asm.propagate(field, z)
        assert torch.allclose(batched[i], single, atol=1e-9, rtol=1e-6)


def test_padding_preserves_shape_and_center_value():
    grid = _small_grid(n=64, extent=300e-6)

    r2 = grid.X**2 + grid.Y**2
    field = torch.exp(-r2 / (10e-6) ** 2).to(grid.complex_dtype)  # tightly confined, well inside the grid

    asm = AngularSpectrumMethod(grid, wavelength=520e-9)
    asm_pad = AngularSpectrumMethod(grid, wavelength=520e-9, pad_factor=2.0)

    out = asm.propagate(field, 1e-3)
    out_pad = asm_pad.propagate(field, 1e-3)

    assert out_pad.shape == field.shape
    # away from the (wraparound-prone) edges, padded and unpadded results
    # should closely agree for a field this well-confined
    c = grid.ny // 2
    assert torch.allclose(out_pad[c - 5 : c + 5, c - 5 : c + 5], out[c - 5 : c + 5, c - 5 : c + 5], atol=1e-6)


def test_pad_factor_rejects_values_below_one():
    grid = _small_grid()
    with pytest.raises(ValueError):
        AngularSpectrumMethod(grid, wavelength=520e-9, pad_factor=0.5)
