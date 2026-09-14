"""Port of the MATLAB ``compound_test.m`` reference script onto the wavesim
platform: a Fourier lens (FL) followed by a 3-lenslet hex microlens array
(MLA), propagated with the Angular Spectrum Method.

Run:
    python examples/compound_test.py

This reproduces, in order:
  1. lens/system parameters + grid sizing identical to the MATLAB script,
  2. the two phase maps (spherical FL, hex MLA) and a plot of their phase,
  3. the "conventional" cascade: point source -> ASM -> FL -> aperture ->
     ASM -> MLA -> ASM -> sensor intensity,
  4. a small defocus sweep (a mini focal/PSF stack) computed as one batched
     ASM call per propagation stage.

Figures are saved under examples/out/ instead of popped up interactively,
so this also runs headless / in CI.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from wavesim import AngularSpectrumMethod, Field, Grid, hex_mla_phase, spherical_lens_phase
from wavesim.backend import get_device
from wavesim.visualization import imshow

OUT_DIR = Path(__file__).parent / "out"


def build_system(device=None):
    # ---- parameters (verbatim from compound_test.m) ----------------
    lam0 = 520e-9
    f_FL = 11.047e-3
    f_ML = 9.5e-3
    d_FL = 4.22e-3
    d_ML = 2.11e-3
    pitch_ML = d_ML / 2 * math.sqrt(3)

    pxsize_img = 350e-9  # meta-atom pitch
    fov_init = pitch_ML * 2.5
    matrixsize = math.ceil(fov_init / pxsize_img / 1000) * 1000 + 1

    print(f"Fourier Lens NA: {d_FL / 2 / f_FL:.4f}")
    print(f"MLA NA:          {d_ML / 2 / f_ML:.4f}")
    print(f"matrix size:     {matrixsize} px  (FOV = {matrixsize * pxsize_img * 1e3:.3f} mm)")

    n_half = matrixsize // 2
    coords = (torch.arange(-n_half + 1, n_half) * pxsize_img).double()
    grid = Grid(coords, coords, device=device, precision="single")

    params = dict(lam0=lam0, f_FL=f_FL, f_ML=f_ML, d_FL=d_FL, d_ML=d_ML, pitch_ML=pitch_ML)
    return grid, params


def main():
    OUT_DIR.mkdir(exist_ok=True)
    device = get_device()
    print(f"Using device: {device}")

    grid, p = build_system(device=device)
    lam0, f_FL, f_ML, d_FL, d_ML, pitch_ML = (
        p["lam0"], p["f_FL"], p["f_ML"], p["d_FL"], p["d_ML"], p["pitch_ML"],
    )

    # ---- phase maps --------------------------------------------------
    phase_FL = spherical_lens_phase(grid, lam0, f_FL, d_FL)
    phase_MLA = hex_mla_phase(grid, lam0, f_ML, pitch_ML, mode=1, level=1)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    imshow(torch.angle(phase_FL), grid, ax=axes[0], title="Fourier lens phase", cmap="twilight")
    imshow(torch.angle(phase_MLA), grid, ax=axes[1], title="MLA phase", cmap="twilight")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "phase_maps.png", dpi=150)
    plt.close(fig)

    # ---- propagators (built once, reused across the cascade) --------
    asm_FL = AngularSpectrumMethod(grid, lam0)
    asm_ML = AngularSpectrumMethod(grid, lam0)

    aperture_FL = (grid.X**2 + grid.Y**2 <= (d_FL / 2) ** 2).to(grid.complex_dtype)

    # ---- conventional cascade: source -> FL -> MLA -> sensor --------
    field = Field.point_source(grid, lam0, radius=1e-6)
    field = field.propagate(asm_FL, f_FL)
    field = field.apply(phase_FL * aperture_FL).propagate(asm_FL, f_FL)
    field = field.apply(phase_MLA).propagate(asm_ML, f_ML)

    fig, ax = plt.subplots(figsize=(4, 3.5))
    imshow(field.intensity(), grid, ax=ax, title="Sensor-plane intensity")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sensor_intensity.png", dpi=150)
    plt.close(fig)
    print(f"Saved {OUT_DIR / 'sensor_intensity.png'}")

    # ---- mini defocus / PSF stack, batched over z --------------------
    delz = torch.linspace(-0.2e-3, 0.2e-3, 5, device=device, dtype=grid.real_dtype)

    src = Field.point_source(grid, lam0, radius=1e-6)
    u = asm_FL.propagate(src.data, f_FL - delz)  # batched over depth: (D, Ny, Nx)
    u = u * (phase_FL * aperture_FL)
    u = asm_FL.propagate(u, f_FL)  # scalar z broadcasts across the (D, Ny, Nx) batch
    u = u * phase_MLA
    u = asm_ML.propagate(u, f_ML)

    psf_stack = (u.abs() ** 2).cpu()

    fig, axes = plt.subplots(1, len(delz), figsize=(3 * len(delz), 3))
    for i, ax in enumerate(axes):
        imshow(psf_stack[i], grid, ax=ax, title=f"dz={delz[i].item()*1e6:.0f} um", colorbar=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "psf_stack.png", dpi=150)
    plt.close(fig)
    print(f"Saved {OUT_DIR / 'psf_stack.png'}")


if __name__ == "__main__":
    main()
