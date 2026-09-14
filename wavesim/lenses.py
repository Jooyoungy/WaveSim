"""Phase-map generators for optical elements.

Ports of the MATLAB reference functions:

- :func:`spherical_lens_phase`  <- ``lens_spherical.m``
    A single ideal (hyperbolic-profile) refractive lens over a circular
    aperture.
- :func:`hex_mla_phase`         <- ``lens_spherical_MLA.m``
    A 3-lens hexagonally-packed microlens-array cluster.

Both return a complex transmission function sampled on a :class:`~wavesim.grid.Grid`,
ready to multiply elementwise into a propagating :class:`~wavesim.field.Field`.
"""
from __future__ import annotations

import math
from typing import List, Tuple

import torch

from .backend import phasor
from .grid import Grid


def spherical_lens_phase(grid: Grid, wavelength: float, focal_length: float, diameter: float) -> torch.Tensor:
    """Ideal thin spherical (hyperbolic-profile) lens transmission.

    Port of MATLAB ``lens_spherical.m``::

        phase = (2*pi/lambda) * (f - sqrt(f^2 + x^2 + y^2))
        PhaseOut = exp(i*phase) .* (circular aperture mask)

    Parameters
    ----------
    grid : Grid
    wavelength : float
        meters.
    focal_length : float
        Lens focal length, meters.
    diameter : float
        Clear-aperture diameter, meters. Transmission is exactly zero
        outside the aperture.

    Returns
    -------
    torch.Tensor
        Complex transmission, shape (grid.ny, grid.nx).
    """
    k = 2.0 * math.pi / wavelength
    R = diameter / 2.0

    r2 = grid.X**2 + grid.Y**2
    mask = (r2 <= R**2).to(grid.real_dtype)
    phase = k * (focal_length - torch.sqrt(focal_length**2 + r2))
    return phasor(phase) * mask


def hex_mla_phase(
    grid: Grid,
    wavelength: float,
    focal_length: float,
    pitch: float,
    mode: int = 1,
    level: float = 1.0,
) -> torch.Tensor:
    """3-lenslet hexagonally-packed microlens-array transmission.

    Port of MATLAB ``lens_spherical_MLA.m``. Places circular/hex-clipped
    lenslets on a hexagonal-close-packed lattice of pitch `pitch`, keeps
    only the cluster of lenslets within radius
    ``(pitch/(2*sqrt(3))) * 2 * level`` of the origin (3 lenslets for the
    default `level=1`), re-centers that cluster on the origin, and assigns
    each lenslet its own spherical phase profile clipped to its hexagonal
    Voronoi cell.

    Parameters
    ----------
    grid : Grid
    wavelength : float
        meters.
    focal_length : float
        Per-lenslet focal length, meters.
    pitch : float
        Center-to-center lenslet spacing, meters.
    mode : int, default 1
        Row-offset convention for the hex lattice (0 or 1); also selects
        whether the resulting cluster is re-centered vertically (mode=0)
        or left as-is (mode=1), matching the MATLAB implementation.
    level : float, default 1.0
        Controls how many lattice rings are kept around the origin before
        clipping to the (typically 3-lenslet) cluster; level=1 reproduces
        the "3 hexagonal microlens" design.

    Returns
    -------
    torch.Tensor
        Complex transmission, shape (grid.ny, grid.nx); zero outside the
        lenslet cluster.
    """
    device, rdtype, cdtype = grid.device, grid.real_dtype, grid.complex_dtype
    k = 2.0 * math.pi / wavelength

    dx = pitch
    dy = pitch * math.sqrt(3) / 2.0
    r = pitch / 2.0
    cluster_radius = (r / math.sqrt(3)) * 2.0 * level
    y_offset = dy / 3.0

    x_min, x_max = float(grid.x.min()), float(grid.x.max())
    y_min, y_max = float(grid.y.min()), float(grid.y.max())

    centers: List[Tuple[float, float]] = []
    i_lo, i_hi = math.floor(y_min / dy), math.ceil(y_max / dy)
    for i in range(i_lo, i_hi + 1):
        cy = i * dy + y_offset * (mode * 2 - 1)
        x_offset = dx / 2.0 if (i % 2 == 0) else 0.0
        j_lo = math.floor((x_min - x_offset) / dx)
        j_hi = math.ceil((x_max - x_offset) / dx)
        for j in range(j_lo, j_hi + 1):
            cx = j * dx + x_offset
            if cx**2 + cy**2 <= cluster_radius**2:
                centers.append((cx, cy))

    if not centers:
        raise ValueError(
            "No microlens centers found within the cluster radius for this "
            "grid/pitch/level combination; increase 'level' or the grid extent."
        )

    # Center the cluster of microlenses (only when exactly 3 were found,
    # matching the MATLAB reference's guard).
    if len(centers) == 3:
        cx_mean = sum(c[0] for c in centers) / 3.0
        cy_mean = sum(c[1] for c in centers) / 3.0
        centers = [(cx - cx_mean, cy - cy_mean) for cx, cy in centers]

    if mode == 0:
        ys = [c[1] for c in centers]
        mla_ymin = min(ys) - pitch / math.sqrt(3)
        mla_ymax = max(ys) + pitch / math.sqrt(3)
        y_offset_all = (mla_ymin + mla_ymax) / 2.0
        centers = [(cx, cy - y_offset_all) for cx, cy in centers]
    else:
        y_offset_all = 0.0

    phase_out = torch.zeros((grid.ny, grid.nx), device=device, dtype=cdtype)

    for cx, cy in centers:
        if cx**2 + (cy + y_offset_all) ** 2 > cluster_radius**2:
            continue
        DX = grid.X - cx
        DY = grid.Y - cy
        mask = (DX.abs() <= r) & (math.sqrt(3) * DY.abs() <= pitch - DX.abs())
        phase = k * (focal_length - torch.sqrt(focal_length**2 + DX**2 + DY**2))
        phase_out = torch.where(mask, phasor(phase), phase_out)

    return phase_out
