"""Uniform 2D sampling grid, real-space and its Fourier-space twin."""
from __future__ import annotations

from typing import Optional

import torch

from .backend import get_device, get_dtypes, to_tensor


class Grid:
    """A uniform 2D sampling grid.

    Mirrors the (x, y) coordinate-vector convention used throughout the
    original MATLAB code: ``x`` indexes columns, ``y`` indexes rows, and
    ``meshgrid(x, y)`` produces field arrays of shape (Ny, Nx).

    Parameters
    ----------
    x, y : 1D array-like
        Coordinate vectors, meters. Must be uniformly spaced.
    device : torch.device, optional
        Compute device; auto-selected (CUDA > MPS > CPU) if omitted.
    precision : {"single", "double"}
        Floating point precision for all derived tensors.
    """

    def __init__(self, x, y, device: Optional[torch.device] = None, precision: str = "single"):
        self.device = device if device is not None else get_device()
        self.real_dtype, self.complex_dtype = get_dtypes(precision)

        self.x = to_tensor(x, self.device, self.real_dtype).reshape(-1)
        self.y = to_tensor(y, self.device, self.real_dtype).reshape(-1)

        if self.x.numel() < 2 or self.y.numel() < 2:
            raise ValueError("Grid requires at least 2 samples along each axis.")

        self.nx = int(self.x.numel())
        self.ny = int(self.y.numel())

        self.dx = float((self.x[1:] - self.x[:-1]).mean())
        self.dy = float((self.y[1:] - self.y[:-1]).mean())

        # meshgrid(x, y) convention -> arrays of shape (Ny, Nx)
        self.X, self.Y = torch.meshgrid(self.x, self.y, indexing="xy")

    @classmethod
    def from_extent(
        cls,
        extent_x: float,
        nx: int,
        extent_y: Optional[float] = None,
        ny: Optional[int] = None,
        device: Optional[torch.device] = None,
        precision: str = "single",
    ) -> "Grid":
        """Build a grid centered at the origin, spanning `extent_x` (and
        optionally a different `extent_y`) meters, with `nx` (and `ny`)
        samples per axis."""
        extent_y = extent_x if extent_y is None else extent_y
        ny = nx if ny is None else ny
        x = torch.linspace(-extent_x / 2, extent_x / 2, nx)
        y = torch.linspace(-extent_y / 2, extent_y / 2, ny)
        return cls(x, y, device=device, precision=precision)

    def freq_grid(self):
        """Return (FX, FY) spatial-frequency meshgrids (cycles/m), DC-centered."""
        fx = torch.fft.fftshift(torch.fft.fftfreq(self.nx, d=self.dx)).to(self.device, self.real_dtype)
        fy = torch.fft.fftshift(torch.fft.fftfreq(self.ny, d=self.dy)).to(self.device, self.real_dtype)
        FX, FY = torch.meshgrid(fx, fy, indexing="xy")
        return FX, FY

    def zeros(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        dtype = self.complex_dtype if dtype is None else dtype
        return torch.zeros((self.ny, self.nx), device=self.device, dtype=dtype)

    def __repr__(self) -> str:
        return (
            f"Grid(nx={self.nx}, ny={self.ny}, dx={self.dx:.3e}, dy={self.dy:.3e}, "
            f"device={self.device}, dtype={self.real_dtype})"
        )
