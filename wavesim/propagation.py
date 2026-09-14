"""Free-space propagation operators.

Currently implements the exact Angular Spectrum Method. The `Propagator`
interface is deliberately small so that Fresnel / Fraunhofer propagators
(planned) can be dropped in later without changing any calling code --
that's the "comprehensive wave propagation simulator" this platform is
meant to grow into.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Union

import torch

from .grid import Grid

ZType = Union[float, torch.Tensor]


class Propagator(ABC):
    """Common interface every propagation model implements."""

    grid: Grid

    @abstractmethod
    def propagate(self, field: torch.Tensor, z: ZType) -> torch.Tensor:
        """Propagate `field` (shape (Ny, Nx)) by distance(s) `z`.

        `z` may be a python scalar/float, a 0-d tensor, or a 1D tensor of
        multiple distances -- in the latter case the whole focal stack is
        computed in one batched FFT call, and the result has an extra
        leading dimension: shape (len(z), Ny, Nx).
        """


class AngularSpectrumMethod(Propagator):
    """Exact (Rayleigh-Sommerfeld) angular-spectrum free-space propagator.

    Reproduces the reference MATLAB ``ASM.m``::

        H(fx, fy) = exp(i k z sqrt(1 - (lambda fx)^2 - (lambda fy)^2))

    with evanescent orders (argument < 0) rolled off by a decaying
    exponential rather than discarded, matching the MATLAB behaviour
    exactly (including the ``exp(-alpha*|z|)`` sign convention, which
    correctly decays evanescent power for both propagation directions).

    The z-independent parts of the transfer function (``kz``, the
    evanescent decay rate ``alpha``) are precomputed once at construction
    time, so repeated calls to :meth:`propagate` at different z (e.g.
    sweeping a focal stack) only pay for the FFTs and a cheap complex
    exponential -- important since this constructor is meant to be reused
    across an entire optical system (each surface-to-surface gap gets its
    own call, exactly like ``ASM(U, x, y, lambda, z)`` in the MATLAB code).

    Parameters
    ----------
    grid : Grid
        Sampling grid the field lives on.
    wavelength : float
        Wavelength, meters.
    pad_factor : float, default 1.0
        Zero-pad the field to `pad_factor` times its size (per axis) before
        the FFT, then crop back to the original extent after propagation.
        1.0 disables padding. Padding suppresses the circular-convolution
        wraparound artifact inherent to FFT-based propagation at the cost
        of extra memory/compute -- exposed here so it can be swept later.
    """

    def __init__(self, grid: Grid, wavelength: float, pad_factor: float = 1.0):
        if pad_factor < 1.0:
            raise ValueError("pad_factor must be >= 1.0")

        self.grid = grid
        self.wavelength = float(wavelength)
        self.pad_factor = float(pad_factor)

        device, rdtype = grid.device, grid.real_dtype

        self._pad_ny = _padded_size(grid.ny, pad_factor)
        self._pad_nx = _padded_size(grid.nx, pad_factor)
        self._pad_top = (self._pad_ny - grid.ny) // 2
        self._pad_left = (self._pad_nx - grid.nx) // 2

        fx = torch.fft.fftshift(torch.fft.fftfreq(self._pad_nx, d=grid.dx)).to(device, rdtype)
        fy = torch.fft.fftshift(torch.fft.fftfreq(self._pad_ny, d=grid.dy)).to(device, rdtype)
        FX, FY = torch.meshgrid(fx, fy, indexing="xy")

        k = 2.0 * math.pi / self.wavelength
        arg = 1.0 - (self.wavelength * FX) ** 2 - (self.wavelength * FY) ** 2

        self.k = k
        self._kz = k * torch.sqrt(torch.clamp(arg, min=0.0))
        self._alpha = k * torch.sqrt(torch.clamp(-arg, min=0.0))

    # -- padding helpers -----------------------------------------------
    def _pad(self, field: torch.Tensor) -> torch.Tensor:
        if self.pad_factor == 1.0:
            return field
        out = torch.zeros((self._pad_ny, self._pad_nx), device=field.device, dtype=field.dtype)
        out[
            self._pad_top : self._pad_top + self.grid.ny,
            self._pad_left : self._pad_left + self.grid.nx,
        ] = field
        return out

    def _crop(self, field: torch.Tensor) -> torch.Tensor:
        if self.pad_factor == 1.0:
            return field
        return field[
            ...,
            self._pad_top : self._pad_top + self.grid.ny,
            self._pad_left : self._pad_left + self.grid.nx,
        ]

    # -- core -------------------------------------------------------------
    def transfer_function(self, z: ZType) -> torch.Tensor:
        """H(fx, fy; z) on the (possibly padded) frequency grid.

        Returns shape (pad_ny, pad_nx) for scalar z, or
        (len(z), pad_ny, pad_nx) for a 1D tensor of distances.
        """
        z_t = torch.as_tensor(z, device=self.grid.device, dtype=self.grid.real_dtype)
        if z_t.ndim > 0:
            kz, alpha = self._kz.unsqueeze(0), self._alpha.unsqueeze(0)
            z_t = z_t.view(-1, 1, 1)
        else:
            kz, alpha = self._kz, self._alpha
        return torch.exp(1j * kz * z_t) * torch.exp(-alpha * torch.abs(z_t))

    def propagate(self, field: torch.Tensor, z: ZType) -> torch.Tensor:
        if not torch.is_complex(field):
            field = field.to(self.grid.complex_dtype)

        Up = self._pad(field)
        Uf = torch.fft.fftshift(torch.fft.fft2(Up), dim=(-2, -1))
        H = self.transfer_function(z)

        Uz_f = Uf * H  # broadcasts (Ny,Nx) against (nz,Ny,Nx) automatically
        Uz = torch.fft.ifft2(torch.fft.ifftshift(Uz_f, dim=(-2, -1)))
        return self._crop(Uz)

    def __repr__(self) -> str:
        return (
            f"AngularSpectrumMethod(wavelength={self.wavelength:.3e}, "
            f"pad_factor={self.pad_factor}, grid={self.grid})"
        )


def _padded_size(n: int, pad_factor: float) -> int:
    return max(n, math.ceil(n * pad_factor))
