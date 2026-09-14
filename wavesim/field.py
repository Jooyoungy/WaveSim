"""A small, chainable wrapper tying a complex field array to its grid."""
from __future__ import annotations

from typing import Union

import torch

from .backend import as_numpy
from .grid import Grid
from .propagation import Propagator, ZType


class Field:
    """A complex scalar field sampled on a :class:`~wavesim.grid.Grid`.

    Wraps a raw tensor with its grid and wavelength so optical systems can
    be expressed as a readable chain, e.g.::

        field = (
            Field.point_source(grid, wavelength, radius=1e-6)
            .propagate(asm_to_lens1, z=d1)
            .apply(lens1_phase * aperture_mask)
            .propagate(asm_to_lens2, z=d2)
            .apply(lens2_phase)
            .propagate(asm_to_sensor, z=d3)
        )
        intensity = field.intensity_numpy()
    """

    def __init__(self, data: torch.Tensor, grid: Grid, wavelength: float):
        self.data = data
        self.grid = grid
        self.wavelength = float(wavelength)

    @classmethod
    def point_source(cls, grid: Grid, wavelength: float, radius: float) -> "Field":
        """Uniform-amplitude disk of the given radius (meters); a simple
        stand-in for a point/pinhole source, matching the MATLAB test
        script's ``pointSource`` construction."""
        r2 = grid.X**2 + grid.Y**2
        data = (r2 <= radius**2).to(grid.complex_dtype)
        return cls(data, grid, wavelength)

    @classmethod
    def plane_wave(cls, grid: Grid, wavelength: float, amplitude: float = 1.0) -> "Field":
        data = torch.full((grid.ny, grid.nx), amplitude, device=grid.device, dtype=grid.complex_dtype)
        return cls(data, grid, wavelength)

    @classmethod
    def from_array(cls, data, grid: Grid, wavelength: float) -> "Field":
        if not isinstance(data, torch.Tensor):
            data = torch.as_tensor(data)
        return cls(data.to(device=grid.device, dtype=grid.complex_dtype), grid, wavelength)

    def propagate(self, propagator: Propagator, z: ZType) -> "Field":
        """Free-space propagate by distance(s) `z`. If `z` is a 1D tensor,
        the result's `.data` gains a leading (depth) dimension."""
        return Field(propagator.propagate(self.data, z), self.grid, self.wavelength)

    def apply(self, transmission: Union[torch.Tensor, "Field"]) -> "Field":
        """Elementwise-multiply by a transmission/phase mask (e.g. a lens)."""
        t = transmission.data if isinstance(transmission, Field) else transmission
        return Field(self.data * t.to(self.data.dtype), self.grid, self.wavelength)

    def intensity(self) -> torch.Tensor:
        return self.data.abs() ** 2

    def intensity_numpy(self):
        return as_numpy(self.intensity())

    def numpy(self):
        return as_numpy(self.data)

    def __repr__(self) -> str:
        return f"Field(shape={tuple(self.data.shape)}, wavelength={self.wavelength:.3e}, grid={self.grid})"
