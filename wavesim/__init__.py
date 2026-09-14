"""wavesim: a GPU-friendly scalar wave propagation simulator.

Core building blocks:

- :class:`~wavesim.grid.Grid`                     sampling grid
- :class:`~wavesim.propagation.AngularSpectrumMethod`  free-space propagation
- :func:`~wavesim.lenses.spherical_lens_phase`     single spherical lens
- :func:`~wavesim.lenses.hex_mla_phase`            3-lenslet hex microlens array
- :class:`~wavesim.field.Field`                    chainable field wrapper

More propagation models (Fresnel, Fraunhofer) will implement the same
:class:`~wavesim.propagation.Propagator` interface as they're added.
"""

from .backend import get_device, get_dtypes
from .field import Field
from .grid import Grid
from .lenses import hex_mla_phase, spherical_lens_phase
from .propagation import AngularSpectrumMethod, Propagator

__all__ = [
    "Grid",
    "Field",
    "Propagator",
    "AngularSpectrumMethod",
    "spherical_lens_phase",
    "hex_mla_phase",
    "get_device",
    "get_dtypes",
]

__version__ = "0.1.0"
