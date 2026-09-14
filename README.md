# wavesim

A GPU-friendly scalar wave propagation simulator for testing custom-designed
phase maps (lenses, microlens arrays, and eventually arbitrary metasurface
phase profiles).

This is the **platform layer**: a reusable `Grid` / propagator / lens /
`Field` core, built on PyTorch so every FFT and elementwise op runs on
CUDA/MPS transparently, plus a direct port of the original MATLAB reference
functions. Later work (lens-design comparisons, pitch sweeps, zero-padding
studies) builds on top of this without changing the core.

## Layout

```
wavesim/
  backend.py       device/dtype selection, tensor conversion helpers
  grid.py           Grid: uniform 2D (x, y) sampling grid + frequency grid
  propagation.py    Propagator interface; AngularSpectrumMethod (ASM)
  lenses.py         spherical_lens_phase, hex_mla_phase
  field.py          Field: chainable (data, grid, wavelength) wrapper
  visualization.py  imagesc-style matplotlib helper
examples/
  compound_test.py  port of compound_test.m (Fourier lens + hex MLA system)
tests/
  test_propagation.py
  test_lenses.py
```

## MATLAB -> Python mapping

| MATLAB              | Python                                   |
|----------------------|-------------------------------------------|
| `ASM.m`               | `wavesim.AngularSpectrumMethod`          |
| `lens_spherical.m`    | `wavesim.spherical_lens_phase`           |
| `lens_spherical_MLA.m`| `wavesim.hex_mla_phase`                  |
| `compound_test.m`     | `examples/compound_test.py`              |

The physics and numerics are a direct, faithful port (same transfer
function, same evanescent-decay handling, same hex-lattice/lenslet-cluster
construction) — only the plumbing changed (tensors instead of matrices,
explicit `Grid`/propagator objects instead of recomputing frequency grids
on every call).

## Install

```bash
pip install -e .
# or
pip install -r requirements.txt
```

GPU use is automatic: `Grid`/`Field` pick CUDA > Apple MPS > CPU unless you
pass `device=...` explicitly (see `wavesim.get_device`).

## Quick start

```python
import torch
from wavesim import Grid, Field, AngularSpectrumMethod, spherical_lens_phase

grid = Grid.from_extent(extent_x=4e-3, nx=2001)          # 4mm x 4mm, 2001x2001
wavelength = 520e-9

asm = AngularSpectrumMethod(grid, wavelength)              # built once, reused
lens = spherical_lens_phase(grid, wavelength, focal_length=11e-3, diameter=4e-3)

field = Field.point_source(grid, wavelength, radius=1e-6)
field = field.propagate(asm, z=11e-3).apply(lens).propagate(asm, z=11e-3)

intensity = field.intensity_numpy()   # numpy array, ready to imshow/save
```

Sweeping a focal/depth stack is a single batched call rather than a Python
loop — pass a 1D tensor of z values and get a `(len(z), Ny, Nx)` field back:

```python
depths = torch.linspace(-0.2e-3, 0.2e-3, 21)
stack = asm.propagate(field.data, depths)  # (21, Ny, Nx)
```

## Design notes

- **Angular Spectrum Method** (`propagation.py`) precomputes the
  z-independent parts of the transfer function (`kz`, evanescent decay
  rate) once per `AngularSpectrumMethod(grid, wavelength)` instance, so
  repeated propagation calls (different z, or a whole depth stack in one
  batched call) are cheap. Evanescent orders are rolled off with a decaying
  exponential rather than discarded, matching `ASM.m` exactly.
- **Zero padding** is already wired in (`pad_factor=` on
  `AngularSpectrumMethod`) but defaults to off (`1.0`); this is one of the
  effects to be swept later.
- **Lens pitch, aperture size, focal length, level/mode** are all plain
  function arguments on `spherical_lens_phase`/`hex_mla_phase` — no need to
  edit the library to try a new design.
- **Precision**: `Grid(..., precision="single"|"double")`. `"single"`
  (complex64) is the default and is what you want for GPU throughput;
  switch to `"double"` for numerically sensitive checks (the test suite
  uses double precision on CPU for tight tolerances).
- **`Propagator`** is an ABC so Fresnel and Fraunhofer propagators can be
  added later as siblings of `AngularSpectrumMethod` without touching
  `Field` or any calling code.

## Tests

```bash
pytest tests/
```

Covers: identity at z=0, Parseval energy conservation for a propagating
(non-evanescent) field, batched-z vs. looped-z equivalence, zero-padding
shape/consistency, and lens aperture-clipping / hex-cluster geometry.

## Example

```bash
python examples/compound_test.py
```

Ports the MATLAB `compound_test.m` cascade (point source -> Fourier lens ->
hex MLA -> sensor) plus a small batched defocus sweep, saving figures to
`examples/out/`. Note: with the original script's parameters the grid is
~14000x14000 samples (matching the MATLAB `matrixsize` sizing) — run this on
a machine with a GPU (or cut `pxsize_img`/FOV down) rather than on CPU only.

## Roadmap

- Fresnel and Fraunhofer propagators behind the same `Propagator` interface.
- Lens-design comparisons (spherical vs. hex MLA vs. others), pitch-effect
  sweeps, and zero-padding-effect studies — planned next, on top of this
  platform.
