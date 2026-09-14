"""Backend utilities: device selection, dtype management, array conversion.

The simulator is built on PyTorch so that every operation -- FFTs included --
transparently runs on GPU (CUDA or Apple MPS) when one is available, and
falls back to CPU otherwise, with no code changes required from the caller.
Using an autograd-capable tensor library also leaves the door open for
future gradient-based phase-map design/optimization on top of this
platform.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

__all__ = ["get_device", "get_dtypes", "to_tensor", "as_numpy", "phasor"]


def get_device(prefer: Optional[str] = None) -> torch.device:
    """Pick the best available compute device.

    Parameters
    ----------
    prefer : str, optional
        Force a specific device string (e.g. "cuda", "cuda:1", "mps", "cpu").
        If None, auto-selects CUDA > MPS > CPU.
    """
    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_dtypes(precision: str = "single") -> Tuple[torch.dtype, torch.dtype]:
    """Return (real_dtype, complex_dtype) for a requested precision.

    "single" -> float32 / complex64  (fast, recommended for GPU)
    "double" -> float64 / complex128 (higher accuracy, slower, more memory)
    """
    if precision == "single":
        return torch.float32, torch.complex64
    if precision == "double":
        return torch.float64, torch.complex128
    raise ValueError(f"Unknown precision {precision!r}; expected 'single' or 'double'.")


def to_tensor(array, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Convert a numpy array / python scalar / tensor to a tensor on device+dtype."""
    if isinstance(array, torch.Tensor):
        return array.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(array), device=device, dtype=dtype)


def as_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Detach a (possibly complex) tensor to a numpy array on CPU."""
    return tensor.detach().to("cpu").numpy()


def phasor(phase: torch.Tensor) -> torch.Tensor:
    """exp(i*phase) built from real cos/sin so the output dtype stays at the
    same precision as `phase` (avoids an accidental promotion to complex128
    that plain ``torch.exp(1j * phase)`` triggers, since python's ``1j`` is
    complex128 by default).
    """
    return torch.complex(torch.cos(phase), torch.sin(phase))
