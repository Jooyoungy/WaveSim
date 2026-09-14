"""Thin matplotlib helpers, roughly mirroring MATLAB's ``imagesc``."""
from __future__ import annotations

from typing import Optional

import torch

from .backend import as_numpy
from .grid import Grid


def _extent_mm(grid: Grid):
    return [
        float(grid.x.min()) * 1e3,
        float(grid.x.max()) * 1e3,
        float(grid.y.min()) * 1e3,
        float(grid.y.max()) * 1e3,
    ]


def imshow(
    array,
    grid: Grid,
    ax=None,
    title: Optional[str] = None,
    cmap: str = "viridis",
    colorbar: bool = True,
):
    """Display a real-valued 2D array (or tensor) on grid coordinates, in mm."""
    import matplotlib.pyplot as plt

    if isinstance(array, torch.Tensor):
        array = as_numpy(array)

    if ax is None:
        _, ax = plt.subplots()

    im = ax.imshow(array, extent=_extent_mm(grid), origin="lower", cmap=cmap, aspect="equal")
    if title:
        ax.set_title(title)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    if colorbar:
        ax.figure.colorbar(im, ax=ax)
    return ax
