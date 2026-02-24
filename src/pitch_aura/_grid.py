"""
Project: PitchAura
File Created: 2026-02-23
Author: Xingnan Zhu
File Name: _grid.py
Description:
    Internal utilities for generating spatial grids.
"""

from __future__ import annotations

import numpy as np

from pitch_aura.types import PitchSpec


def make_grid(
    pitch: PitchSpec,
    resolution: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a regular grid over the pitch.

    Parameters:
        pitch: Pitch specification defining the coordinate system.
        resolution: ``(nx, ny)`` — number of cells along each axis.

    Returns:
        Tuple of ``(target_positions, x_edges, y_edges)`` where
        *target_positions* has shape ``(nx * ny, 2)`` containing the
        center of each cell, and *x_edges* / *y_edges* are the cell
        boundary arrays.
    """
    nx, ny = resolution
    x_min, x_max = pitch.x_range
    y_min, y_max = pitch.y_range

    x_edges = np.linspace(x_min, x_max, nx + 1)
    y_edges = np.linspace(y_min, y_max, ny + 1)

    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0

    # shape (nx * ny, 2) — row-major order
    xx, yy = np.meshgrid(x_centers, y_centers, indexing="ij")
    target_positions = np.column_stack([xx.ravel(), yy.ravel()])

    return target_positions, x_edges, y_edges
