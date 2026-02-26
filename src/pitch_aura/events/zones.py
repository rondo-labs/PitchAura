"""
Project: PitchAura
File Created: 2026-02-26
Author: Xingnan Zhu
File Name: zones.py
Description:
    Zone-based event statistics and kernel density estimation.
    Provides coarse zone counts and smooth event density surfaces
    reusing the ProbabilityGrid type for seamless visualisation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from pitch_aura._grid import make_grid
from pitch_aura.types import EventRecord, PitchSpec, ProbabilityGrid


def _filter_events(
    events: list[EventRecord],
    event_types: tuple[str, ...] | None,
    team_id: str | None,
) -> list[EventRecord]:
    """Filter events by type and team, keeping only those with coordinates."""
    filtered: list[EventRecord] = []
    allowed = {et.lower() for et in event_types} if event_types is not None else None
    for ev in events:
        if ev.coordinates is None:
            continue
        if allowed is not None and ev.event_type.lower() not in allowed:
            continue
        if team_id is not None and ev.team_id != team_id:
            continue
        filtered.append(ev)
    return filtered


def zone_counts(
    events: list[EventRecord],
    *,
    pitch: PitchSpec,
    nx: int = 6,
    ny: int = 3,
    event_types: tuple[str, ...] | None = None,
    team_id: str | None = None,
) -> pd.DataFrame:
    """Divide the pitch into *nx* x *ny* zones and count events per zone.

    Parameters:
        events:      List of event records.
        pitch:       Pitch specification.
        nx:          Number of zones along x-axis.
        ny:          Number of zones along y-axis.
        event_types: Filter to these event types (``None`` = all).
        team_id:     Filter to this team (``None`` = all).

    Returns:
        DataFrame with columns ``zone_x``, ``zone_y``, ``x_center``,
        ``y_center``, ``count``, ``frequency``.
    """
    cols = ["zone_x", "zone_y", "x_center", "y_center", "count", "frequency"]
    filtered = _filter_events(events, event_types, team_id)

    x_min, x_max = pitch.x_range
    y_min, y_max = pitch.y_range
    x_edges = np.linspace(x_min, x_max, nx + 1)
    y_edges = np.linspace(y_min, y_max, ny + 1)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0

    counts = np.zeros((nx, ny), dtype=int)

    for ev in filtered:
        x, y = ev.coordinates[0], ev.coordinates[1]  # type: ignore[index]
        xi = int(np.clip(np.searchsorted(x_edges, x, side="right") - 1, 0, nx - 1))
        yi = int(np.clip(np.searchsorted(y_edges, y, side="right") - 1, 0, ny - 1))
        counts[xi, yi] += 1

    total = counts.sum()
    rows = []
    for i in range(nx):
        for j in range(ny):
            rows.append({
                "zone_x": i,
                "zone_y": j,
                "x_center": float(x_centers[i]),
                "y_center": float(y_centers[j]),
                "count": int(counts[i, j]),
                "frequency": float(counts[i, j] / total) if total > 0 else 0.0,
            })

    return pd.DataFrame(rows, columns=cols)


def event_density(
    events: list[EventRecord],
    *,
    pitch: PitchSpec,
    resolution: tuple[int, int] = (50, 34),
    sigma: float = 5.0,
    event_types: tuple[str, ...] | None = None,
    team_id: str | None = None,
) -> ProbabilityGrid:
    """Generate a smooth event density surface via Gaussian kernel estimation.

    The resulting :class:`~pitch_aura.types.ProbabilityGrid` is normalised
    to ``[0, 1]`` and can be rendered directly with
    :func:`~pitch_aura.viz.heatmap.plot_heatmap`.

    Parameters:
        events:      List of event records.
        pitch:       Pitch specification.
        resolution:  ``(nx, ny)`` grid cells.
        sigma:       Gaussian kernel standard deviation in meters.
        event_types: Filter to these event types (``None`` = all).
        team_id:     Filter to this team (``None`` = all).

    Returns:
        :class:`ProbabilityGrid` with density values in ``[0, 1]``.
    """
    filtered = _filter_events(events, event_types, team_id)
    _, x_edges, y_edges = make_grid(pitch, resolution)
    nx, ny = resolution

    raw = np.zeros((nx, ny), dtype=np.float64)

    for ev in filtered:
        x, y = ev.coordinates[0], ev.coordinates[1]  # type: ignore[index]
        xi = int(np.clip(np.searchsorted(x_edges, x, side="right") - 1, 0, nx - 1))
        yi = int(np.clip(np.searchsorted(y_edges, y, side="right") - 1, 0, ny - 1))
        raw[xi, yi] += 1.0

    # Convert sigma from meters to grid cells
    dx = float(x_edges[1] - x_edges[0])
    dy = float(y_edges[1] - y_edges[0])
    sigma_cells = (sigma / dx, sigma / dy)

    smoothed = gaussian_filter(raw, sigma=sigma_cells)

    # Normalise to [0, 1]
    max_val = smoothed.max()
    if max_val > 0:
        smoothed /= max_val

    return ProbabilityGrid(
        values=smoothed,
        x_edges=x_edges,
        y_edges=y_edges,
        pitch=pitch,
        timestamp=0.0,
    )
