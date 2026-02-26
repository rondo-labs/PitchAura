"""
Project: PitchAura
File Created: 2026-02-23
Author: Xingnan Zhu
File Name: types.py
Description:
    Core data models for pitch_aura.
    All types are designed around contiguous NumPy arrays for efficient
    vectorized computation. Positions are in meters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import overload

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class PitchSpec:
    """Pitch dimensions and coordinate system.

    Attributes:
        length: Pitch length in meters (x-axis). Default 105m (FIFA standard).
        width: Pitch width in meters (y-axis). Default 68m (FIFA standard).
        origin: Coordinate origin — ``"center"`` or ``"bottom_left"``.
    """

    length: float = 105.0
    width: float = 68.0
    origin: str = "center"

    @property
    def x_range(self) -> tuple[float, float]:
        if self.origin == "center":
            return (-self.length / 2, self.length / 2)
        return (0.0, self.length)

    @property
    def y_range(self) -> tuple[float, float]:
        if self.origin == "center":
            return (-self.width / 2, self.width / 2)
        return (0.0, self.width)

    @property
    def area(self) -> float:
        return self.length * self.width


@dataclass(slots=True)
class FrameRecord:
    """One instant of match state, optimized for vectorized computation.

    All positional arrays are in meters. Player data is stored as
    contiguous arrays indexed identically — ``positions[i]`` corresponds
    to ``player_ids[i]``.
    """

    timestamp: float
    """Seconds from period start."""
    period: int
    """Match period (1 or 2)."""
    ball_position: np.ndarray
    """Ball coordinates, shape ``(2,)`` or ``(3,)`` if z available."""

    player_ids: list[str]
    """Player identifiers, length N."""
    team_ids: list[str]
    """Team identifier per player, length N."""
    positions: np.ndarray
    """Player positions, shape ``(N, 2)``."""
    velocities: np.ndarray | None = None
    """Player velocities in m/s, shape ``(N, 2)`` or ``None``."""
    is_goalkeeper: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    """Boolean mask for goalkeepers, shape ``(N,)``."""

    @property
    def n_players(self) -> int:
        return self.positions.shape[0]

    def team_mask(self, team_id: str) -> np.ndarray:
        """Boolean mask selecting players on *team_id*."""
        return np.array([t == team_id for t in self.team_ids], dtype=bool)

    def team_positions(self, team_id: str) -> np.ndarray:
        """Positions of players on *team_id*, shape ``(M, 2)``."""
        return self.positions[self.team_mask(team_id)]

    def team_velocities(self, team_id: str) -> np.ndarray | None:
        """Velocities of players on *team_id*, shape ``(M, 2)`` or ``None``."""
        if self.velocities is None:
            return None
        return self.velocities[self.team_mask(team_id)]


@dataclass(slots=True)
class FrameSequence:
    """Ordered collection of :class:`FrameRecord` with match metadata."""

    frames: list[FrameRecord]
    frame_rate: float
    """Frames per second (e.g. 25.0)."""
    pitch: PitchSpec
    home_team_id: str
    away_team_id: str

    @property
    def timestamps(self) -> np.ndarray:
        return np.array([f.timestamp for f in self.frames])

    def slice_time(self, t_start: float, t_end: float) -> FrameSequence:
        """Return sub-sequence within ``[t_start, t_end]``."""
        selected = [f for f in self.frames if t_start <= f.timestamp <= t_end]
        return FrameSequence(
            frames=selected,
            frame_rate=self.frame_rate,
            pitch=self.pitch,
            home_team_id=self.home_team_id,
            away_team_id=self.away_team_id,
        )

    def __len__(self) -> int:
        return len(self.frames)

    @overload
    def __getitem__(self, idx: int) -> FrameRecord: ...
    @overload
    def __getitem__(self, idx: slice) -> FrameSequence: ...

    def __getitem__(self, idx: int | slice) -> FrameRecord | FrameSequence:
        if isinstance(idx, slice):
            return FrameSequence(
                frames=self.frames[idx],
                frame_rate=self.frame_rate,
                pitch=self.pitch,
                home_team_id=self.home_team_id,
                away_team_id=self.away_team_id,
            )
        return self.frames[idx]


@dataclass(slots=True)
class ProbabilityGrid:
    """2D probability surface over the pitch.

    ``values[i, j]`` is the probability that a team controls the cell
    bounded by ``x_edges[i:i+2]`` and ``y_edges[j:j+2]``.
    """

    values: np.ndarray
    """Control probabilities, shape ``(nx, ny)``, values in ``[0, 1]``."""
    x_edges: np.ndarray
    """Cell boundaries along x-axis, shape ``(nx + 1,)``."""
    y_edges: np.ndarray
    """Cell boundaries along y-axis, shape ``(ny + 1,)``."""
    pitch: PitchSpec
    timestamp: float

    @property
    def resolution(self) -> tuple[int, int]:
        return self.values.shape  # type: ignore[return-value]

    @property
    def x_centers(self) -> np.ndarray:
        return (self.x_edges[:-1] + self.x_edges[1:]) / 2.0

    @property
    def y_centers(self) -> np.ndarray:
        return (self.y_edges[:-1] + self.y_edges[1:]) / 2.0

    @property
    def cell_area(self) -> float:
        """Area of a single cell in m²."""
        dx = float(self.x_edges[1] - self.x_edges[0])
        dy = float(self.y_edges[1] - self.y_edges[0])
        return dx * dy

    def total_area(self, threshold: float = 0.5) -> float:
        """Total area (m²) where probability exceeds *threshold*."""
        return float(np.sum(self.values >= threshold) * self.cell_area)

    def to_dataframe(self) -> pd.DataFrame:
        """Melt grid to long-form DataFrame with columns ``[x, y, probability]``."""
        xx, yy = np.meshgrid(self.x_centers, self.y_centers, indexing="ij")
        return pd.DataFrame(
            {
                "x": xx.ravel(),
                "y": yy.ravel(),
                "probability": self.values.ravel(),
            }
        )


@dataclass(slots=True)
class VoronoiResult:
    """Result of a Voronoi tessellation on a single frame."""

    regions: dict[str, np.ndarray]
    """Mapping from player_id to polygon vertices, each shape ``(M, 2)``."""
    areas: dict[str, float]
    """Mapping from player_id to controlled area in m²."""
    team_areas: dict[str, float]
    """Mapping from team_id to total controlled area in m²."""
    timestamp: float


@dataclass(frozen=True, slots=True)
class EventRecord:
    """Event representation with spatial context for event-based analysis.

    Core fields capture temporal and spatial identity.  Optional fields
    carry richer spatial information extracted from providers such as
    StatsBomb, Opta, and Wyscout via kloppy.
    """

    timestamp: float
    """Seconds from period start."""
    period: int
    event_type: str
    player_id: str | None = None
    team_id: str | None = None
    coordinates: np.ndarray | None = None
    """Event start location, shape ``(2,)`` or ``None``."""
    end_coordinates: np.ndarray | None = None
    """Event end location (pass receive point / carry end / shot target),
    shape ``(2,)`` or ``None``."""
    result: str | None = None
    """Outcome string, e.g. ``"complete"``, ``"goal"``, ``"incomplete"``."""
    qualifiers: tuple[str, ...] = ()
    """Provider qualifiers, e.g. ``("CROSS", "THROUGH_BALL")``."""
    freeze_frame: FrameRecord | None = None
    """Player position snapshot at the moment of the event."""
