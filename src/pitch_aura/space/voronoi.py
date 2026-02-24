"""
Project: PitchAura
File Created: 2026-02-23
Author: Xingnan Zhu
File Name: voronoi.py
Description:
    Voronoi tessellation model for spatial division.
    Uses mirror-point augmentation + Sutherland-Hodgman clipping to
    produce finite, pitch-bounded Voronoi regions without extra deps.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import Voronoi

from pitch_aura.types import FrameRecord, PitchSpec, VoronoiResult


# ---------------------------------------------------------------------------
# Internal geometry helpers
# ---------------------------------------------------------------------------

def _polygon_area(vertices: np.ndarray) -> float:
    """Shoelace formula for the signed area of a polygon."""
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _clip_polygon_to_rect(
    polygon: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> np.ndarray | None:
    """Sutherland-Hodgman clip of a convex polygon to an axis-aligned rectangle.

    Parameters:
        polygon: ``(M, 2)`` array of vertices.
        x_min, x_max, y_min, y_max: Clipping bounds.

    Returns:
        Clipped polygon as ``(K, 2)`` array, or ``None`` if fully outside.
    """

    def _intersect(p1: np.ndarray, p2: np.ndarray, axis: int, value: float) -> np.ndarray:
        """Intersection of segment p1-p2 with the line axis==value."""
        t = (value - p1[axis]) / (p2[axis] - p1[axis])
        return p1 + t * (p2 - p1)

    def _clip_by_plane(
        poly: list[np.ndarray], axis: int, value: float, keep_above: bool
    ) -> list[np.ndarray]:
        if not poly:
            return []
        result: list[np.ndarray] = []
        n = len(poly)
        for i in range(n):
            curr = poly[i]
            prev = poly[i - 1]
            c_in = (curr[axis] >= value) if keep_above else (curr[axis] <= value)
            p_in = (prev[axis] >= value) if keep_above else (prev[axis] <= value)
            if c_in:
                if not p_in:
                    result.append(_intersect(prev, curr, axis, value))
                result.append(curr)
            elif p_in:
                result.append(_intersect(prev, curr, axis, value))
        return result

    verts = [polygon[i] for i in range(len(polygon))]
    verts = _clip_by_plane(verts, axis=0, value=x_min, keep_above=True)
    verts = _clip_by_plane(verts, axis=0, value=x_max, keep_above=False)
    verts = _clip_by_plane(verts, axis=1, value=y_min, keep_above=True)
    verts = _clip_by_plane(verts, axis=1, value=y_max, keep_above=False)

    if len(verts) < 3:
        return None
    return np.array(verts)


def _bounded_voronoi_regions(
    points: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> list[np.ndarray | None]:
    """Compute finite, pitch-clipped Voronoi regions for each input point.

    Strategy: augment with mirror reflections across each pitch wall so
    all Voronoi cells of original points become finite, then clip each
    region polygon to the pitch rectangle.

    Parameters:
        points: ``(N, 2)`` array of player positions.
        x_min, x_max, y_min, y_max: Pitch bounds.

    Returns:
        List of ``N`` polygon arrays (each ``(M, 2)``) in the same order as
        ``points``, or ``None`` where clipping yields an empty polygon.
    """
    n = len(points)

    # Mirror reflections across each boundary wall (4 sets of N ghost points)
    mirror_left   = np.column_stack([2 * x_min - points[:, 0], points[:, 1]])
    mirror_right  = np.column_stack([2 * x_max - points[:, 0], points[:, 1]])
    mirror_bottom = np.column_stack([points[:, 0], 2 * y_min - points[:, 1]])
    mirror_top    = np.column_stack([points[:, 0], 2 * y_max - points[:, 1]])

    augmented = np.vstack([points, mirror_left, mirror_right, mirror_bottom, mirror_top])

    vor = Voronoi(augmented)

    regions: list[np.ndarray | None] = []
    for i in range(n):
        region_idx = vor.point_region[i]
        vertex_indices = vor.regions[region_idx]

        # Skip degenerate regions (should not occur with mirror augmentation)
        if -1 in vertex_indices or len(vertex_indices) < 3:
            regions.append(None)
            continue

        polygon = vor.vertices[vertex_indices]
        clipped = _clip_polygon_to_rect(polygon, x_min, x_max, y_min, y_max)
        regions.append(clipped)

    return regions


# ---------------------------------------------------------------------------
# Public model
# ---------------------------------------------------------------------------

class VoronoiModel:
    """Voronoi-based spatial control model.

    Divides the pitch into regions closest to each player using Voronoi
    tessellation, clipped to pitch boundaries. Fast and suitable for
    large-batch coarse feature extraction.

    Parameters:
        pitch: Pitch specification. If ``None``, uses the frame's pitch.
    """

    def __init__(self, *, pitch: PitchSpec | None = None) -> None:
        self._pitch = pitch

    def _resolve_pitch(self) -> PitchSpec:
        return self._pitch if self._pitch is not None else PitchSpec()

    def control(self, frame: FrameRecord) -> VoronoiResult:
        """Compute Voronoi tessellation for a single frame.

        Parameters:
            frame: A :class:`FrameRecord` with player positions.

        Returns:
            :class:`VoronoiResult` containing per-player polygon vertices,
            areas, and per-team total areas.
        """
        pitch = self._resolve_pitch()
        x_min, x_max = pitch.x_range
        y_min, y_max = pitch.y_range

        raw_regions = _bounded_voronoi_regions(
            frame.positions, x_min, x_max, y_min, y_max
        )

        region_map: dict[str, np.ndarray] = {}
        area_map: dict[str, float] = {}
        team_area_map: dict[str, float] = {}

        for idx, polygon in enumerate(raw_regions):
            pid = frame.player_ids[idx]
            tid = frame.team_ids[idx]

            if polygon is None:
                region_map[pid] = np.empty((0, 2))
                area_map[pid] = 0.0
            else:
                # Ensure counter-clockwise winding for consistent area sign
                area_signed = _polygon_area(polygon)
                if area_signed < 0:
                    polygon = polygon[::-1]
                area = abs(area_signed)
                region_map[pid] = polygon
                area_map[pid] = area

            team_area_map[tid] = team_area_map.get(tid, 0.0) + area_map[pid]

        return VoronoiResult(
            regions=region_map,
            areas=area_map,
            team_areas=team_area_map,
            timestamp=frame.timestamp,
        )

    def control_batch(self, frames: list[FrameRecord]) -> list[VoronoiResult]:
        """Compute Voronoi tessellation for a sequence of frames.

        Parameters:
            frames: List of :class:`FrameRecord` objects.

        Returns:
            List of :class:`VoronoiResult` in the same order.
        """
        return [self.control(f) for f in frames]
