"""
Project: PitchAura
File Created: 2026-02-23
Author: Xingnan Zhu
File Name: test_voronoi.py
Description:
    Tests for VoronoiModel and internal geometry helpers.
    Uses analytically verifiable configurations to validate correctness.
"""

from __future__ import annotations

import numpy as np
import pytest

from pitch_aura.space.voronoi import (
    VoronoiModel,
    _bounded_voronoi_regions,
    _clip_polygon_to_rect,
    _polygon_area,
)
from pitch_aura.types import FrameRecord, PitchSpec, VoronoiResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_frame(
    positions: list[tuple[float, float]],
    player_ids: list[str] | None = None,
    team_ids: list[str] | None = None,
) -> FrameRecord:
    n = len(positions)
    return FrameRecord(
        timestamp=0.0,
        period=1,
        ball_position=np.array([0.0, 0.0]),
        player_ids=player_ids or [f"p{i}" for i in range(n)],
        team_ids=team_ids or ["t1"] * n,
        positions=np.array(positions, dtype=np.float64),
        is_goalkeeper=np.zeros(n, dtype=bool),
    )


# ---------------------------------------------------------------------------
# Tests: _polygon_area
# ---------------------------------------------------------------------------

class TestPolygonArea:
    def test_unit_square(self):
        sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        assert abs(_polygon_area(sq)) == pytest.approx(1.0)

    def test_known_triangle(self):
        tri = np.array([[0, 0], [4, 0], [0, 3]], dtype=float)
        assert abs(_polygon_area(tri)) == pytest.approx(6.0)

    def test_winding_sign(self):
        # CW should give negative, CCW positive
        ccw = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        cw = ccw[::-1]
        assert _polygon_area(ccw) > 0
        assert _polygon_area(cw) < 0


# ---------------------------------------------------------------------------
# Tests: _clip_polygon_to_rect
# ---------------------------------------------------------------------------

class TestClipPolygonToRect:
    def test_fully_inside(self):
        poly = np.array([[1, 1], [3, 1], [3, 3], [1, 3]], dtype=float)
        clipped = _clip_polygon_to_rect(poly, 0, 5, 0, 5)
        assert clipped is not None
        assert abs(_polygon_area(clipped)) == pytest.approx(4.0, rel=1e-6)

    def test_fully_outside(self):
        poly = np.array([[6, 6], [8, 6], [8, 8], [6, 8]], dtype=float)
        clipped = _clip_polygon_to_rect(poly, 0, 5, 0, 5)
        assert clipped is None

    def test_half_cut(self):
        # Square from x=3..7, clipped to x=0..5 -> strips right half
        poly = np.array([[3, 0], [7, 0], [7, 4], [3, 4]], dtype=float)
        clipped = _clip_polygon_to_rect(poly, 0, 5, 0, 5)
        assert clipped is not None
        # Remaining area: 2 wide × 4 tall = 8
        assert abs(_polygon_area(clipped)) == pytest.approx(8.0, rel=1e-6)

    def test_corner_overlap(self):
        # Diamond centered at (5, 5), overlapping corner of [0,5]×[0,5]
        poly = np.array([[5, 3], [7, 5], [5, 7], [3, 5]], dtype=float)
        clipped = _clip_polygon_to_rect(poly, 0, 5, 0, 5)
        assert clipped is not None
        assert abs(_polygon_area(clipped)) > 0


# ---------------------------------------------------------------------------
# Tests: _bounded_voronoi_regions
# ---------------------------------------------------------------------------

class TestBoundedVoronoiRegions:
    def test_two_players_symmetric(self):
        """Two players symmetric about x=0 → equal areas covering the pitch."""
        points = np.array([[-10.0, 0.0], [10.0, 0.0]])
        regions = _bounded_voronoi_regions(points, -20, 20, -10, 10)
        assert len(regions) == 2
        assert all(r is not None for r in regions)

        area0 = abs(_polygon_area(regions[0]))
        area1 = abs(_polygon_area(regions[1]))
        assert area0 == pytest.approx(area1, rel=1e-6)
        # Together should cover the full pitch area: 40 × 20 = 800
        assert area0 + area1 == pytest.approx(800.0, rel=1e-4)

    def test_regions_count_matches_players(self):
        pts = np.array([[0.0, 0.0], [30.0, 0.0], [-30.0, 0.0]], dtype=float)
        regions = _bounded_voronoi_regions(pts, -52.5, 52.5, -34.0, 34.0)
        assert len(regions) == 3


# ---------------------------------------------------------------------------
# Tests: VoronoiModel.control
# ---------------------------------------------------------------------------

class TestVoronoiModelControl:
    @pytest.fixture
    def pitch(self) -> PitchSpec:
        return PitchSpec(length=100.0, width=50.0, origin="center")

    @pytest.fixture
    def model(self, pitch: PitchSpec) -> VoronoiModel:
        return VoronoiModel(pitch=pitch)

    def test_returns_voronoi_result(self, model: VoronoiModel):
        frame = _make_frame([(-10.0, 0.0), (10.0, 0.0)])
        result = model.control(frame)
        assert isinstance(result, VoronoiResult)

    def test_player_ids_present(self, model: VoronoiModel):
        frame = _make_frame([(-10.0, 0.0), (10.0, 0.0)], player_ids=["alice", "bob"])
        result = model.control(frame)
        assert "alice" in result.regions
        assert "bob" in result.regions

    def test_areas_positive(self, model: VoronoiModel):
        frame = _make_frame([(-10.0, 5.0), (10.0, 5.0), (-10.0, -5.0), (10.0, -5.0)])
        result = model.control(frame)
        for pid, area in result.areas.items():
            assert area > 0, f"Player {pid} has non-positive area {area}"

    def test_total_area_equals_pitch(self, model: VoronoiModel, pitch: PitchSpec):
        """Sum of all player areas should equal total pitch area."""
        frame = _make_frame(
            [(-20.0, -10.0), (20.0, -10.0), (-20.0, 10.0), (20.0, 10.0)]
        )
        result = model.control(frame)
        total = sum(result.areas.values())
        assert total == pytest.approx(pitch.area, rel=1e-3)

    def test_team_areas_sum_to_pitch(self, model: VoronoiModel, pitch: PitchSpec):
        frame = _make_frame(
            [(-10.0, 0.0), (-5.0, 0.0), (5.0, 0.0), (10.0, 0.0)],
            team_ids=["home", "home", "away", "away"],
        )
        result = model.control(frame)
        total = sum(result.team_areas.values())
        assert total == pytest.approx(pitch.area, rel=1e-3)

    def test_symmetric_split_gives_equal_team_areas(self, model: VoronoiModel):
        """Perfectly symmetric 3v3 → both teams control equal area."""
        frame = _make_frame(
            [(-30.0, 0.0), (-30.0, 15.0), (-30.0, -15.0),
             (30.0, 0.0),  (30.0, 15.0),  (30.0, -15.0)],
            team_ids=["home"] * 3 + ["away"] * 3,
        )
        result = model.control(frame)
        home_area = result.team_areas.get("home", 0.0)
        away_area = result.team_areas.get("away", 0.0)
        assert home_area == pytest.approx(away_area, rel=1e-3)

    def test_timestamp_preserved(self, model: VoronoiModel):
        frame = _make_frame([(-10.0, 0.0), (10.0, 0.0)])
        frame.timestamp = 42.5
        result = model.control(frame)
        assert result.timestamp == 42.5

    def test_default_pitch_is_standard(self):
        """VoronoiModel with no pitch arg defaults to 105×68."""
        model = VoronoiModel()
        frame = _make_frame([(-20.0, 0.0), (20.0, 0.0)])
        result = model.control(frame)
        total = sum(result.areas.values())
        assert total == pytest.approx(105.0 * 68.0, rel=1e-3)

    def test_control_batch(self, model: VoronoiModel):
        frames = [_make_frame([(-10.0, 0.0), (10.0, 0.0)]) for _ in range(5)]
        results = model.control_batch(frames)
        assert len(results) == 5
        assert all(isinstance(r, VoronoiResult) for r in results)
