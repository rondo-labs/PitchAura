"""
Project: PitchAura
File Created: 2026-02-23
Author: Xingnan Zhu
File Name: test_types.py
Description:
    Tests for core data models in pitch_aura.types.
"""

import numpy as np
import pandas as pd
import pytest

from pitch_aura.types import (
    EventRecord,
    FrameRecord,
    FrameSequence,
    PitchSpec,
    ProbabilityGrid,
    VoronoiResult,
)


class TestPitchSpec:
    def test_default_dimensions(self):
        p = PitchSpec()
        assert p.length == 105.0
        assert p.width == 68.0
        assert p.origin == "center"

    def test_area(self):
        p = PitchSpec(length=100.0, width=50.0)
        assert p.area == 5000.0

    def test_center_origin_ranges(self):
        p = PitchSpec(length=105.0, width=68.0, origin="center")
        assert p.x_range == (-52.5, 52.5)
        assert p.y_range == (-34.0, 34.0)

    def test_bottom_left_origin_ranges(self):
        p = PitchSpec(length=105.0, width=68.0, origin="bottom_left")
        assert p.x_range == (0.0, 105.0)
        assert p.y_range == (0.0, 68.0)

    def test_frozen(self):
        p = PitchSpec()
        with pytest.raises(AttributeError):
            p.length = 110.0  # type: ignore[misc]


class TestFrameRecord:
    def test_n_players(self, simple_frame: FrameRecord):
        assert simple_frame.n_players == 6

    def test_team_mask(self, simple_frame: FrameRecord):
        home_mask = simple_frame.team_mask("home")
        assert home_mask.sum() == 3
        assert home_mask.dtype == bool
        away_mask = simple_frame.team_mask("away")
        assert away_mask.sum() == 3

    def test_team_positions(self, simple_frame: FrameRecord):
        home_pos = simple_frame.team_positions("home")
        assert home_pos.shape == (3, 2)
        np.testing.assert_array_equal(home_pos[0], [-20.0, 0.0])

    def test_team_velocities(self, simple_frame: FrameRecord):
        home_vel = simple_frame.team_velocities("home")
        assert home_vel is not None
        assert home_vel.shape == (3, 2)

    def test_team_velocities_none(self):
        frame = FrameRecord(
            timestamp=0.0,
            period=1,
            ball_position=np.array([0.0, 0.0]),
            player_ids=["p1"],
            team_ids=["t1"],
            positions=np.array([[0.0, 0.0]]),
            velocities=None,
        )
        assert frame.team_velocities("t1") is None


class TestFrameSequence:
    def test_len(self, simple_sequence: FrameSequence):
        assert len(simple_sequence) == 5

    def test_getitem_int(self, simple_sequence: FrameSequence):
        frame = simple_sequence[0]
        assert isinstance(frame, FrameRecord)
        assert frame.timestamp == 0.0

    def test_getitem_slice(self, simple_sequence: FrameSequence):
        sub = simple_sequence[1:3]
        assert isinstance(sub, FrameSequence)
        assert len(sub) == 2
        assert sub.frame_rate == 25.0
        assert sub.home_team_id == "home"

    def test_timestamps(self, simple_sequence: FrameSequence):
        ts = simple_sequence.timestamps
        assert ts.shape == (5,)
        np.testing.assert_allclose(ts[1], 0.04)

    def test_slice_time(self, simple_sequence: FrameSequence):
        sub = simple_sequence.slice_time(0.04, 0.12)
        assert len(sub) >= 2
        for f in sub.frames:
            assert 0.04 <= f.timestamp <= 0.12


class TestProbabilityGrid:
    @pytest.fixture
    def grid(self, pitch: PitchSpec) -> ProbabilityGrid:
        nx, ny = 10, 7
        x_edges = np.linspace(-52.5, 52.5, nx + 1)
        y_edges = np.linspace(-34.0, 34.0, ny + 1)
        values = np.random.default_rng(42).random((nx, ny))
        return ProbabilityGrid(
            values=values,
            x_edges=x_edges,
            y_edges=y_edges,
            pitch=pitch,
            timestamp=0.0,
        )

    def test_resolution(self, grid: ProbabilityGrid):
        assert grid.resolution == (10, 7)

    def test_centers(self, grid: ProbabilityGrid):
        assert grid.x_centers.shape == (10,)
        assert grid.y_centers.shape == (7,)

    def test_cell_area(self, grid: ProbabilityGrid):
        expected = (105.0 / 10) * (68.0 / 7)
        assert abs(grid.cell_area - expected) < 1e-10

    def test_total_area(self, grid: ProbabilityGrid):
        area = grid.total_area(threshold=0.5)
        assert 0.0 <= area <= 105.0 * 68.0

    def test_to_dataframe(self, grid: ProbabilityGrid):
        df = grid.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == {"x", "y", "probability"}
        assert len(df) == 10 * 7


class TestVoronoiResult:
    def test_construction(self):
        result = VoronoiResult(
            regions={"p1": np.array([[0, 0], [1, 0], [1, 1]])},
            areas={"p1": 0.5},
            team_areas={"t1": 0.5},
            timestamp=0.0,
        )
        assert result.areas["p1"] == 0.5


class TestEventRecord:
    def test_basic(self):
        event = EventRecord(
            timestamp=10.5,
            period=1,
            event_type="pass",
            player_id="p1",
            team_id="t1",
            coordinates=np.array([30.0, 20.0]),
        )
        assert event.event_type == "pass"
        assert event.coordinates is not None
        assert event.coordinates.shape == (2,)

    def test_minimal(self):
        event = EventRecord(timestamp=0.0, period=1, event_type="whistle")
        assert event.player_id is None
        assert event.coordinates is None
