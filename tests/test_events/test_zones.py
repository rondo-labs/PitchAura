"""
Project: PitchAura
File Created: 2026-02-26
Author: Xingnan Zhu
File Name: test_zones.py
Description:
    Tests for zone-based event statistics and density estimation.
"""

from __future__ import annotations

import numpy as np
import pytest

from pitch_aura.events.zones import event_density, zone_counts
from pitch_aura.types import EventRecord, PitchSpec, ProbabilityGrid


@pytest.fixture
def pitch() -> PitchSpec:
    return PitchSpec(length=105.0, width=68.0, origin="bottom_left")


def _event_at(x: float, y: float, event_type: str = "pass", team_id: str = "t1") -> EventRecord:
    return EventRecord(
        timestamp=0.0,
        period=1,
        event_type=event_type,
        player_id="p1",
        team_id=team_id,
        coordinates=np.array([x, y], dtype=np.float64),
    )


class TestZoneCounts:
    def test_empty_events(self, pitch: PitchSpec):
        df = zone_counts([], pitch=pitch, nx=3, ny=2)
        assert len(df) == 6  # 3*2 zones, all with count=0
        assert (df["count"] == 0).all()

    def test_single_event_correct_zone(self, pitch: PitchSpec):
        ev = _event_at(10.0, 10.0)
        df = zone_counts([ev], pitch=pitch, nx=3, ny=3)

        # Should land in zone (0, 0) — bottom-left
        total = df["count"].sum()
        assert total == 1

        # Find the zone with count 1
        hit = df[df["count"] == 1]
        assert len(hit) == 1
        assert hit.iloc[0]["zone_x"] == 0
        assert hit.iloc[0]["zone_y"] == 0

    def test_event_at_far_corner(self, pitch: PitchSpec):
        ev = _event_at(100.0, 65.0)
        df = zone_counts([ev], pitch=pitch, nx=3, ny=3)

        hit = df[df["count"] == 1]
        assert len(hit) == 1
        assert hit.iloc[0]["zone_x"] == 2
        assert hit.iloc[0]["zone_y"] == 2

    def test_frequency_sums_to_one(self, pitch: PitchSpec):
        events = [_event_at(20.0, 20.0), _event_at(80.0, 50.0), _event_at(50.0, 34.0)]
        df = zone_counts(events, pitch=pitch, nx=6, ny=3)
        np.testing.assert_allclose(df["frequency"].sum(), 1.0)

    def test_event_type_filter(self, pitch: PitchSpec):
        events = [
            _event_at(50.0, 34.0, event_type="pass"),
            _event_at(50.0, 34.0, event_type="shot"),
        ]
        df = zone_counts(events, pitch=pitch, nx=3, ny=3, event_types=("pass",))
        assert df["count"].sum() == 1

    def test_team_filter(self, pitch: PitchSpec):
        events = [
            _event_at(50.0, 34.0, team_id="t1"),
            _event_at(50.0, 34.0, team_id="t2"),
        ]
        df = zone_counts(events, pitch=pitch, nx=3, ny=3, team_id="t1")
        assert df["count"].sum() == 1

    def test_events_without_coordinates_skipped(self, pitch: PitchSpec):
        ev = EventRecord(
            timestamp=0.0, period=1, event_type="pass",
            coordinates=None,
        )
        df = zone_counts([ev], pitch=pitch)
        assert df["count"].sum() == 0


class TestEventDensity:
    def test_returns_probability_grid(self, pitch: PitchSpec):
        events = [_event_at(50.0, 34.0)]
        grid = event_density(events, pitch=pitch, resolution=(10, 7))

        assert isinstance(grid, ProbabilityGrid)
        assert grid.values.shape == (10, 7)
        assert grid.pitch == pitch

    def test_normalised_to_zero_one(self, pitch: PitchSpec):
        events = [_event_at(50.0, 34.0)] * 10
        grid = event_density(events, pitch=pitch, resolution=(20, 14))

        assert grid.values.min() >= 0.0
        assert grid.values.max() <= 1.0
        np.testing.assert_allclose(grid.values.max(), 1.0)

    def test_empty_events_all_zero(self, pitch: PitchSpec):
        grid = event_density([], pitch=pitch, resolution=(10, 7))
        assert grid.values.max() == 0.0

    def test_peak_near_event_location(self, pitch: PitchSpec):
        events = [_event_at(52.5, 34.0)] * 20
        grid = event_density(events, pitch=pitch, resolution=(50, 34), sigma=3.0)

        # Peak should be near center of pitch
        peak_idx = np.unravel_index(np.argmax(grid.values), grid.values.shape)
        peak_x = grid.x_centers[peak_idx[0]]
        peak_y = grid.y_centers[peak_idx[1]]
        assert abs(peak_x - 52.5) < 5.0
        assert abs(peak_y - 34.0) < 5.0

    def test_event_type_filter(self, pitch: PitchSpec):
        events = [
            _event_at(50.0, 34.0, event_type="pass"),
            _event_at(50.0, 34.0, event_type="shot"),
        ]
        grid_all = event_density(events, pitch=pitch, resolution=(10, 7))
        grid_pass = event_density(
            events, pitch=pitch, resolution=(10, 7), event_types=("pass",),
        )

        # All events should have higher density than pass-only
        assert grid_all.values.sum() >= grid_pass.values.sum()
