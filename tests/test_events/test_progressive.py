"""
Project: PitchAura
File Created: 2026-02-26
Author: Xingnan Zhu
File Name: test_progressive.py
Description:
    Tests for progressive distance metrics.
"""

from __future__ import annotations

import numpy as np
import pytest

from pitch_aura.events.progressive import progressive_actions
from pitch_aura.types import EventRecord, PitchSpec


@pytest.fixture
def pitch() -> PitchSpec:
    return PitchSpec(length=105.0, width=68.0, origin="bottom_left")


def _pass_event(
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    timestamp: float = 0.0,
    event_type: str = "pass",
    player_id: str = "p1",
    team_id: str = "t1",
) -> EventRecord:
    return EventRecord(
        timestamp=timestamp,
        period=1,
        event_type=event_type,
        player_id=player_id,
        team_id=team_id,
        coordinates=np.array(start, dtype=np.float64),
        end_coordinates=np.array(end, dtype=np.float64),
        result="complete",
    )


class TestProgressiveActions:
    def test_empty_events(self, pitch: PitchSpec):
        df = progressive_actions([], pitch=pitch)
        assert df.empty
        assert list(df.columns) == [
            "timestamp", "period", "event_type", "player_id", "team_id",
            "start_x", "start_y", "end_x", "end_y",
            "distance", "progressive_distance", "is_progressive",
        ]

    def test_forward_pass_is_progressive(self, pitch: PitchSpec):
        ev = _pass_event((30.0, 34.0), (70.0, 34.0))
        df = progressive_actions([ev], pitch=pitch)

        assert len(df) == 1
        assert bool(df.iloc[0]["is_progressive"]) is True
        assert df.iloc[0]["progressive_distance"] > 0

    def test_backward_pass_not_progressive(self, pitch: PitchSpec):
        ev = _pass_event((70.0, 34.0), (30.0, 34.0))
        df = progressive_actions([ev], pitch=pitch)

        assert len(df) == 1
        assert bool(df.iloc[0]["is_progressive"]) is False
        assert df.iloc[0]["progressive_distance"] < 0

    def test_lateral_pass_not_progressive(self, pitch: PitchSpec):
        ev = _pass_event((50.0, 10.0), (50.0, 58.0))
        df = progressive_actions([ev], pitch=pitch)

        assert len(df) == 1
        # Lateral pass: almost no forward progress
        assert bool(df.iloc[0]["is_progressive"]) is False

    def test_filters_event_types(self, pitch: PitchSpec):
        events = [
            _pass_event((30.0, 34.0), (70.0, 34.0), event_type="pass"),
            _pass_event((30.0, 34.0), (70.0, 34.0), event_type="shot"),
        ]
        df = progressive_actions(events, pitch=pitch, event_types=("pass",))
        assert len(df) == 1

    def test_skips_events_without_end_coordinates(self, pitch: PitchSpec):
        ev = EventRecord(
            timestamp=0.0,
            period=1,
            event_type="pass",
            player_id="p1",
            team_id="t1",
            coordinates=np.array([30.0, 34.0]),
            end_coordinates=None,
        )
        df = progressive_actions([ev], pitch=pitch)
        assert df.empty

    def test_custom_target_x(self, pitch: PitchSpec):
        # Attack toward x=0 instead of x=105
        ev = _pass_event((70.0, 34.0), (30.0, 34.0))
        df = progressive_actions([ev], pitch=pitch, target_x=0.0)

        assert len(df) == 1
        assert df.iloc[0]["progressive_distance"] > 0
        assert bool(df.iloc[0]["is_progressive"]) is True

    def test_min_distance_threshold(self, pitch: PitchSpec):
        # Small forward pass
        ev = _pass_event((50.0, 34.0), (53.0, 34.0))
        # Without min_distance, might be progressive
        df1 = progressive_actions([ev], pitch=pitch, min_distance=0.0)
        # With high min_distance, should not be
        df2 = progressive_actions([ev], pitch=pitch, min_distance=20.0)

        assert bool(df1.iloc[0]["is_progressive"]) is True
        assert bool(df2.iloc[0]["is_progressive"]) is False

    def test_carry_event_type(self, pitch: PitchSpec):
        ev = _pass_event((30.0, 34.0), (60.0, 34.0), event_type="carry")
        df = progressive_actions([ev], pitch=pitch, event_types=("carry",))
        assert len(df) == 1
        assert bool(df.iloc[0]["is_progressive"]) is True

    def test_distance_column(self, pitch: PitchSpec):
        ev = _pass_event((0.0, 0.0), (3.0, 4.0))
        df = progressive_actions([ev], pitch=pitch)
        np.testing.assert_allclose(df.iloc[0]["distance"], 5.0)
