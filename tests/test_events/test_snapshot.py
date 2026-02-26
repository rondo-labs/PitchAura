"""
Project: PitchAura
File Created: 2026-02-26
Author: Xingnan Zhu
File Name: test_snapshot.py
Description:
    Tests for freeze-frame spatial analysis utilities.
"""

from __future__ import annotations

import numpy as np
import pytest

from pitch_aura.events.snapshot import batch_event_control, event_control
from pitch_aura.types import EventRecord, FrameRecord, PitchSpec, ProbabilityGrid


@pytest.fixture
def pitch() -> PitchSpec:
    return PitchSpec(length=105.0, width=68.0, origin="bottom_left")


def _frame_record() -> FrameRecord:
    """Create a minimal frame with 4 players (2 per team)."""
    return FrameRecord(
        timestamp=10.0,
        period=1,
        ball_position=np.array([52.5, 34.0]),
        player_ids=["h1", "h2", "a1", "a2"],
        team_ids=["home", "home", "away", "away"],
        positions=np.array([
            [20.0, 34.0],
            [40.0, 50.0],
            [70.0, 34.0],
            [85.0, 20.0],
        ]),
        velocities=None,
        is_goalkeeper=np.array([False, False, False, False]),
    )


def _event_with_ff(event_type: str = "pass") -> EventRecord:
    return EventRecord(
        timestamp=10.0,
        period=1,
        event_type=event_type,
        player_id="h1",
        team_id="home",
        coordinates=np.array([30.0, 34.0]),
        freeze_frame=_frame_record(),
    )


def _event_without_ff() -> EventRecord:
    return EventRecord(
        timestamp=5.0,
        period=1,
        event_type="pass",
        player_id="h1",
        team_id="home",
        coordinates=np.array([30.0, 34.0]),
    )


class TestEventControl:
    def test_no_freeze_frame_raises(self, pitch: PitchSpec):
        ev = _event_without_ff()
        with pytest.raises(ValueError, match="no freeze_frame"):
            event_control(ev, pitch=pitch)

    def test_returns_probability_grid(self, pitch: PitchSpec):
        ev = _event_with_ff()
        grid = event_control(ev, pitch=pitch)

        assert isinstance(grid, ProbabilityGrid)
        assert grid.values.shape[0] > 0
        assert grid.values.shape[1] > 0

    def test_uses_event_team_id(self, pitch: PitchSpec):
        ev = _event_with_ff()
        # team_id inferred from event.team_id ("home")
        grid = event_control(ev, pitch=pitch)
        assert isinstance(grid, ProbabilityGrid)

    def test_explicit_team_id(self, pitch: PitchSpec):
        ev = _event_with_ff()
        grid = event_control(ev, team_id="away", pitch=pitch)
        assert isinstance(grid, ProbabilityGrid)

    def test_custom_control_model(self, pitch: PitchSpec):
        class MockModel:
            def control(self, frame):
                return ProbabilityGrid(
                    values=np.ones((5, 5)),
                    x_edges=np.linspace(0, 105, 6),
                    y_edges=np.linspace(0, 68, 6),
                    pitch=pitch,
                    timestamp=frame.timestamp,
                )

        ev = _event_with_ff()
        grid = event_control(ev, control_model=MockModel())
        assert grid.values.shape == (5, 5)
        np.testing.assert_allclose(grid.values, 1.0)

    def test_pitch_required_without_model(self):
        ev = _event_with_ff()
        with pytest.raises(ValueError, match="pitch is required"):
            event_control(ev)

    def test_team_id_required_without_event_team(self, pitch: PitchSpec):
        ev = EventRecord(
            timestamp=10.0,
            period=1,
            event_type="pass",
            player_id="h1",
            team_id=None,
            freeze_frame=_frame_record(),
        )
        with pytest.raises(ValueError, match="team_id is required"):
            event_control(ev, pitch=pitch)

    def test_control_values_in_range(self, pitch: PitchSpec):
        ev = _event_with_ff()
        grid = event_control(ev, pitch=pitch)
        assert grid.values.min() >= 0.0
        assert grid.values.max() <= 1.0


class TestBatchEventControl:
    def test_filters_events_without_ff(self, pitch: PitchSpec):
        events = [_event_with_ff(), _event_without_ff(), _event_with_ff()]
        results = batch_event_control(events, pitch=pitch)
        assert len(results) == 2

    def test_event_type_filter(self, pitch: PitchSpec):
        events = [
            _event_with_ff(event_type="pass"),
            _event_with_ff(event_type="shot"),
        ]
        results = batch_event_control(events, pitch=pitch, event_types=("pass",))
        assert len(results) == 1
        assert results[0][0].event_type == "pass"

    def test_empty_events(self, pitch: PitchSpec):
        results = batch_event_control([], pitch=pitch)
        assert results == []

    def test_returns_tuples(self, pitch: PitchSpec):
        events = [_event_with_ff()]
        results = batch_event_control(events, pitch=pitch)
        assert len(results) == 1
        ev, grid = results[0]
        assert isinstance(ev, EventRecord)
        assert isinstance(grid, ProbabilityGrid)
