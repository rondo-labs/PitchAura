"""
Project: PitchAura
File Created: 2026-02-23
Author: Xingnan Zhu
File Name: test_alignment.py
Description:
    Tests for sync.align(): nearest-frame and interpolation alignment.
"""

from __future__ import annotations

import numpy as np
import pytest

from pitch_aura.sync.alignment import align
from pitch_aura.types import EventRecord, FrameRecord, FrameSequence, PitchSpec


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_seq(n_frames: int = 10, frame_rate: float = 25.0) -> FrameSequence:
    """Sequence of n_frames at uniform intervals, two players moving linearly."""
    dt = 1.0 / frame_rate
    pitch = PitchSpec(length=105.0, width=68.0, origin="center")
    frames = []
    for i in range(n_frames):
        t = i * dt
        frames.append(FrameRecord(
            timestamp=t,
            period=1,
            ball_position=np.array([float(i), 0.0]),
            player_ids=["p1", "p2"],
            team_ids=["home", "away"],
            positions=np.array([[float(i), 0.0], [-float(i), 0.0]]),
            velocities=np.zeros((2, 2)),
            is_goalkeeper=np.zeros(2, dtype=bool),
        ))
    return FrameSequence(frames=frames, frame_rate=frame_rate, pitch=pitch,
                         home_team_id="home", away_team_id="away")


def _event(t: float) -> EventRecord:
    return EventRecord(timestamp=t, period=1, event_type="pass")


# ---------------------------------------------------------------------------
# Tests: align — nearest
# ---------------------------------------------------------------------------

class TestAlignNearest:
    def test_empty_events_returns_empty(self):
        seq = _make_seq(5)
        result = align(seq, [], method="nearest")
        assert len(result) == 0

    def test_empty_frames_returns_empty(self):
        pitch = PitchSpec()
        empty = FrameSequence(frames=[], frame_rate=25.0, pitch=pitch,
                              home_team_id="home", away_team_id="away")
        result = align(empty, [_event(1.0)], method="nearest")
        assert len(result) == 0

    def test_one_event_picks_closest_frame(self):
        seq = _make_seq(n_frames=10, frame_rate=25.0)
        # Frame timestamps: 0, 0.04, 0.08, 0.12, ...
        # Event at t=0.05 → closest is 0.04 (idx=1)
        result = align(seq, [_event(0.05)], method="nearest")
        assert len(result) == 1
        assert result[0].timestamp == pytest.approx(0.04)

    def test_multiple_events_correct_count(self):
        seq = _make_seq(10)
        events = [_event(t) for t in [0.0, 0.12, 0.20]]
        result = align(seq, events, method="nearest")
        assert len(result) == 3

    def test_event_before_range_clamps_to_first(self):
        seq = _make_seq(5)
        result = align(seq, [_event(-999.0)], method="nearest")
        assert result[0].timestamp == pytest.approx(seq[0].timestamp)

    def test_event_after_range_clamps_to_last(self):
        seq = _make_seq(5)
        result = align(seq, [_event(999.0)], method="nearest")
        assert result[0].timestamp == pytest.approx(seq[-1].timestamp)

    def test_metadata_preserved(self):
        seq = _make_seq(5)
        result = align(seq, [_event(0.0)])
        assert result.frame_rate == seq.frame_rate
        assert result.home_team_id == seq.home_team_id
        assert result.pitch.length == seq.pitch.length

    def test_invalid_method_raises(self):
        seq = _make_seq(3)
        with pytest.raises(ValueError, match="method"):
            align(seq, [_event(0.0)], method="bad_method")


# ---------------------------------------------------------------------------
# Tests: align — interpolate
# ---------------------------------------------------------------------------

class TestAlignInterpolate:
    def test_exact_frame_timestamp_unchanged(self):
        """Event exactly at a frame boundary → same position."""
        seq = _make_seq(5)
        t_exact = seq[2].timestamp
        result = align(seq, [_event(t_exact)], method="interpolate")
        np.testing.assert_allclose(result[0].positions, seq[2].positions, atol=1e-10)

    def test_midpoint_position_is_average(self):
        """Event halfway between two frames → position is mean of both frames."""
        seq = _make_seq(4)
        t_a = seq[1].timestamp
        t_b = seq[2].timestamp
        t_mid = (t_a + t_b) / 2.0
        result = align(seq, [_event(t_mid)], method="interpolate")
        expected = (seq[1].positions + seq[2].positions) / 2.0
        np.testing.assert_allclose(result[0].positions, expected, atol=1e-10)

    def test_interpolated_timestamp_correct(self):
        seq = _make_seq(5)
        t = seq[1].timestamp + 0.012
        result = align(seq, [_event(t)], method="interpolate")
        assert result[0].timestamp == pytest.approx(t)

    def test_before_range_clamps(self):
        seq = _make_seq(5)
        result = align(seq, [_event(-1.0)], method="interpolate")
        assert result[0].timestamp == pytest.approx(seq[0].timestamp)

    def test_after_range_clamps(self):
        seq = _make_seq(5)
        result = align(seq, [_event(999.0)], method="interpolate")
        assert result[0].timestamp == pytest.approx(seq[-1].timestamp)

    def test_ball_interpolated(self):
        seq = _make_seq(4)
        t_mid = (seq[0].timestamp + seq[1].timestamp) / 2.0
        result = align(seq, [_event(t_mid)], method="interpolate")
        expected_ball = (seq[0].ball_position + seq[1].ball_position) / 2.0
        np.testing.assert_allclose(result[0].ball_position, expected_ball, atol=1e-10)
