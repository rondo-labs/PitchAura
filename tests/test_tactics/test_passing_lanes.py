"""
Project: PitchAura
File Created: 2026-02-23
Author: Xingnan Zhu
File Name: test_passing_lanes.py
Description:
    Tests for tactics.passing_lane_lifespan() and the internal
    _lane_obstructed() helper. Covers clear lanes, blocked lanes,
    partial obstruction, edge cases (missing players, zero-length window).
"""

from __future__ import annotations

import numpy as np
import pytest

from pitch_aura.tactics.passing_lanes import _lane_obstructed, passing_lane_lifespan
from pitch_aura.types import FrameRecord, FrameSequence, PitchSpec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _frame(passer_pos, receiver_pos, defenders, *, frame_id: int = 0) -> FrameRecord:
    """Build a FrameRecord with one passer, one receiver, and N defenders."""
    player_ids = ["passer", "receiver"] + [f"def{i}" for i in range(len(defenders))]
    team_ids = ["home", "home"] + ["away"] * len(defenders)
    positions = np.array([passer_pos, receiver_pos] + defenders, dtype=float)
    return FrameRecord(
        timestamp=frame_id * 0.04,
        period=1,
        ball_position=np.array([0.0, 0.0]),
        player_ids=player_ids,
        team_ids=team_ids,
        positions=positions,
        velocities=np.zeros((len(player_ids), 2)),
        is_goalkeeper=np.zeros(len(player_ids), dtype=bool),
    )


def _seq(frames: list[FrameRecord]) -> FrameSequence:
    return FrameSequence(
        frames=frames, frame_rate=25.0, pitch=PitchSpec(),
        home_team_id="home", away_team_id="away",
    )


# ---------------------------------------------------------------------------
# _lane_obstructed
# ---------------------------------------------------------------------------

class TestLaneObstructed:
    def test_empty_defenders_not_obstructed(self):
        frame = _frame([0.0, 0.0], [10.0, 0.0], defenders=[])
        assert not _lane_obstructed(frame, "passer", "receiver", "away", lane_width=2.0)

    def test_defender_directly_in_lane(self):
        frame = _frame([0.0, 0.0], [10.0, 0.0], defenders=[[5.0, 0.0]])
        assert _lane_obstructed(frame, "passer", "receiver", "away", lane_width=2.0)

    def test_defender_beside_lane_not_obstructed(self):
        # Defender is 3 m to the side of a 2 m wide lane
        frame = _frame([0.0, 0.0], [10.0, 0.0], defenders=[[5.0, 5.0]])
        assert not _lane_obstructed(frame, "passer", "receiver", "away", lane_width=2.0)

    def test_defender_behind_passer_not_obstructed(self):
        frame = _frame([5.0, 0.0], [15.0, 0.0], defenders=[[2.0, 0.0]])
        assert not _lane_obstructed(frame, "passer", "receiver", "away", lane_width=2.0)

    def test_defender_beyond_receiver_not_obstructed(self):
        frame = _frame([0.0, 0.0], [10.0, 0.0], defenders=[[15.0, 0.0]])
        assert not _lane_obstructed(frame, "passer", "receiver", "away", lane_width=2.0)

    def test_diagonal_lane_with_obstruction(self):
        # Diagonal pass; defender near midpoint
        frame = _frame([0.0, 0.0], [10.0, 10.0], defenders=[[5.0, 5.0]])
        assert _lane_obstructed(frame, "passer", "receiver", "away", lane_width=2.0)

    def test_missing_passer_returns_false(self):
        frame = _frame([0.0, 0.0], [10.0, 0.0], defenders=[[5.0, 0.0]])
        assert not _lane_obstructed(frame, "nobody", "receiver", "away", lane_width=2.0)


# ---------------------------------------------------------------------------
# passing_lane_lifespan
# ---------------------------------------------------------------------------

class TestPassingLaneLifespan:
    def test_clear_lane_all_frames(self):
        frames = [_frame([0.0, 0.0], [10.0, 0.0], defenders=[], frame_id=i) for i in range(25)]
        seq = _seq(frames)
        lifespan = passing_lane_lifespan(seq, passer_id="passer", receiver_id="receiver")
        assert lifespan == pytest.approx(1.0, rel=1e-6)  # 25 frames at 25 fps = 1 s

    def test_fully_blocked_lane(self):
        frames = [_frame([0.0, 0.0], [10.0, 0.0], defenders=[[5.0, 0.0]], frame_id=i) for i in range(10)]
        seq = _seq(frames)
        lifespan = passing_lane_lifespan(seq, passer_id="passer", receiver_id="receiver")
        assert lifespan == pytest.approx(0.0)

    def test_partial_lifespan(self):
        # 10 open frames, then 10 blocked
        clear = [_frame([0.0, 0.0], [10.0, 0.0], defenders=[], frame_id=i) for i in range(10)]
        blocked = [_frame([0.0, 0.0], [10.0, 0.0], defenders=[[5.0, 0.0]], frame_id=10 + i) for i in range(10)]
        seq = _seq(clear + blocked)
        lifespan = passing_lane_lifespan(seq, passer_id="passer", receiver_id="receiver")
        assert lifespan == pytest.approx(10 / 25.0, rel=1e-6)

    def test_empty_sequence(self):
        seq = _seq([])
        lifespan = passing_lane_lifespan(seq, passer_id="passer", receiver_id="receiver")
        assert lifespan == 0.0

    def test_missing_passer_raises(self):
        frames = [_frame([0.0, 0.0], [10.0, 0.0], defenders=[], frame_id=0)]
        seq = _seq(frames)
        with pytest.raises(ValueError, match="passer_id"):
            passing_lane_lifespan(seq, passer_id="ghost", receiver_id="receiver")

    def test_missing_receiver_raises(self):
        frames = [_frame([0.0, 0.0], [10.0, 0.0], defenders=[], frame_id=0)]
        seq = _seq(frames)
        with pytest.raises(ValueError, match="receiver_id"):
            passing_lane_lifespan(seq, passer_id="passer", receiver_id="ghost")
