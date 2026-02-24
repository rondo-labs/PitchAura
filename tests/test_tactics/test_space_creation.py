"""
Project: PitchAura
File Created: 2026-02-23
Author: Xingnan Zhu
File Name: test_space_creation.py
Description:
    Tests for tactics.space_creation().
    Verifies DataFrame structure, time-window filtering, player validation,
    and delta_area computation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pitch_aura.tactics.space_creation import space_creation
from pitch_aura.types import FrameRecord, FrameSequence, PitchSpec


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_seq(n_frames: int = 10, frame_rate: float = 25.0) -> FrameSequence:
    """Two-player sequence: p1 (home) moves linearly, p2 (away) is static."""
    pitch = PitchSpec()
    dt = 1.0 / frame_rate
    frames = []
    for i in range(n_frames):
        frames.append(FrameRecord(
            timestamp=i * dt,
            period=1,
            ball_position=np.array([0.0, 0.0]),
            player_ids=["p1", "p2"],
            team_ids=["home", "away"],
            positions=np.array([[float(i) * 2.0 - 20.0, 0.0], [10.0, 5.0]]),
            velocities=np.zeros((2, 2)),
            is_goalkeeper=np.zeros(2, dtype=bool),
        ))
    return FrameSequence(
        frames=frames, frame_rate=frame_rate, pitch=pitch,
        home_team_id="home", away_team_id="away",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSpaceCreation:
    def test_returns_dataframe(self):
        seq = _make_seq(10)
        df = space_creation(seq, player_id="p1")
        assert isinstance(df, pd.DataFrame)

    def test_columns_present(self):
        seq = _make_seq(10)
        df = space_creation(seq, player_id="p1")
        assert set(df.columns) >= {"timestamp", "area_m2", "delta_area"}

    def test_area_nonnegative(self):
        seq = _make_seq(10)
        df = space_creation(seq, player_id="p1")
        assert (df["area_m2"] >= 0).all()

    def test_first_delta_is_nan(self):
        seq = _make_seq(10)
        df = space_creation(seq, player_id="p1")
        assert np.isnan(df["delta_area"].iloc[0])

    def test_delta_matches_diff(self):
        seq = _make_seq(10)
        df = space_creation(seq, player_id="p1")
        areas = df["area_m2"].values
        expected_deltas = np.diff(areas)
        np.testing.assert_allclose(df["delta_area"].iloc[1:], expected_deltas, atol=1e-10)

    def test_time_window_filters_frames(self):
        seq = _make_seq(25, frame_rate=25.0)  # 1-second sequence
        # Window of 0.2 s → only 5 frames (25 fps * 0.2)
        df_narrow = space_creation(seq, player_id="p1", time_window=0.2)
        df_wide = space_creation(seq, player_id="p1", time_window=10.0)
        assert len(df_narrow) < len(df_wide)

    def test_empty_sequence_returns_empty_df(self):
        pitch = PitchSpec()
        empty = FrameSequence(frames=[], frame_rate=25.0, pitch=pitch,
                              home_team_id="home", away_team_id="away")
        df = space_creation(empty, player_id="p1")
        assert len(df) == 0
        assert set(df.columns) >= {"timestamp", "area_m2", "delta_area"}

    def test_invalid_player_raises(self):
        seq = _make_seq(5)
        with pytest.raises(ValueError, match="player_id"):
            space_creation(seq, player_id="ghost")

    def test_timestamps_monotonic(self):
        seq = _make_seq(15)
        df = space_creation(seq, player_id="p1")
        assert (df["timestamp"].diff().iloc[1:] >= 0).all()
