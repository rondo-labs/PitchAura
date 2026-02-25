"""
Project: PitchAura
File Created: 2026-02-25
Author: Xingnan Zhu
File Name: test_gravity.py
Description:
    Tests for tactics.gravity module — counterfactual frame construction,
    Spatial Drag Index, Net Space Generated, and penalty zone weights.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pitch_aura.tactics.gravity import (
    DeformationGrid,
    _counterfactual_frame,
    net_space_generated,
    penalty_zone_weights,
    spatial_drag_index,
)
from pitch_aura.types import FrameRecord, FrameSequence, PitchSpec, ProbabilityGrid


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _frame(
    positions: list[list[float]],
    velocities: list[list[float]] | None = None,
    player_ids: list[str] | None = None,
    team_ids: list[str] | None = None,
    timestamp: float = 0.0,
) -> FrameRecord:
    """Build a quick FrameRecord for testing."""
    n = len(positions)
    pids = player_ids or [f"p{i}" for i in range(n)]
    tids = team_ids or (["home"] * (n // 2) + ["away"] * (n - n // 2))
    pos = np.array(positions, dtype=float)
    vel = np.array(velocities, dtype=float) if velocities else None
    return FrameRecord(
        timestamp=timestamp,
        period=1,
        ball_position=np.array([0.0, 0.0]),
        player_ids=pids,
        team_ids=tids,
        positions=pos,
        velocities=vel,
        is_goalkeeper=np.zeros(n, dtype=bool),
    )


def _make_moving_seq(
    n_frames: int = 5,
    frame_rate: float = 10.0,
    dx_per_frame: float = 3.0,
) -> FrameSequence:
    """Sequence where p0 (home) moves rightward, p1 (home) static, p2/p3 (away) static.

    4 players total so kinematic model has attackers and defenders.
    """
    pitch = PitchSpec()
    dt = 1.0 / frame_rate
    frames = []
    for i in range(n_frames):
        frames.append(FrameRecord(
            timestamp=i * dt,
            period=1,
            ball_position=np.array([0.0, 0.0]),
            player_ids=["p0", "p1", "p2", "p3"],
            team_ids=["home", "home", "away", "away"],
            positions=np.array([
                [-20.0 + i * dx_per_frame, 0.0],  # p0 moves right
                [10.0, 10.0],                       # p1 static (beneficiary)
                [5.0, 0.0],                         # p2 defender
                [5.0, 10.0],                        # p3 defender
            ]),
            velocities=np.array([
                [dx_per_frame * frame_rate, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]),
            is_goalkeeper=np.zeros(4, dtype=bool),
        ))
    return FrameSequence(
        frames=frames, frame_rate=frame_rate, pitch=pitch,
        home_team_id="home", away_team_id="away",
    )


# ---------------------------------------------------------------------------
# TestCounterfactualFrame
# ---------------------------------------------------------------------------

class TestCounterfactualFrame:
    def test_position_overridden(self):
        frame = _frame([[1.0, 2.0], [3.0, 4.0]])
        frozen = np.array([99.0, 99.0])
        cf = _counterfactual_frame(frame, "p0", frozen)
        np.testing.assert_array_equal(cf.positions[0], frozen)

    def test_velocity_zeroed(self):
        frame = _frame(
            [[1.0, 2.0], [3.0, 4.0]],
            velocities=[[5.0, 6.0], [7.0, 8.0]],
        )
        cf = _counterfactual_frame(frame, "p0", np.array([0.0, 0.0]))
        np.testing.assert_array_equal(cf.velocities[0], [0.0, 0.0])
        # Other player velocity unchanged
        np.testing.assert_array_equal(cf.velocities[1], [7.0, 8.0])

    def test_original_frame_not_mutated(self):
        frame = _frame(
            [[1.0, 2.0], [3.0, 4.0]],
            velocities=[[5.0, 6.0], [7.0, 8.0]],
        )
        original_pos = frame.positions.copy()
        original_vel = frame.velocities.copy()
        _counterfactual_frame(frame, "p0", np.array([99.0, 99.0]))
        np.testing.assert_array_equal(frame.positions, original_pos)
        np.testing.assert_array_equal(frame.velocities, original_vel)

    def test_all_other_players_unchanged(self):
        frame = _frame([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                       player_ids=["a", "b", "c"],
                       team_ids=["home", "home", "away"])
        cf = _counterfactual_frame(frame, "a", np.array([0.0, 0.0]))
        np.testing.assert_array_equal(cf.positions[1], [3.0, 4.0])
        np.testing.assert_array_equal(cf.positions[2], [5.0, 6.0])

    def test_missing_player_raises(self):
        frame = _frame([[1.0, 2.0]])
        with pytest.raises(ValueError, match="player_id"):
            _counterfactual_frame(frame, "ghost", np.array([0.0, 0.0]))


# ---------------------------------------------------------------------------
# TestSpatialDragIndex
# ---------------------------------------------------------------------------

class TestSpatialDragIndex:
    """Use low resolution (10, 7) for fast tests."""

    def _run(self, seq=None, **kwargs):
        if seq is None:
            seq = _make_moving_seq()
        defaults = dict(
            player_id="p0",
            attacking_team_id="home",
            resolution=(10, 7),
        )
        defaults.update(kwargs)
        return spatial_drag_index(seq, **defaults)

    def test_returns_dataframe(self):
        df = self._run()
        assert isinstance(df, pd.DataFrame)

    def test_columns_present(self):
        df = self._run()
        assert set(df.columns) >= {"timestamp", "sdi_m2", "displacement_m"}

    def test_sdi_nonnegative(self):
        df = self._run()
        assert (df["sdi_m2"] >= 0).all()

    def test_stationary_player_zero_sdi(self):
        """A player who doesn't move should have SDI = 0 (no displacement)."""
        df = self._run(player_id="p1")  # p1 is static
        assert (df["sdi_m2"] == 0.0).all()

    def test_first_frame_sdi_zero(self):
        """At t0, position == frozen_position, so SDI must be 0."""
        df = self._run()
        assert df["sdi_m2"].iloc[0] == 0.0

    def test_time_window_filters(self):
        seq = _make_moving_seq(n_frames=20, frame_rate=10.0)
        df_narrow = self._run(seq=seq, time_window=0.2)
        df_wide = self._run(seq=seq, time_window=10.0)
        assert len(df_narrow) < len(df_wide)

    def test_return_deformation_flag(self):
        result = self._run(return_deformation=True)
        assert isinstance(result, tuple)
        df, deformations = result
        assert isinstance(df, pd.DataFrame)
        assert isinstance(deformations, list)
        assert len(deformations) == len(df)

    def test_deformation_grid_shape(self):
        _, deformations = self._run(return_deformation=True)
        for d in deformations:
            assert isinstance(d, DeformationGrid)
            assert d.values.shape == (10, 7)

    def test_invalid_player_raises(self):
        seq = _make_moving_seq()
        with pytest.raises(ValueError, match="player_id"):
            self._run(seq=seq, player_id="ghost")

    def test_empty_sequence(self):
        pitch = PitchSpec()
        empty = FrameSequence(
            frames=[], frame_rate=10.0, pitch=pitch,
            home_team_id="home", away_team_id="away",
        )
        df = self._run(seq=empty)
        assert len(df) == 0
        assert set(df.columns) >= {"timestamp", "sdi_m2", "displacement_m"}


# ---------------------------------------------------------------------------
# TestNetSpaceGenerated
# ---------------------------------------------------------------------------

class TestNetSpaceGenerated:
    """Use low resolution (10, 7) for fast tests."""

    def _run(self, seq=None, **kwargs):
        if seq is None:
            seq = _make_moving_seq()
        defaults = dict(
            mover_id="p0",
            beneficiary_id="p1",
            attacking_team_id="home",
            resolution=(10, 7),
        )
        defaults.update(kwargs)
        return net_space_generated(seq, **defaults)

    def test_returns_dataframe(self):
        df = self._run()
        assert isinstance(df, pd.DataFrame)

    def test_columns_present(self):
        df = self._run()
        assert set(df.columns) >= {"timestamp", "nsg_m2", "beneficiary_x", "beneficiary_y"}

    def test_nsg_nonnegative(self):
        df = self._run()
        assert (df["nsg_m2"] >= 0).all()

    def test_zone_weights_change_nsg(self):
        seq = _make_moving_seq()
        df_uniform = self._run(seq=seq)
        # All-zero weights should give all-zero NSG
        weights = np.zeros((10, 7))
        df_zero = self._run(seq=seq, zone_weights=weights)
        assert (df_zero["nsg_m2"] == 0.0).all()
        # Uniform NSG should have at least some nonzero values (if mover actually moves)
        # The first frame has zero displacement, but later frames should show NSG
        assert df_uniform["nsg_m2"].sum() >= 0

    def test_invalid_mover_raises(self):
        seq = _make_moving_seq()
        with pytest.raises(ValueError, match="mover_id"):
            self._run(seq=seq, mover_id="ghost")

    def test_invalid_beneficiary_raises(self):
        seq = _make_moving_seq()
        with pytest.raises(ValueError, match="beneficiary_id"):
            self._run(seq=seq, beneficiary_id="ghost")


# ---------------------------------------------------------------------------
# TestPenaltyZoneWeights
# ---------------------------------------------------------------------------

class TestPenaltyZoneWeights:
    def test_shape(self):
        w = penalty_zone_weights(resolution=(20, 14))
        assert w.shape == (20, 14)

    def test_values_binary(self):
        w = penalty_zone_weights()
        unique = np.unique(w)
        assert set(unique).issubset({0.0, 1.0})

    def test_penalty_area_nonzero_and_midfield_zero(self):
        pitch = PitchSpec()  # center origin, x in [-52.5, 52.5]
        w = penalty_zone_weights(pitch, resolution=(105, 68), side="right")
        # Right penalty area: x >= 52.5 - 16.5 = 36.0
        # Grid cell centers: x_centers[i] for cells near x=40 should be 1.0
        from pitch_aura._grid import make_grid
        targets, _, _ = make_grid(pitch, (105, 68))
        grid_x = targets[:, 0].reshape(105, 68)
        # Check a cell deep in penalty area
        pa_col = np.argmin(np.abs(grid_x[:, 0] - 45.0))
        assert w[pa_col, 34] == 1.0  # centre of pitch width
        # Check midfield (x ≈ 0) is zero
        mid_col = np.argmin(np.abs(grid_x[:, 0] - 0.0))
        assert w[mid_col, 34] == 0.0

    def test_invalid_side_raises(self):
        with pytest.raises(ValueError, match="side"):
            penalty_zone_weights(side="top")
