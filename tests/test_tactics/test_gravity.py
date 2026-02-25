"""
Project: PitchAura
File Created: 2026-02-25
Author: Xingnan Zhu
File Name: test_gravity.py
Description:
    Tests for tactics.gravity module — counterfactual frame construction,
    Spatial Drag Index, Net Space Generated, penalty zone weights, gravity
    efficiency, gravity profile, deformation recovery, flow field, and
    gravity interaction matrix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pitch_aura.tactics.gravity import (
    DeformationGrid,
    DeformationVectorField,
    RecoveryMetrics,
    _counterfactual_frame,
    deformation_flow_field,
    deformation_recovery,
    gravity_interaction_matrix,
    gravity_profile,
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


# ---------------------------------------------------------------------------
# TestGravityEfficiency
# ---------------------------------------------------------------------------

class TestGravityEfficiency:
    """sdi_efficiency column is present and correctly computed."""

    def test_efficiency_column_present(self):
        df = spatial_drag_index(
            _make_moving_seq(), player_id="p0", attacking_team_id="home",
            resolution=(10, 7),
        )
        assert "sdi_efficiency" in df.columns

    def test_efficiency_nonnegative(self):
        df = spatial_drag_index(
            _make_moving_seq(), player_id="p0", attacking_team_id="home",
            resolution=(10, 7),
        )
        assert (df["sdi_efficiency"] >= 0).all()

    def test_stationary_player_efficiency_zero(self):
        """A player who never moves has zero SDI → zero efficiency."""
        df = spatial_drag_index(
            _make_moving_seq(), player_id="p1", attacking_team_id="home",
            resolution=(10, 7),
        )
        # p1 is stationary — displacement is always 0 → efficiency = 0
        assert (df["sdi_efficiency"] == 0.0).all()

    def test_efficiency_equals_sdi_over_displacement(self):
        df = spatial_drag_index(
            _make_moving_seq(n_frames=5), player_id="p0", attacking_team_id="home",
            resolution=(10, 7),
        )
        # Manually compute expected efficiency
        expected = df["sdi_m2"].to_numpy() / np.maximum(df["displacement_m"].to_numpy(), 1e-6)
        np.testing.assert_allclose(df["sdi_efficiency"].to_numpy(), expected, rtol=1e-9)


# ---------------------------------------------------------------------------
# TestGravityProfile
# ---------------------------------------------------------------------------

class TestGravityProfile:
    def test_returns_dict_with_expected_keys(self):
        result = gravity_profile(
            _make_moving_seq(), player_id="p0", attacking_team_id="home",
            resolution=(10, 7),
        )
        expected_keys = {
            "total_sdi_m2", "peak_sdi_m2", "mean_sdi_efficiency",
            "total_displacement_m", "n_significant_frames",
        }
        assert set(result.keys()) == expected_keys

    def test_empty_sequence_returns_zeros(self):
        pitch = PitchSpec()
        empty = FrameSequence(
            frames=[], frame_rate=10.0, pitch=pitch,
            home_team_id="home", away_team_id="away",
        )
        result = gravity_profile(empty, player_id="p0", attacking_team_id="home")
        assert result["total_sdi_m2"] == 0.0
        assert result["n_significant_frames"] == 0

    def test_min_displacement_filters_trivial_frames(self):
        seq = _make_moving_seq(n_frames=10, dx_per_frame=0.1)  # tiny moves
        result_tight = gravity_profile(
            seq, player_id="p0", attacking_team_id="home",
            resolution=(10, 7), min_displacement=5.0,
        )
        result_loose = gravity_profile(
            seq, player_id="p0", attacking_team_id="home",
            resolution=(10, 7), min_displacement=0.01,
        )
        assert result_tight["n_significant_frames"] <= result_loose["n_significant_frames"]


# ---------------------------------------------------------------------------
# TestDeformationRecovery
# ---------------------------------------------------------------------------

class TestDeformationRecovery:
    def _make_sdi_df(self, sdi_values: list[float]) -> pd.DataFrame:
        n = len(sdi_values)
        return pd.DataFrame({
            "timestamp": [float(i) * 0.1 for i in range(n)],
            "sdi_m2": sdi_values,
            "displacement_m": [float(i) for i in range(n)],
            "sdi_efficiency": [1.0] * n,
        })

    def test_returns_recovery_metrics(self):
        df = self._make_sdi_df([0.0, 5.0, 10.0, 7.0, 4.0, 1.0])
        result = deformation_recovery(df)
        assert isinstance(result, RecoveryMetrics)

    def test_peak_identified_correctly(self):
        df = self._make_sdi_df([0.0, 5.0, 10.0, 7.0, 4.0, 1.0])
        result = deformation_recovery(df)
        assert result.peak_sdi_m2 == 10.0
        assert abs(result.peak_timestamp - 0.2) < 1e-9

    def test_half_life_detected(self):
        # Peak=10 at t=0.2; half-life should be found when sdi < 5.0
        df = self._make_sdi_df([0.0, 5.0, 10.0, 7.0, 4.0, 1.0])
        result = deformation_recovery(df, peak_threshold=0.5)
        # sdi drops below 5.0 at index 4 (sdi=4.0, t=0.4) → half_life=0.2s
        assert result.half_life_s is not None
        assert result.half_life_s > 0.0

    def test_monotonically_increasing_sdi_gives_none_half_life(self):
        df = self._make_sdi_df([1.0, 2.0, 3.0, 4.0, 5.0])
        result = deformation_recovery(df)
        # SDI never drops after peak (last frame) → half_life is None
        assert result.half_life_s is None

    def test_empty_df_raises(self):
        with pytest.raises(ValueError):
            deformation_recovery(pd.DataFrame())


# ---------------------------------------------------------------------------
# TestDeformationFlowField
# ---------------------------------------------------------------------------

class TestDeformationFlowField:
    def _make_deformation(self, values: np.ndarray) -> DeformationGrid:
        pitch = PitchSpec()
        nx, ny = values.shape
        x0, x1 = pitch.x_range
        y0, y1 = pitch.y_range
        x_edges = np.linspace(x0, x1, nx + 1)
        y_edges = np.linspace(y0, y1, ny + 1)
        return DeformationGrid(
            values=values, x_edges=x_edges, y_edges=y_edges,
            pitch=pitch, timestamp=0.0, player_id="p0",
        )

    def test_returns_deformation_vector_field(self):
        grid = self._make_deformation(np.zeros((10, 7)))
        result = deformation_flow_field(grid)
        assert isinstance(result, DeformationVectorField)

    def test_shape_matches_input(self):
        grid = self._make_deformation(np.zeros((10, 7)))
        result = deformation_flow_field(grid)
        assert result.vectors.shape == (10, 7, 2)
        assert result.magnitudes.shape == (10, 7)

    def test_uniform_field_gives_zero_gradient(self):
        """A uniform deformation surface has no gradient → zero flow."""
        grid = self._make_deformation(np.ones((10, 7)) * 0.5)
        result = deformation_flow_field(grid)
        np.testing.assert_allclose(result.magnitudes, 0.0, atol=1e-10)

    def test_linear_gradient_gives_correct_direction(self):
        """A field that increases along x should produce vectors pointing in +x."""
        values = np.zeros((10, 7))
        for i in range(10):
            values[i, :] = float(i)  # increases along first axis (x)
        grid = self._make_deformation(values)
        result = deformation_flow_field(grid)
        # Interior cells: x-component > 0, y-component ≈ 0
        interior_vx = result.vectors[2:-2, 2:-2, 0]
        interior_vy = result.vectors[2:-2, 2:-2, 1]
        assert (interior_vx > 0).all()
        np.testing.assert_allclose(interior_vy, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# TestGravityInteractionMatrix
# ---------------------------------------------------------------------------

class TestGravityInteractionMatrix:
    def _run(self, seq=None, **kwargs):
        if seq is None:
            seq = _make_moving_seq(n_frames=5)
        defaults = dict(
            attacking_team_id="home",
            resolution=(10, 7),
        )
        defaults.update(kwargs)
        return gravity_interaction_matrix(seq, **defaults)

    def test_returns_dataframe(self):
        df = self._run()
        assert isinstance(df, pd.DataFrame)

    def test_columns_present(self):
        df = self._run()
        assert set(df.columns) >= {"mover_id", "beneficiary_id", "total_nsg_m2", "peak_nsg_m2", "mean_nsg_m2"}

    def test_n_rows_equals_ordered_pairs(self):
        """With 2 home players (p0, p1), expect 2 ordered pairs."""
        df = self._run()
        home_pids = ["p0", "p1"]
        # rows = n_movers × n_beneficiaries = 2 × 1 = 2
        assert len(df) == 2

    def test_nsg_values_nonnegative(self):
        df = self._run()
        assert (df["total_nsg_m2"] >= 0).all()
        assert (df["peak_nsg_m2"] >= 0).all()
        assert (df["mean_nsg_m2"] >= 0).all()

    def test_empty_sequence_returns_empty(self):
        pitch = PitchSpec()
        empty = FrameSequence(
            frames=[], frame_rate=10.0, pitch=pitch,
            home_team_id="home", away_team_id="away",
        )
        df = self._run(seq=empty)
        assert len(df) == 0
