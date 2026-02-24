"""
Project: PitchAura
File Created: 2026-02-23
Author: Xingnan Zhu
File Name: test_kinematic.py
Description:
    Tests for the Spearman kinematic pitch control model and its physics
    building blocks. Uses analytically verifiable configurations where
    possible, plus monotonicity and boundary checks.
"""

from __future__ import annotations

import numpy as np
import pytest

from pitch_aura.space._physics import (
    accumulate_control,
    sigmoid_influence,
    time_to_intercept,
)
from pitch_aura.space.kinematic import KinematicControlModel
from pitch_aura.types import FrameRecord, PitchSpec, ProbabilityGrid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _frame(
    att_positions: list[tuple[float, float]],
    def_positions: list[tuple[float, float]],
    att_team: str = "home",
    def_team: str = "away",
) -> FrameRecord:
    """Build a FrameRecord with two teams at given positions."""
    positions = att_positions + def_positions
    n_att = len(att_positions)
    n_def = len(def_positions)
    return FrameRecord(
        timestamp=0.0,
        period=1,
        ball_position=np.array([0.0, 0.0]),
        player_ids=[f"a{i}" for i in range(n_att)] + [f"d{i}" for i in range(n_def)],
        team_ids=[att_team] * n_att + [def_team] * n_def,
        positions=np.array(positions, dtype=np.float64),
        velocities=np.zeros((n_att + n_def, 2)),
        is_goalkeeper=np.zeros(n_att + n_def, dtype=bool),
    )


# ---------------------------------------------------------------------------
# Tests: time_to_intercept
# ---------------------------------------------------------------------------

class TestTimeToIntercept:
    def test_shape(self):
        pos = np.zeros((4, 2))
        targets = np.zeros((7140, 2))
        tti = time_to_intercept(pos, targets, reaction_time=0.7, v_max=5.0)
        assert tti.shape == (4, 7140)

    def test_player_on_target_equals_reaction_time(self):
        """Player already at target: TTI = reaction_time."""
        pos = np.array([[10.0, 20.0]])
        targets = np.array([[10.0, 20.0]])
        tti = time_to_intercept(pos, targets, reaction_time=0.7, v_max=5.0)
        assert tti[0, 0] == pytest.approx(0.7)

    def test_linear_in_distance(self):
        """TTI increases linearly with distance: t = rt + dist / v_max."""
        pos = np.array([[0.0, 0.0]])
        targets = np.array([[10.0, 0.0], [20.0, 0.0]])
        tti = time_to_intercept(pos, targets, reaction_time=0.0, v_max=5.0)
        assert tti[0, 0] == pytest.approx(2.0)
        assert tti[0, 1] == pytest.approx(4.0)

    def test_symmetric(self):
        """Two players equidistant from a target get identical TTI."""
        pos = np.array([[-5.0, 0.0], [5.0, 0.0]])
        targets = np.array([[0.0, 0.0]])
        tti = time_to_intercept(pos, targets, reaction_time=0.0, v_max=5.0)
        assert tti[0, 0] == pytest.approx(tti[1, 0])


# ---------------------------------------------------------------------------
# Tests: sigmoid_influence
# ---------------------------------------------------------------------------

class TestSigmoidInfluence:
    def test_shape_preserved(self):
        tti = np.ones((6, 100))
        f = sigmoid_influence(tti, t=1.0, sigma=0.45)
        assert f.shape == (6, 100)

    def test_at_tti_equals_half(self):
        """At t == TTI the sigmoid equals 0.5 regardless of sigma."""
        tti = np.array([[2.0, 3.0]])
        f = sigmoid_influence(tti, t=2.0, sigma=0.45)
        assert f[0, 0] == pytest.approx(0.5, abs=1e-10)

    def test_above_tti_above_half(self):
        """For t > TTI the player has had more than enough time → f > 0.5."""
        tti = np.array([[1.0]])
        f = sigmoid_influence(tti, t=5.0, sigma=0.45)
        assert f[0, 0] > 0.5

    def test_below_tti_below_half(self):
        """For t < TTI the player hasn't arrived yet → f < 0.5."""
        tti = np.array([[5.0]])
        f = sigmoid_influence(tti, t=1.0, sigma=0.45)
        assert f[0, 0] < 0.5

    def test_monotone_in_t(self):
        """Influence is monotonically increasing in time."""
        tti = np.array([[2.0] * 10])
        times = np.linspace(0, 10, 50)
        values = [sigmoid_influence(tti, t, sigma=0.45)[0, 0] for t in times]
        assert all(v1 <= v2 for v1, v2 in zip(values, values[1:]))

    def test_values_in_01(self):
        tti = np.random.default_rng(0).uniform(0, 10, (11, 500))
        f = sigmoid_influence(tti, t=3.0, sigma=0.45)
        assert np.all(f >= 0.0) and np.all(f <= 1.0)


# ---------------------------------------------------------------------------
# Tests: accumulate_control
# ---------------------------------------------------------------------------

class TestAccumulateControl:
    def _run(self, tti_att, tti_def, **kwargs):
        defaults = dict(
            sigma_att=0.45, sigma_def=0.45,
            lam_att=4.3, lam_def=4.3,
            dt=0.04, t_max=10.0,
            convergence_threshold=0.01,
        )
        defaults.update(kwargs)
        return accumulate_control(tti_att, tti_def, **defaults)

    def test_shapes(self):
        tti_att = np.ones((3, 50))
        tti_def = np.ones((3, 50))
        pa, pd = self._run(tti_att, tti_def)
        assert pa.shape == (50,)
        assert pd.shape == (50,)

    def test_values_in_01(self):
        tti_att = np.ones((3, 200)) * 0.5
        tti_def = np.ones((3, 200)) * 2.0
        pa, pd = self._run(tti_att, tti_def)
        assert np.all(pa >= 0.0) and np.all(pa <= 1.0)
        assert np.all(pd >= 0.0) and np.all(pd <= 1.0)

    def test_ppcf_sum_le_one(self):
        """Total claimed probability never exceeds 1 at any grid point."""
        tti_att = np.random.default_rng(1).uniform(0.5, 3.0, (5, 300))
        tti_def = np.random.default_rng(2).uniform(0.5, 3.0, (5, 300))
        pa, pd = self._run(tti_att, tti_def)
        assert np.all(pa + pd <= 1.0 + 1e-9)

    def test_closer_attacker_dominates(self):
        """Attacker right next to a target should control it with high probability."""
        G = 1
        # Attacker is 1 m away, defender is 40 m away
        tti_att = np.array([[0.7 + 1.0 / 5.0]])   # rt + 1/v_max
        tti_def = np.array([[0.7 + 40.0 / 5.0]])
        pa, pd = self._run(tti_att, tti_def)
        assert pa[0] > 0.9, f"Attacker near target should dominate; got {pa[0]:.3f}"

    def test_symmetric_gives_equal_control(self):
        """Symmetric configuration: both teams control equal fractions."""
        G = 100
        # Both teams identical distance from all targets
        tti_equal = np.ones((3, G)) * 1.5
        pa, pd = self._run(tti_equal, tti_equal)
        np.testing.assert_allclose(pa, pd, atol=0.05)


# ---------------------------------------------------------------------------
# Tests: KinematicControlModel
# ---------------------------------------------------------------------------

class TestKinematicControlModel:
    @pytest.fixture
    def small_model(self) -> KinematicControlModel:
        """Low-resolution model for fast tests."""
        return KinematicControlModel(
            resolution=(21, 14),
            pitch=PitchSpec(length=105.0, width=68.0, origin="center"),
        )

    def test_returns_probability_grid(self, small_model: KinematicControlModel):
        frame = _frame([(-20.0, 0.0)], [(20.0, 0.0)])
        result = small_model.control(frame, team_id="home")
        assert isinstance(result, ProbabilityGrid)

    def test_grid_resolution(self, small_model: KinematicControlModel):
        frame = _frame([(-20.0, 0.0)], [(20.0, 0.0)])
        result = small_model.control(frame, team_id="home")
        assert result.values.shape == (21, 14)

    def test_values_in_01(self, small_model: KinematicControlModel):
        frame = _frame([(-20.0, 5.0), (-20.0, -5.0)], [(20.0, 5.0), (20.0, -5.0)])
        result = small_model.control(frame, team_id="home")
        assert np.all(result.values >= 0.0)
        assert np.all(result.values <= 1.0)

    def test_timestamp_preserved(self, small_model: KinematicControlModel):
        frame = _frame([(-20.0, 0.0)], [(20.0, 0.0)])
        frame.timestamp = 99.5
        result = small_model.control(frame, team_id="home")
        assert result.timestamp == 99.5

    def test_attacker_dominates_near_territory(self, small_model: KinematicControlModel):
        """Cells near the attacker cluster should be controlled by attackers."""
        frame = _frame(
            att_positions=[(-40.0, 0.0), (-40.0, 10.0), (-40.0, -10.0)],
            def_positions=[(40.0, 0.0),  (40.0, 10.0),  (40.0, -10.0)],
        )
        result = small_model.control(frame, team_id="home")
        # The leftmost column (attacker side) should be mostly > 0.5
        left_col = result.values[0, :]
        assert np.mean(left_col > 0.5) >= 0.6

    def test_defender_dominates_far_territory(self, small_model: KinematicControlModel):
        """Cells near the defender cluster should be controlled by defenders."""
        frame = _frame(
            att_positions=[(-40.0, 0.0), (-40.0, 10.0), (-40.0, -10.0)],
            def_positions=[(40.0, 0.0),  (40.0, 10.0),  (40.0, -10.0)],
        )
        result = small_model.control(frame, team_id="home")
        # The rightmost column (defender side) should be mostly < 0.5
        right_col = result.values[-1, :]
        assert np.mean(right_col < 0.5) >= 0.6

    def test_symmetric_frame_gives_symmetric_grid(self, small_model: KinematicControlModel):
        """Perfect left-right mirror configuration → grid is anti-symmetric."""
        frame = _frame(
            att_positions=[(-20.0, 0.0), (-20.0, 15.0), (-20.0, -15.0)],
            def_positions=[(20.0, 0.0),  (20.0, 15.0),  (20.0, -15.0)],
        )
        result = small_model.control(frame, team_id="home")
        # Left half average > 0.5, right half average < 0.5
        nx = result.values.shape[0]
        left_mean = result.values[: nx // 2].mean()
        right_mean = result.values[nx // 2 :].mean()
        assert left_mean > 0.5
        assert right_mean < 0.5

    def test_control_batch(self, small_model: KinematicControlModel):
        frames = [_frame([(-20.0, 0.0)], [(20.0, 0.0)]) for _ in range(3)]
        results = small_model.control_batch(frames, team_id="home")
        assert len(results) == 3
        assert all(isinstance(r, ProbabilityGrid) for r in results)

    def test_default_pitch_used_when_none(self):
        """Model with pitch=None defaults to standard 105x68 pitch."""
        model = KinematicControlModel(resolution=(21, 14))
        frame = _frame([(-20.0, 0.0)], [(20.0, 0.0)])
        result = model.control(frame, team_id="home")
        assert result.pitch.length == pytest.approx(105.0)
        assert result.pitch.width == pytest.approx(68.0)

    def test_x_edges_cover_full_pitch(self, small_model: KinematicControlModel):
        frame = _frame([(-20.0, 0.0)], [(20.0, 0.0)])
        result = small_model.control(frame, team_id="home")
        assert result.x_edges[0] == pytest.approx(-52.5)
        assert result.x_edges[-1] == pytest.approx(52.5)
        assert result.y_edges[0] == pytest.approx(-34.0)
        assert result.y_edges[-1] == pytest.approx(34.0)
