"""
Project: PitchAura
File Created: 2026-02-23
Author: Xingnan Zhu
File Name: test_blind_spots.py
Description:
    Tests for cognitive.VisionModel.apply().
    Verifies that the adjusted ProbabilityGrid has correct type/shape,
    probabilities remain in [0,1], blind-spot cells are boosted for the
    attacker, and edge cases (no defenders, penalty=1.0) behave correctly.
"""

from __future__ import annotations

import numpy as np
import pytest

from pitch_aura.cognitive.blind_spots import VisionModel
from pitch_aura.types import FrameRecord, PitchSpec, ProbabilityGrid


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_grid(nx: int = 10, ny: int = 10, value: float = 0.5) -> ProbabilityGrid:
    """Uniform probability grid for a small pitch."""
    pitch = PitchSpec(length=20.0, width=10.0)
    x_edges = np.linspace(-10.0, 10.0, nx + 1)
    y_edges = np.linspace(-5.0, 5.0, ny + 1)
    return ProbabilityGrid(
        values=np.full((nx, ny), value),
        x_edges=x_edges,
        y_edges=y_edges,
        pitch=pitch,
        timestamp=0.0,
    )


def _make_frame(
    attacker_pos: list[tuple],
    defender_pos: list[tuple],
    defender_vel: list[tuple] | None = None,
) -> FrameRecord:
    all_pos = list(attacker_pos) + list(defender_pos)
    n_att = len(attacker_pos)
    n_def = len(defender_pos)
    player_ids = [f"a{i}" for i in range(n_att)] + [f"d{i}" for i in range(n_def)]
    team_ids = ["home"] * n_att + ["away"] * n_def

    if defender_vel is None:
        def_v = [(0.0, 0.0)] * n_def
    else:
        def_v = defender_vel
    velocities = [(0.0, 0.0)] * n_att + list(def_v)

    return FrameRecord(
        timestamp=0.0,
        period=1,
        ball_position=np.array([0.0, 0.0]),
        player_ids=player_ids,
        team_ids=team_ids,
        positions=np.array(all_pos, dtype=float),
        velocities=np.array(velocities, dtype=float),
        is_goalkeeper=np.zeros(n_att + n_def, dtype=bool),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestVisionModelApply:
    def test_returns_probability_grid(self):
        grid = _make_grid()
        frame = _make_frame([(0.0, 0.0)], [(-5.0, 0.0)])
        model = VisionModel()
        result = model.apply(grid, frame, defending_team_id="away")
        assert isinstance(result, ProbabilityGrid)

    def test_output_shape_preserved(self):
        grid = _make_grid(10, 8)
        frame = _make_frame([(0.0, 0.0)], [(-5.0, 0.0)])
        model = VisionModel()
        result = model.apply(grid, frame, defending_team_id="away")
        assert result.values.shape == (10, 8)

    def test_probabilities_in_range(self):
        grid = _make_grid()
        frame = _make_frame([(0.0, 0.0)], [(-5.0, 0.0)])
        model = VisionModel()
        result = model.apply(grid, frame, defending_team_id="away")
        assert result.values.min() >= 0.0
        assert result.values.max() <= 1.0

    def test_no_defenders_returns_unchanged(self):
        grid = _make_grid(value=0.6)
        frame = _make_frame([(0.0, 0.0)], [])  # no defenders
        model = VisionModel()
        result = model.apply(grid, frame, defending_team_id="away")
        np.testing.assert_allclose(result.values, grid.values)

    def test_full_penalty_no_effect(self):
        """peripheral_penalty=1.0 → mask is all-ones → no adjustment."""
        grid = _make_grid(value=0.5)
        frame = _make_frame([(0.0, 0.0)], [(-5.0, 0.0)])
        model = VisionModel(peripheral_penalty=1.0)
        result = model.apply(grid, frame, defending_team_id="away")
        np.testing.assert_allclose(result.values, grid.values, atol=1e-6)

    def test_blind_spot_increases_attacker_probability(self):
        """With penalty < 1.0 at least some cells should be boosted for attacker."""
        grid = _make_grid(value=0.5)
        frame = _make_frame([(0.0, 0.0)], [(-8.0, 0.0)],
                            defender_vel=[(-5.0, 0.0)])  # defender facing away
        model = VisionModel(peripheral_penalty=0.0, cone_half_angle=60.0,
                            transition_sharpness=20.0)
        result = model.apply(grid, frame, defending_team_id="away")
        # At least some cells should be > 0.5 (attacker boosted)
        assert result.values.max() > 0.5

    def test_timestamps_and_edges_preserved(self):
        grid = _make_grid()
        frame = _make_frame([(0.0, 0.0)], [(-5.0, 0.0)])
        model = VisionModel()
        result = model.apply(grid, frame, defending_team_id="away")
        assert result.timestamp == grid.timestamp
        np.testing.assert_array_equal(result.x_edges, grid.x_edges)
        np.testing.assert_array_equal(result.y_edges, grid.y_edges)

    def test_pitch_preserved(self):
        grid = _make_grid()
        frame = _make_frame([(0.0, 0.0)], [(-5.0, 0.0)])
        model = VisionModel()
        result = model.apply(grid, frame, defending_team_id="away")
        assert result.pitch == grid.pitch
