"""
Project: PitchAura
File Created: 2026-02-23
Author: Xingnan Zhu
File Name: test_vision_cone.py
Description:
    Tests for cognitive.vision_cone: player_heading() and vision_cone_mask().
    Verifies heading inference, angular mask geometry, penalty application,
    and boundary conditions.
"""

from __future__ import annotations

import numpy as np
import pytest

from pitch_aura.cognitive.vision_cone import player_heading, vision_cone_mask


# ---------------------------------------------------------------------------
# player_heading
# ---------------------------------------------------------------------------

class TestPlayerHeading:
    def test_moving_player_unit_length(self):
        v = np.array([3.0, 4.0])
        h = player_heading(v)
        assert abs(np.linalg.norm(h) - 1.0) < 1e-9

    def test_moving_player_correct_direction(self):
        v = np.array([1.0, 0.0])
        h = player_heading(v)
        np.testing.assert_allclose(h, [1.0, 0.0])

    def test_stationary_uses_fallback(self):
        v = np.array([0.0, 0.0])
        fallback = np.array([0.0, 5.0])
        h = player_heading(v, fallback_direction=fallback)
        np.testing.assert_allclose(h, [0.0, 1.0], atol=1e-9)

    def test_stationary_no_fallback_returns_default(self):
        v = np.array([0.0, 0.0])
        h = player_heading(v)
        assert np.linalg.norm(h) == pytest.approx(1.0)

    def test_speed_threshold(self):
        # Speed exactly 0.09 → treated as stationary
        v = np.array([0.09, 0.0])
        h = player_heading(v, fallback_direction=np.array([0.0, 1.0]))
        np.testing.assert_allclose(h, [0.0, 1.0], atol=1e-9)

    def test_speed_above_threshold(self):
        v = np.array([0.11, 0.0])
        h = player_heading(v)
        np.testing.assert_allclose(h, [1.0, 0.0], atol=1e-9)


# ---------------------------------------------------------------------------
# vision_cone_mask
# ---------------------------------------------------------------------------

class TestVisionConeMask:
    def _targets_ahead_behind(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return position, heading, and two-point targets (ahead and behind)."""
        pos = np.array([0.0, 0.0])
        heading = np.array([1.0, 0.0])
        targets = np.array([
            [5.0, 0.0],   # directly ahead
            [-5.0, 0.0],  # directly behind
        ])
        return pos, heading, targets

    def test_output_shape(self):
        pos = np.array([0.0, 0.0])
        heading = np.array([1.0, 0.0])
        targets = np.random.default_rng(0).random((20, 2))
        mask = vision_cone_mask(pos, heading, targets)
        assert mask.shape == (20,)

    def test_ahead_higher_weight_than_behind(self):
        pos, heading, targets = self._targets_ahead_behind()
        mask = vision_cone_mask(pos, heading, targets, cone_half_angle=90.0)
        assert mask[0] > mask[1]

    def test_mask_within_bounds(self):
        pos = np.array([0.0, 0.0])
        heading = np.array([1.0, 0.0])
        targets = np.linspace(-10, 10, 50).reshape(25, 2)
        mask = vision_cone_mask(pos, heading, targets, peripheral_penalty=0.3)
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0

    def test_zero_peripheral_penalty(self):
        """With penalty=0 outside cone is fully dark; inside = 1."""
        pos = np.array([0.0, 0.0])
        heading = np.array([1.0, 0.0])
        targets = np.array([[5.0, 0.0], [-5.0, 0.0]])
        mask = vision_cone_mask(pos, heading, targets,
                                cone_half_angle=45.0, peripheral_penalty=0.0,
                                transition_sharpness=50.0)
        assert mask[0] > 0.9   # ahead → high
        assert mask[1] < 0.1   # behind → near zero

    def test_full_peripheral_penalty_is_identity(self):
        """With penalty=1.0 the mask is all-ones (no effect)."""
        pos = np.array([0.0, 0.0])
        heading = np.array([1.0, 0.0])
        targets = np.random.default_rng(1).random((15, 2))
        mask = vision_cone_mask(pos, heading, targets, peripheral_penalty=1.0)
        np.testing.assert_allclose(mask, 1.0)

    def test_player_position_cell_is_fully_visible(self):
        """A target at the player's exact position should get weight 1.0."""
        pos = np.array([5.0, 5.0])
        heading = np.array([0.0, -1.0])  # facing away from itself
        targets = np.array([[5.0, 5.0], [100.0, 100.0]])
        mask = vision_cone_mask(pos, heading, targets,
                                cone_half_angle=10.0, peripheral_penalty=0.0)
        assert mask[0] == pytest.approx(1.0)

    def test_symmetry_left_right(self):
        """Targets equidistant left and right should have equal weight."""
        pos = np.array([0.0, 0.0])
        heading = np.array([1.0, 0.0])
        targets = np.array([[5.0, 3.0], [5.0, -3.0]])
        mask = vision_cone_mask(pos, heading, targets, cone_half_angle=60.0)
        assert abs(mask[0] - mask[1]) < 1e-9
