"""
Project: PitchAura
File Created: 2026-02-23
Author: Xingnan Zhu
File Name: test_grid.py
Description:
    Tests for grid generation utilities.
"""

import numpy as np

from pitch_aura._grid import make_grid
from pitch_aura.types import PitchSpec


class TestMakeGrid:
    def test_shapes(self):
        pitch = PitchSpec(length=105.0, width=68.0, origin="center")
        targets, x_edges, y_edges = make_grid(pitch, resolution=(105, 68))

        assert targets.shape == (105 * 68, 2)
        assert x_edges.shape == (106,)
        assert y_edges.shape == (69,)

    def test_center_origin_bounds(self):
        pitch = PitchSpec(length=100.0, width=50.0, origin="center")
        targets, x_edges, y_edges = make_grid(pitch, resolution=(10, 5))

        assert x_edges[0] == -50.0
        assert x_edges[-1] == 50.0
        assert y_edges[0] == -25.0
        assert y_edges[-1] == 25.0

    def test_bottom_left_origin(self):
        pitch = PitchSpec(length=105.0, width=68.0, origin="bottom_left")
        targets, x_edges, y_edges = make_grid(pitch, resolution=(10, 10))

        assert x_edges[0] == 0.0
        assert x_edges[-1] == 105.0
        assert y_edges[0] == 0.0
        assert y_edges[-1] == 68.0

    def test_target_centers(self):
        pitch = PitchSpec(length=10.0, width=10.0, origin="bottom_left")
        targets, x_edges, y_edges = make_grid(pitch, resolution=(2, 2))

        # 2x2 grid: cells are [0, 5] and [5, 10] along each axis
        # Centers should be at 2.5 and 7.5
        expected = np.array([[2.5, 2.5], [2.5, 7.5], [7.5, 2.5], [7.5, 7.5]])
        np.testing.assert_allclose(targets, expected)
