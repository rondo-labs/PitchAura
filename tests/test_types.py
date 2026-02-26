"""
Project: PitchAura
File Created: 2026-02-23
Author: Xingnan Zhu
File Name: test_types.py
Description:
    Tests for PitchAura-specific types (VoronoiResult).
    Shared types (PitchSpec, FrameRecord, etc.) are tested in PitchCore.
"""

import numpy as np

from pitch_aura.types import VoronoiResult


class TestVoronoiResult:
    def test_construction(self):
        result = VoronoiResult(
            regions={"p1": np.array([[0, 0], [1, 0], [1, 1]])},
            areas={"p1": 0.5},
            team_areas={"t1": 0.5},
            timestamp=0.0,
        )
        assert result.areas["p1"] == 0.5
