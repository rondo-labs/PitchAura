"""
Project: PitchAura
File Created: 2026-02-25
Author: Xingnan Zhu
File Name: test_gravity_vision.py
Description:
    Tests for cognitive.gravity_vision — VisionAwareControlModel.
    Verifies interface compatibility, return types, and that vision
    penalisation produces grids that differ from the base kinematic model.
"""

from __future__ import annotations

import numpy as np
import pytest

from pitch_aura.cognitive.gravity_vision import VisionAwareControlModel
from pitch_aura.space.kinematic import KinematicControlModel
from pitch_aura.tactics.gravity import spatial_drag_index
from pitch_aura.types import FrameRecord, FrameSequence, PitchSpec, ProbabilityGrid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _frame_with_two_teams() -> FrameRecord:
    """Simple 4-player frame: 2 attackers (home), 2 defenders (away)."""
    return FrameRecord(
        timestamp=0.0,
        period=1,
        ball_position=np.array([0.0, 0.0]),
        player_ids=["a0", "a1", "d0", "d1"],
        team_ids=["home", "home", "away", "away"],
        positions=np.array([
            [-10.0, 0.0],    # attacker
            [-5.0, 5.0],     # attacker
            [10.0, 0.0],     # defender facing away (velocity pointing right)
            [10.0, -5.0],    # defender
        ]),
        velocities=np.array([
            [2.0, 0.0],
            [0.0, 0.0],
            [5.0, 0.0],   # running away from attackers → blind spot behind
            [0.0, 0.0],
        ]),
        is_goalkeeper=np.zeros(4, dtype=bool),
    )


def _make_moving_seq(n_frames: int = 5, frame_rate: float = 10.0) -> FrameSequence:
    """Sequence with p0 moving rightward (home) and two static defenders (away)."""
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
                [-20.0 + i * 3.0, 0.0],
                [10.0, 10.0],
                [5.0, 0.0],
                [5.0, 10.0],
            ]),
            velocities=np.array([
                [30.0, 0.0],
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
# TestVisionAwareControlModel
# ---------------------------------------------------------------------------

class TestVisionAwareControlModel:
    def test_instantiates_with_defaults(self):
        model = VisionAwareControlModel()
        assert model is not None

    def test_control_returns_probability_grid(self):
        model = VisionAwareControlModel(resolution=(10, 7))
        frame = _frame_with_two_teams()
        result = model.control(frame, team_id="home")
        assert isinstance(result, ProbabilityGrid)

    def test_grid_values_in_range(self):
        model = VisionAwareControlModel(resolution=(10, 7))
        frame = _frame_with_two_teams()
        result = model.control(frame, team_id="home")
        assert result.values.min() >= -1e-9
        assert result.values.max() <= 1.0 + 1e-9

    def test_values_differ_from_base_kinematic(self):
        """Vision adjustment should change probabilities vs. plain kinematic."""
        pitch = PitchSpec()
        frame = _frame_with_two_teams()
        base = KinematicControlModel(pitch=pitch, resolution=(10, 7))
        vision = VisionAwareControlModel(
            pitch=pitch, resolution=(10, 7), peripheral_penalty=0.0
        )
        base_grid = base.control(frame, team_id="home")
        vision_grid = vision.control(frame, team_id="home")
        # With peripheral_penalty=0.0 (fully blind outside cone), grids must differ
        assert not np.allclose(base_grid.values, vision_grid.values)

    def test_works_as_control_model_in_spatial_drag_index(self):
        """VisionAwareControlModel is accepted by spatial_drag_index."""
        seq = _make_moving_seq()
        model = VisionAwareControlModel(pitch=seq.pitch, resolution=(10, 7))
        df = spatial_drag_index(
            seq,
            player_id="p0",
            attacking_team_id="home",
            control_model=model,
        )
        assert not df.empty
        assert "sdi_m2" in df.columns
