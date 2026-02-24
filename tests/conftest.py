"""
Project: PitchAura
File Created: 2026-02-23
Author: Xingnan Zhu
File Name: conftest.py
Description:
    Shared test fixtures for pitch_aura.
"""

import numpy as np
import pytest

from pitch_aura.types import FrameRecord, FrameSequence, PitchSpec


@pytest.fixture
def pitch() -> PitchSpec:
    """Standard FIFA pitch."""
    return PitchSpec(length=105.0, width=68.0, origin="center")


@pytest.fixture
def pitch_bottom_left() -> PitchSpec:
    """Pitch with bottom-left origin."""
    return PitchSpec(length=105.0, width=68.0, origin="bottom_left")


@pytest.fixture
def simple_frame() -> FrameRecord:
    """Two teams, 3 outfield players each, known positions."""
    return FrameRecord(
        timestamp=0.0,
        period=1,
        ball_position=np.array([0.0, 0.0]),
        player_ids=["h1", "h2", "h3", "a1", "a2", "a3"],
        team_ids=["home", "home", "home", "away", "away", "away"],
        positions=np.array(
            [
                [-20.0, 0.0],
                [-10.0, 15.0],
                [-10.0, -15.0],
                [20.0, 0.0],
                [10.0, 15.0],
                [10.0, -15.0],
            ],
            dtype=np.float64,
        ),
        velocities=np.array(
            [
                [2.0, 0.0],
                [1.0, 1.0],
                [1.0, -1.0],
                [-2.0, 0.0],
                [-1.0, 1.0],
                [-1.0, -1.0],
            ],
            dtype=np.float64,
        ),
        is_goalkeeper=np.array([False] * 6),
    )


@pytest.fixture
def simple_sequence(simple_frame: FrameRecord, pitch: PitchSpec) -> FrameSequence:
    """Short sequence of 5 frames at 25 fps."""
    frames = []
    for i in range(5):
        f = FrameRecord(
            timestamp=i * 0.04,
            period=1,
            ball_position=simple_frame.ball_position.copy(),
            player_ids=list(simple_frame.player_ids),
            team_ids=list(simple_frame.team_ids),
            positions=simple_frame.positions + i * 0.04 * simple_frame.velocities,
            velocities=simple_frame.velocities.copy(),
            is_goalkeeper=simple_frame.is_goalkeeper.copy(),
        )
        frames.append(f)
    return FrameSequence(
        frames=frames,
        frame_rate=25.0,
        pitch=pitch,
        home_team_id="home",
        away_team_id="away",
    )
