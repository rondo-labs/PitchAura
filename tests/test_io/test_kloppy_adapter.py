"""
Project: PitchAura
File Created: 2026-02-23
Author: Xingnan Zhu
File Name: test_kloppy_adapter.py
Description:
    Tests for the kloppy I/O adapter.
    Unit tests use mock kloppy objects; integration tests load real data.
"""

from __future__ import annotations

from datetime import timedelta
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from pitch_aura.io.kloppy_adapter import (
    _compute_velocities,
    _extract_frame,
    from_events,
    from_tracking,
)
from pitch_aura.types import FrameRecord


# ---------------------------------------------------------------------------
# Lightweight mock objects mimicking kloppy's data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MockPoint:
    x: float
    y: float


@dataclass(frozen=True)
class MockPoint3D(MockPoint):
    z: float | None = None


@dataclass(unsafe_hash=True)
class MockTeam:
    team_id: str
    name: str = ""
    ground: str = "HOME"


@dataclass(unsafe_hash=True)
class MockPlayer:
    player_id: str
    team: MockTeam
    jersey_no: int = 0
    starting_position: Any = None


@dataclass
class MockPlayerData:
    coordinates: MockPoint | None = None
    speed: float | None = None
    distance: float | None = None
    other_data: dict = field(default_factory=dict)


@dataclass
class MockPeriod:
    id: int


@dataclass
class MockFrame:
    frame_id: int
    timestamp: timedelta
    period: MockPeriod
    ball_coordinates: MockPoint3D | None
    players_data: dict[MockPlayer, MockPlayerData] = field(default_factory=dict)


@dataclass
class MockPitchDimensions:
    pitch_length: float | None = 105.0
    pitch_width: float | None = 68.0


@dataclass
class MockMetadata:
    teams: list[MockTeam] = field(default_factory=list)
    pitch_dimensions: MockPitchDimensions = field(default_factory=MockPitchDimensions)
    frame_rate: float | None = 25.0
    periods: list = field(default_factory=list)


@dataclass
class MockTrackingDataset:
    metadata: MockMetadata
    records: list[MockFrame] = field(default_factory=list)


@dataclass
class MockEvent:
    event_id: str
    timestamp: timedelta
    period: MockPeriod
    event_name: str
    player: MockPlayer | None = None
    team: MockTeam | None = None
    coordinates: MockPoint | None = None


@dataclass
class MockEventDataset:
    metadata: MockMetadata
    records: list[MockEvent] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def home_team() -> MockTeam:
    return MockTeam(team_id="home", name="Home FC", ground="HOME")


@pytest.fixture
def away_team() -> MockTeam:
    return MockTeam(team_id="away", name="Away FC", ground="AWAY")


@pytest.fixture
def players(home_team: MockTeam, away_team: MockTeam) -> list[MockPlayer]:
    return [
        MockPlayer(player_id="h1", team=home_team, jersey_no=1),
        MockPlayer(player_id="h2", team=home_team, jersey_no=7),
        MockPlayer(player_id="a1", team=away_team, jersey_no=1),
        MockPlayer(player_id="a2", team=away_team, jersey_no=9),
    ]


def _make_frame(
    frame_id: int,
    timestamp_sec: float,
    players: list[MockPlayer],
    positions: list[tuple[float, float]],
    ball: tuple[float, float] = (0.5, 0.5),
) -> MockFrame:
    """Helper to build a MockFrame."""
    players_data = {}
    for player, pos in zip(players, positions):
        players_data[player] = MockPlayerData(
            coordinates=MockPoint(x=pos[0], y=pos[1]),
        )
    return MockFrame(
        frame_id=frame_id,
        timestamp=timedelta(seconds=timestamp_sec),
        period=MockPeriod(id=1),
        ball_coordinates=MockPoint3D(x=ball[0], y=ball[1], z=0.0),
        players_data=players_data,
    )


# ---------------------------------------------------------------------------
# Tests: _extract_frame
# ---------------------------------------------------------------------------

class TestExtractFrame:
    def test_basic_extraction(self, players: list[MockPlayer]):
        frame = _make_frame(
            frame_id=0,
            timestamp_sec=0.0,
            players=players,
            positions=[(0.1, 0.2), (0.3, 0.4), (0.6, 0.7), (0.8, 0.9)],
        )
        rec = _extract_frame(frame, pitch_length=105.0, pitch_width=68.0)

        assert rec is not None
        assert rec.n_players == 4
        assert rec.timestamp == 0.0
        assert rec.period == 1

        # Check coordinate scaling: x * 105, y * 68
        np.testing.assert_allclose(rec.positions[0], [0.1 * 105, 0.2 * 68])
        np.testing.assert_allclose(rec.positions[2], [0.6 * 105, 0.7 * 68])

    def test_ball_coordinates_scaled(self, players: list[MockPlayer]):
        frame = _make_frame(
            frame_id=0,
            timestamp_sec=0.0,
            players=players,
            positions=[(0.5, 0.5)] * 4,
            ball=(0.5, 0.5),
        )
        rec = _extract_frame(frame, pitch_length=105.0, pitch_width=68.0)
        assert rec is not None
        np.testing.assert_allclose(rec.ball_position[:2], [52.5, 34.0])

    def test_empty_frame_returns_none(self):
        frame = MockFrame(
            frame_id=0,
            timestamp=timedelta(0),
            period=MockPeriod(id=1),
            ball_coordinates=None,
            players_data={},
        )
        assert _extract_frame(frame, 105.0, 68.0) is None

    def test_player_with_none_coordinates_skipped(self, players: list[MockPlayer]):
        players_data = {
            players[0]: MockPlayerData(coordinates=MockPoint(0.5, 0.5)),
            players[1]: MockPlayerData(coordinates=None),  # skipped
        }
        frame = MockFrame(
            frame_id=0,
            timestamp=timedelta(0),
            period=MockPeriod(id=1),
            ball_coordinates=MockPoint3D(0.5, 0.5, 0.0),
            players_data=players_data,
        )
        rec = _extract_frame(frame, 105.0, 68.0)
        assert rec is not None
        assert rec.n_players == 1

    def test_team_ids(self, players: list[MockPlayer]):
        frame = _make_frame(
            frame_id=0,
            timestamp_sec=0.0,
            players=players,
            positions=[(0.1, 0.1)] * 4,
        )
        rec = _extract_frame(frame, 105.0, 68.0)
        assert rec is not None
        assert rec.team_ids.count("home") == 2
        assert rec.team_ids.count("away") == 2


# ---------------------------------------------------------------------------
# Tests: _compute_velocities
# ---------------------------------------------------------------------------

class TestComputeVelocities:
    def test_central_difference(self):
        """Three frames, velocity at middle frame uses central diff."""
        frames = [
            FrameRecord(
                timestamp=i * 0.04,
                period=1,
                ball_position=np.array([0.0, 0.0]),
                player_ids=["p1"],
                team_ids=["t1"],
                positions=np.array([[float(i), 0.0]]),
                is_goalkeeper=np.array([False]),
            )
            for i in range(3)
        ]
        _compute_velocities(frames, frame_rate=25.0)

        # Central diff for middle frame: (2.0 - 0.0) / (2 * 0.04) = 25.0
        assert frames[1].velocities is not None
        np.testing.assert_allclose(frames[1].velocities[0, 0], 25.0, atol=1e-10)

    def test_forward_and_backward(self):
        """First frame uses forward diff, last uses backward diff."""
        frames = [
            FrameRecord(
                timestamp=i * 0.04,
                period=1,
                ball_position=np.array([0.0, 0.0]),
                player_ids=["p1"],
                team_ids=["t1"],
                positions=np.array([[float(i), 0.0]]),
                is_goalkeeper=np.array([False]),
            )
            for i in range(3)
        ]
        _compute_velocities(frames, frame_rate=25.0)

        # Forward diff: (1.0 - 0.0) / 0.04 = 25.0
        assert frames[0].velocities is not None
        np.testing.assert_allclose(frames[0].velocities[0, 0], 25.0, atol=1e-10)

        # Backward diff: (2.0 - 1.0) / 0.04 = 25.0
        assert frames[2].velocities is not None
        np.testing.assert_allclose(frames[2].velocities[0, 0], 25.0, atol=1e-10)

    def test_single_frame_no_velocity(self):
        """Single frame: velocities remain None."""
        frames = [
            FrameRecord(
                timestamp=0.0,
                period=1,
                ball_position=np.array([0.0, 0.0]),
                player_ids=["p1"],
                team_ids=["t1"],
                positions=np.array([[5.0, 3.0]]),
                is_goalkeeper=np.array([False]),
            )
        ]
        _compute_velocities(frames, frame_rate=25.0)
        assert frames[0].velocities is None


# ---------------------------------------------------------------------------
# Tests: from_tracking
# ---------------------------------------------------------------------------

class TestFromTracking:
    def test_full_conversion(
        self,
        home_team: MockTeam,
        away_team: MockTeam,
        players: list[MockPlayer],
    ):
        meta = MockMetadata(
            teams=[home_team, away_team],
            frame_rate=25.0,
        )
        frames = [
            _make_frame(i, i * 0.04, players, [
                (0.1 + i * 0.001, 0.2),
                (0.3, 0.4 + i * 0.001),
                (0.6, 0.7),
                (0.8, 0.9),
            ])
            for i in range(5)
        ]
        dataset = MockTrackingDataset(metadata=meta, records=frames)

        seq = from_tracking(dataset)

        assert len(seq) == 5
        assert seq.frame_rate == 25.0
        assert seq.home_team_id == "home"
        assert seq.away_team_id == "away"
        assert seq.pitch.length == 105.0
        assert seq.pitch.width == 68.0

        # Velocities should be computed
        assert seq[2].velocities is not None
        assert seq[2].velocities.shape == (4, 2)

    def test_empty_dataset(self, home_team: MockTeam, away_team: MockTeam):
        meta = MockMetadata(teams=[home_team, away_team])
        dataset = MockTrackingDataset(metadata=meta, records=[])
        seq = from_tracking(dataset)
        assert len(seq) == 0


# ---------------------------------------------------------------------------
# Tests: from_events
# ---------------------------------------------------------------------------

class TestFromEvents:
    def test_basic_events(self, home_team: MockTeam, players: list[MockPlayer]):
        meta = MockMetadata(teams=[home_team])
        events = [
            MockEvent(
                event_id="e1",
                timestamp=timedelta(seconds=10.5),
                period=MockPeriod(id=1),
                event_name="pass",
                player=players[0],
                team=home_team,
                coordinates=MockPoint(0.3, 0.4),
            ),
            MockEvent(
                event_id="e2",
                timestamp=timedelta(seconds=12.0),
                period=MockPeriod(id=1),
                event_name="shot",
                player=players[1],
                team=home_team,
                coordinates=MockPoint(0.8, 0.5),
            ),
        ]
        dataset = MockEventDataset(metadata=meta, records=events)

        records = from_events(dataset)

        assert len(records) == 2
        assert records[0].event_type == "pass"
        assert records[0].timestamp == 10.5
        assert records[0].player_id == "h1"
        assert records[0].coordinates is not None
        np.testing.assert_allclose(
            records[0].coordinates, [0.3 * 105, 0.4 * 68],
        )

    def test_event_without_player(self, home_team: MockTeam):
        meta = MockMetadata(teams=[home_team])
        events = [
            MockEvent(
                event_id="e1",
                timestamp=timedelta(seconds=0.0),
                period=MockPeriod(id=1),
                event_name="whistle",
                player=None,
                team=None,
                coordinates=None,
            ),
        ]
        dataset = MockEventDataset(metadata=meta, records=events)

        records = from_events(dataset)
        assert len(records) == 1
        assert records[0].player_id is None
        assert records[0].team_id is None
        assert records[0].coordinates is None
