"""
Project: PitchAura
File Created: 2026-02-26
Author: Xingnan Zhu
File Name: test_kloppy_events_enriched.py
Description:
    Tests for enriched event extraction from kloppy adapter.
    Covers end_coordinates, result, qualifiers, and freeze_frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import pytest

from pitch_aura.io.kloppy_adapter import from_events


# ---------------------------------------------------------------------------
# Mock objects extending the base mocks for enriched event fields
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MockPoint:
    x: float
    y: float


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
    starting_position: object = None


@dataclass
class MockPlayerData:
    coordinates: MockPoint | None = None
    speed: float | None = None


@dataclass
class MockPeriod:
    id: int


@dataclass
class MockFrame:
    players_data: dict = field(default_factory=dict)
    ball_coordinates: MockPoint | None = None


@dataclass
class MockPitchDimensions:
    pitch_length: float | None = 105.0
    pitch_width: float | None = 68.0


@dataclass
class MockMetadata:
    teams: list = field(default_factory=list)
    pitch_dimensions: MockPitchDimensions = field(default_factory=MockPitchDimensions)
    frame_rate: float | None = 25.0
    periods: list = field(default_factory=list)


@dataclass
class MockEventDataset:
    metadata: MockMetadata
    records: list = field(default_factory=list)


class MockResult:
    """Mock for kloppy result enums."""
    def __init__(self, name: str):
        self.name = name


class MockQualifierValue:
    """Mock for qualifier value enum."""
    def __init__(self, name: str):
        self.name = name


class MockQualifier:
    """Mock for a qualifier."""
    def __init__(self, value_name: str):
        self.value = MockQualifierValue(value_name)


@dataclass
class MockPassEvent:
    event_id: str
    timestamp: timedelta
    period: MockPeriod
    event_name: str = "pass"
    player: MockPlayer | None = None
    team: MockTeam | None = None
    coordinates: MockPoint | None = None
    receiver_coordinates: MockPoint | None = None
    result: MockResult | None = None
    qualifiers: list = field(default_factory=list)
    freeze_frame: MockFrame | None = None


@dataclass
class MockCarryEvent:
    event_id: str
    timestamp: timedelta
    period: MockPeriod
    event_name: str = "carry"
    player: MockPlayer | None = None
    team: MockTeam | None = None
    coordinates: MockPoint | None = None
    end_coordinates: MockPoint | None = None
    result: MockResult | None = None
    qualifiers: list = field(default_factory=list)
    freeze_frame: MockFrame | None = None


@dataclass
class MockShotEvent:
    event_id: str
    timestamp: timedelta
    period: MockPeriod
    event_name: str = "shot"
    player: MockPlayer | None = None
    team: MockTeam | None = None
    coordinates: MockPoint | None = None
    result_coordinates: MockPoint | None = None
    result: MockResult | None = None
    qualifiers: list = field(default_factory=list)
    freeze_frame: MockFrame | None = None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def team() -> MockTeam:
    return MockTeam(team_id="home", name="Home FC")


@pytest.fixture
def player(team: MockTeam) -> MockPlayer:
    return MockPlayer(player_id="p1", team=team)


@pytest.fixture
def meta(team: MockTeam) -> MockMetadata:
    return MockMetadata(teams=[team])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEnrichedFromEvents:
    def test_pass_end_coordinates(self, meta: MockMetadata, team: MockTeam, player: MockPlayer):
        event = MockPassEvent(
            event_id="e1",
            timestamp=timedelta(seconds=10.0),
            period=MockPeriod(id=1),
            player=player,
            team=team,
            coordinates=MockPoint(0.3, 0.4),
            receiver_coordinates=MockPoint(0.6, 0.5),
        )
        dataset = MockEventDataset(metadata=meta, records=[event])
        records = from_events(dataset)

        assert len(records) == 1
        assert records[0].end_coordinates is not None
        np.testing.assert_allclose(records[0].end_coordinates, [0.6 * 105, 0.5 * 68])

    def test_carry_end_coordinates(self, meta: MockMetadata, team: MockTeam, player: MockPlayer):
        event = MockCarryEvent(
            event_id="e1",
            timestamp=timedelta(seconds=10.0),
            period=MockPeriod(id=1),
            player=player,
            team=team,
            coordinates=MockPoint(0.3, 0.4),
            end_coordinates=MockPoint(0.5, 0.6),
        )
        dataset = MockEventDataset(metadata=meta, records=[event])
        records = from_events(dataset)

        assert records[0].end_coordinates is not None
        np.testing.assert_allclose(records[0].end_coordinates, [0.5 * 105, 0.6 * 68])

    def test_shot_result_coordinates(self, meta: MockMetadata, team: MockTeam, player: MockPlayer):
        event = MockShotEvent(
            event_id="e1",
            timestamp=timedelta(seconds=10.0),
            period=MockPeriod(id=1),
            player=player,
            team=team,
            coordinates=MockPoint(0.8, 0.5),
            result_coordinates=MockPoint(1.0, 0.5),
        )
        dataset = MockEventDataset(metadata=meta, records=[event])
        records = from_events(dataset)

        assert records[0].end_coordinates is not None
        np.testing.assert_allclose(records[0].end_coordinates, [105.0, 34.0])

    def test_result_extraction(self, meta: MockMetadata, team: MockTeam, player: MockPlayer):
        event = MockPassEvent(
            event_id="e1",
            timestamp=timedelta(seconds=10.0),
            period=MockPeriod(id=1),
            player=player,
            team=team,
            coordinates=MockPoint(0.3, 0.4),
            result=MockResult("COMPLETE"),
        )
        dataset = MockEventDataset(metadata=meta, records=[event])
        records = from_events(dataset)

        assert records[0].result == "complete"

    def test_qualifiers_extraction(self, meta: MockMetadata, team: MockTeam, player: MockPlayer):
        event = MockPassEvent(
            event_id="e1",
            timestamp=timedelta(seconds=10.0),
            period=MockPeriod(id=1),
            player=player,
            team=team,
            coordinates=MockPoint(0.3, 0.4),
            qualifiers=[MockQualifier("CROSS"), MockQualifier("THROUGH_BALL")],
        )
        dataset = MockEventDataset(metadata=meta, records=[event])
        records = from_events(dataset)

        assert records[0].qualifiers == ("CROSS", "THROUGH_BALL")

    def test_freeze_frame_extraction(self, meta: MockMetadata, team: MockTeam, player: MockPlayer):
        other_team = MockTeam(team_id="away")
        other_player = MockPlayer(player_id="a1", team=other_team)

        ff = MockFrame(
            players_data={
                player: MockPlayerData(coordinates=MockPoint(0.3, 0.4)),
                other_player: MockPlayerData(coordinates=MockPoint(0.7, 0.6)),
            },
            ball_coordinates=MockPoint(0.5, 0.5),
        )
        event = MockPassEvent(
            event_id="e1",
            timestamp=timedelta(seconds=10.0),
            period=MockPeriod(id=1),
            player=player,
            team=team,
            coordinates=MockPoint(0.3, 0.4),
            freeze_frame=ff,
        )
        dataset = MockEventDataset(metadata=meta, records=[event])
        records = from_events(dataset)

        assert records[0].freeze_frame is not None
        frame = records[0].freeze_frame
        assert frame.n_players == 2
        assert frame.velocities is None
        assert frame.timestamp == 10.0
        np.testing.assert_allclose(frame.ball_position[:2], [52.5, 34.0])

    def test_no_enriched_fields(self, meta: MockMetadata, team: MockTeam, player: MockPlayer):
        """Basic event without enriched fields should still work."""
        event = MockPassEvent(
            event_id="e1",
            timestamp=timedelta(seconds=5.0),
            period=MockPeriod(id=1),
            player=player,
            team=team,
            coordinates=MockPoint(0.3, 0.4),
        )
        dataset = MockEventDataset(metadata=meta, records=[event])
        records = from_events(dataset)

        assert records[0].end_coordinates is None
        assert records[0].result is None
        assert records[0].qualifiers == ()
        assert records[0].freeze_frame is None
