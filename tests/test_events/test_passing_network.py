"""
Project: PitchAura
File Created: 2026-02-26
Author: Xingnan Zhu
File Name: test_passing_network.py
Description:
    Tests for spatial passing network construction.
"""

from __future__ import annotations

import numpy as np
import pytest

from pitch_aura.events.passing_network import PassingNetwork, passing_network
from pitch_aura.types import EventRecord


def _pass_event(
    passer: str,
    receiver_id: str | None,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    team_id: str = "t1",
    result: str = "complete",
    timestamp: float = 0.0,
) -> list[EventRecord]:
    """Create a pass event followed by a receiver touch event."""
    events = [
        EventRecord(
            timestamp=timestamp,
            period=1,
            event_type="pass",
            player_id=passer,
            team_id=team_id,
            coordinates=np.array(start, dtype=np.float64),
            end_coordinates=np.array(end, dtype=np.float64),
            result=result,
        ),
    ]
    if receiver_id is not None and result == "complete":
        events.append(
            EventRecord(
                timestamp=timestamp + 1.0,
                period=1,
                event_type="receipt",
                player_id=receiver_id,
                team_id=team_id,
                coordinates=np.array(end, dtype=np.float64),
            ),
        )
    return events


class TestPassingNetwork:
    def test_empty_events(self):
        net = passing_network([])
        assert isinstance(net, PassingNetwork)
        assert net.nodes.empty
        assert net.edges.empty

    def test_single_completed_pass(self):
        events = _pass_event("p1", "p2", (30.0, 34.0), (50.0, 34.0))
        net = passing_network(events)

        assert len(net.nodes) == 1  # only passer has pass_count
        assert net.nodes.iloc[0]["player_id"] == "p1"
        assert net.nodes.iloc[0]["pass_count"] == 1
        np.testing.assert_allclose(net.nodes.iloc[0]["avg_x"], 30.0)

        assert len(net.edges) == 1
        assert net.edges.iloc[0]["passer"] == "p1"
        assert net.edges.iloc[0]["receiver"] == "p2"
        assert net.edges.iloc[0]["count"] == 1

    def test_multiple_passes_same_pair(self):
        events = (
            _pass_event("p1", "p2", (30.0, 30.0), (50.0, 30.0), timestamp=0.0)
            + _pass_event("p1", "p2", (40.0, 40.0), (60.0, 40.0), timestamp=5.0)
        )
        net = passing_network(events)

        assert len(net.nodes) == 1
        assert net.nodes.iloc[0]["pass_count"] == 2
        np.testing.assert_allclose(net.nodes.iloc[0]["avg_x"], 35.0)

        assert len(net.edges) == 1
        assert net.edges.iloc[0]["count"] == 2

    def test_incomplete_pass_no_edge(self):
        events = _pass_event(
            "p1", None, (30.0, 34.0), (50.0, 34.0), result="incomplete",
        )
        net = passing_network(events)

        # Node exists (player made a pass attempt)
        assert len(net.nodes) == 1
        # No edge (pass was incomplete)
        assert net.edges.empty

    def test_team_filter(self):
        events = (
            _pass_event("p1", "p2", (30.0, 34.0), (50.0, 34.0), team_id="t1")
            + _pass_event("p3", "p4", (60.0, 34.0), (80.0, 34.0), team_id="t2")
        )
        net = passing_network(events, team_id="t1")

        assert len(net.nodes) == 1
        assert net.nodes.iloc[0]["player_id"] == "p1"

    def test_min_passes_filter(self):
        events = _pass_event("p1", "p2", (30.0, 34.0), (50.0, 34.0))
        net = passing_network(events, min_passes=5)

        # p1 only has 1 pass, min_passes=5 filters them out
        assert net.nodes.empty

    def test_edge_avg_coordinates(self):
        events = (
            _pass_event("p1", "p2", (20.0, 30.0), (40.0, 30.0), timestamp=0.0)
            + _pass_event("p1", "p2", (30.0, 40.0), (50.0, 40.0), timestamp=5.0)
        )
        net = passing_network(events)

        edge = net.edges.iloc[0]
        np.testing.assert_allclose(edge["avg_start_x"], 25.0)
        np.testing.assert_allclose(edge["avg_start_y"], 35.0)
        np.testing.assert_allclose(edge["avg_end_x"], 45.0)
        np.testing.assert_allclose(edge["avg_end_y"], 35.0)

    def test_no_pass_events(self):
        events = [
            EventRecord(
                timestamp=0.0, period=1, event_type="shot",
                player_id="p1", team_id="t1",
                coordinates=np.array([80.0, 34.0]),
            ),
        ]
        net = passing_network(events)
        assert net.nodes.empty
        assert net.edges.empty
