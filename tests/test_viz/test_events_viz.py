"""
Project: PitchAura
File Created: 2026-02-26
Author: Xingnan Zhu
File Name: test_events_viz.py
Description:
    Tests for event-based visualisation functions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from pitch_aura.events.passing_network import PassingNetwork
from pitch_aura.types import PitchSpec
from pitch_aura.viz.events import (
    plot_event_zones,
    plot_passing_network,
    plot_progressive_passes,
)


@pytest.fixture
def pitch() -> PitchSpec:
    return PitchSpec(length=105.0, width=68.0, origin="bottom_left")


class TestPlotPassingNetwork:
    def test_returns_figure(self, pitch: PitchSpec):
        nodes = pd.DataFrame({
            "player_id": ["p1", "p2"],
            "team_id": ["t1", "t1"],
            "avg_x": [30.0, 60.0],
            "avg_y": [34.0, 34.0],
            "pass_count": [5, 3],
        })
        edges = pd.DataFrame({
            "passer": ["p1"],
            "receiver": ["p2"],
            "avg_start_x": [30.0],
            "avg_start_y": [34.0],
            "avg_end_x": [60.0],
            "avg_end_y": [34.0],
            "count": [3],
            "completion_rate": [0.75],
        })
        net = PassingNetwork(nodes=nodes, edges=edges)
        fig = plot_passing_network(net, pitch=pitch)
        assert isinstance(fig, go.Figure)

    def test_empty_network(self, pitch: PitchSpec):
        net = PassingNetwork(
            nodes=pd.DataFrame(columns=["player_id", "team_id", "avg_x", "avg_y", "pass_count"]),
            edges=pd.DataFrame(columns=[
                "passer", "receiver", "avg_start_x", "avg_start_y",
                "avg_end_x", "avg_end_y", "count", "completion_rate",
            ]),
        )
        fig = plot_passing_network(net, pitch=pitch)
        assert isinstance(fig, go.Figure)

    def test_accepts_existing_fig(self, pitch: PitchSpec):
        existing = go.Figure()
        existing.add_trace(go.Scatter(x=[0], y=[0], name="dummy"))

        net = PassingNetwork(
            nodes=pd.DataFrame(columns=["player_id", "team_id", "avg_x", "avg_y", "pass_count"]),
            edges=pd.DataFrame(columns=[
                "passer", "receiver", "avg_start_x", "avg_start_y",
                "avg_end_x", "avg_end_y", "count", "completion_rate",
            ]),
        )
        fig = plot_passing_network(net, fig=existing)
        assert len(fig.data) >= 1


class TestPlotProgressivePasses:
    def test_returns_figure(self, pitch: PitchSpec):
        df = pd.DataFrame({
            "start_x": [30.0, 60.0],
            "start_y": [34.0, 34.0],
            "end_x": [60.0, 40.0],
            "end_y": [34.0, 34.0],
            "is_progressive": [True, False],
        })
        fig = plot_progressive_passes(df, pitch=pitch)
        assert isinstance(fig, go.Figure)

    def test_empty_dataframe(self, pitch: PitchSpec):
        df = pd.DataFrame(columns=["start_x", "start_y", "end_x", "end_y", "is_progressive"])
        fig = plot_progressive_passes(df, pitch=pitch)
        assert isinstance(fig, go.Figure)


class TestPlotEventZones:
    def test_returns_figure(self, pitch: PitchSpec):
        df = pd.DataFrame({
            "zone_x": [0, 0, 1, 1],
            "zone_y": [0, 1, 0, 1],
            "x_center": [26.25, 26.25, 78.75, 78.75],
            "y_center": [17.0, 51.0, 17.0, 51.0],
            "count": [5, 3, 8, 2],
            "frequency": [0.278, 0.167, 0.444, 0.111],
        })
        fig = plot_event_zones(df, pitch=pitch)
        assert isinstance(fig, go.Figure)

    def test_empty_dataframe(self, pitch: PitchSpec):
        df = pd.DataFrame(columns=["zone_x", "zone_y", "x_center", "y_center", "count", "frequency"])
        fig = plot_event_zones(df, pitch=pitch)
        assert isinstance(fig, go.Figure)
