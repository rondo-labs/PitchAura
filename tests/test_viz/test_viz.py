"""
Project: PitchAura
File Created: 2026-02-24
Author: Xingnan Zhu
File Name: test_viz.py
Description:
    Tests for the pitch_aura.viz module.
    Verifies that all public functions return go.Figure instances with the
    correct number of traces and proper structure. Does not perform visual
    regression testing (no golden images).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("plotly")

import plotly.graph_objects as go

from pitch_aura.types import FrameRecord, FrameSequence, PitchSpec, ProbabilityGrid
from pitch_aura.viz._pitch_draw import pitch_background
from pitch_aura.viz.animation import animate_sequence
from pitch_aura.viz.heatmap import plot_heatmap
from pitch_aura.viz.players import plot_players
from pitch_aura.viz.tactics import plot_passing_lane, plot_pockets
from pitch_aura.viz.voronoi import plot_voronoi
from pitch_aura.viz import plot_pitch_control, plot_voronoi_control
from pitch_aura.tactics.line_breaking import Pocket


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pitch():
    return PitchSpec(length=105.0, width=68.0)


@pytest.fixture
def simple_grid(pitch):
    nx, ny = 10, 7
    x_edges = np.linspace(-52.5, 52.5, nx + 1)
    y_edges = np.linspace(-34.0, 34.0, ny + 1)
    return ProbabilityGrid(
        values=np.full((nx, ny), 0.5),
        x_edges=x_edges,
        y_edges=y_edges,
        pitch=pitch,
        timestamp=0.0,
    )


@pytest.fixture
def simple_frame(pitch):
    return FrameRecord(
        timestamp=0.0,
        period=1,
        ball_position=np.array([0.0, 0.0]),
        player_ids=["h1", "h2", "a1"],
        team_ids=["home", "home", "away"],
        positions=np.array([[10.0, 5.0], [-10.0, -5.0], [20.0, 0.0]]),
        velocities=np.array([[2.0, 1.0], [-1.0, 0.5], [3.0, -1.0]]),
        is_goalkeeper=np.array([False, True, False]),
    )


@pytest.fixture
def simple_voronoi(simple_frame):
    from pitch_aura.space.voronoi import VoronoiModel
    model = VoronoiModel()
    return model.control(simple_frame)


@pytest.fixture
def simple_sequence(pitch):
    dt = 1 / 25.0
    frames = [
        FrameRecord(
            timestamp=i * dt,
            period=1,
            ball_position=np.array([0.0, 0.0]),
            player_ids=["h1", "a1"],
            team_ids=["home", "away"],
            positions=np.array([[float(i), 0.0], [-float(i), 0.0]]),
            velocities=np.zeros((2, 2)),
            is_goalkeeper=np.zeros(2, dtype=bool),
        )
        for i in range(10)
    ]
    return FrameSequence(frames=frames, frame_rate=25.0, pitch=pitch,
                         home_team_id="home", away_team_id="away")


# ---------------------------------------------------------------------------
# plot_heatmap
# ---------------------------------------------------------------------------

class TestPlotHeatmap:
    def test_returns_figure(self, simple_grid):
        fig = plot_heatmap(simple_grid)
        assert isinstance(fig, go.Figure)

    def test_heatmap_trace_present(self, simple_grid):
        fig = plot_heatmap(simple_grid)
        heat_traces = [t for t in fig.data if isinstance(t, go.Heatmap)]
        assert len(heat_traces) == 1

    def test_composable_with_existing_fig(self, simple_grid, pitch):
        base_fig = pitch_background(pitch)
        n_before = len(base_fig.data)
        fig = plot_heatmap(simple_grid, fig=base_fig)
        assert len(fig.data) == n_before + 1

    def test_heatmap_shape(self, simple_grid):
        fig = plot_heatmap(simple_grid)
        heat = next(t for t in fig.data if isinstance(t, go.Heatmap))
        assert len(heat.x) == 10   # nx
        assert len(heat.y) == 7    # ny


# ---------------------------------------------------------------------------
# plot_voronoi
# ---------------------------------------------------------------------------

class TestPlotVoronoi:
    def test_returns_figure(self, simple_voronoi, simple_frame):
        fig = plot_voronoi(simple_voronoi, simple_frame)
        assert isinstance(fig, go.Figure)

    def test_polygon_traces_present(self, simple_voronoi, simple_frame):
        fig = plot_voronoi(simple_voronoi, simple_frame, show_players=False)
        filled = [t for t in fig.data
                  if isinstance(t, go.Scatter) and getattr(t, "fill", None) == "toself"]
        assert len(filled) == len(simple_voronoi.regions)


# ---------------------------------------------------------------------------
# plot_pockets
# ---------------------------------------------------------------------------

class TestPlotPockets:
    def test_returns_figure_empty(self, pitch):
        fig = plot_pockets([])
        assert isinstance(fig, go.Figure)

    def test_one_pocket_one_trace(self, pitch):
        pocket = Pocket(line_depth=30.0, y_left=-10.0, y_right=10.0,
                        width=20.0, player_left="d1", player_right="d2")
        fig = plot_pockets([pocket])
        filled = [t for t in fig.data
                  if isinstance(t, go.Scatter) and getattr(t, "fill", None) == "toself"]
        assert len(filled) == 1

    def test_multiple_pockets(self, pitch):
        pockets = [
            Pocket(30.0, -10.0, 0.0, 10.0, "d1", "d2"),
            Pocket(30.0, 5.0, 15.0, 10.0, "d2", "d3"),
        ]
        fig = plot_pockets(pockets)
        filled = [t for t in fig.data
                  if isinstance(t, go.Scatter) and getattr(t, "fill", None) == "toself"]
        assert len(filled) == 2


# ---------------------------------------------------------------------------
# plot_passing_lane
# ---------------------------------------------------------------------------

class TestPlotPassingLane:
    def test_returns_figure(self, simple_frame):
        fig = plot_passing_lane(simple_frame, "h1", "a1")
        assert isinstance(fig, go.Figure)

    def test_lane_trace_added(self, simple_frame):
        n_before = len(simple_frame.player_ids)  # just a proxy
        fig = plot_passing_lane(simple_frame, "h1", "a1", obstructed=False)
        lane_traces = [t for t in fig.data
                       if isinstance(t, go.Scatter) and "Lane" in (t.name or "")]
        assert len(lane_traces) == 1

    def test_missing_player_no_crash(self, simple_frame):
        fig = plot_passing_lane(simple_frame, "ghost", "a1")
        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# animate_sequence
# ---------------------------------------------------------------------------

class TestAnimateSequence:
    def test_returns_figure(self, simple_sequence):
        fig = animate_sequence(simple_sequence)
        assert isinstance(fig, go.Figure)

    def test_frames_created(self, simple_sequence):
        fig = animate_sequence(simple_sequence)
        assert len(fig.frames) == len(simple_sequence)

    def test_step_subsamples(self, simple_sequence):
        fig = animate_sequence(simple_sequence, step=2)
        expected = len(simple_sequence.frames[::2])
        assert len(fig.frames) == expected

    def test_slider_present(self, simple_sequence):
        fig = animate_sequence(simple_sequence)
        assert len(fig.layout.sliders) == 1

    def test_grids_length_mismatch_raises(self, simple_sequence, simple_grid):
        with pytest.raises(ValueError, match="grids length"):
            animate_sequence(simple_sequence, grids=[simple_grid])  # 1 grid, 10 frames

    def test_with_grids(self, simple_sequence, simple_grid):
        grids = [simple_grid] * len(simple_sequence)
        fig = animate_sequence(simple_sequence, grids=grids)
        assert len(fig.frames) == len(simple_sequence)

    def test_empty_sequence_returns_figure(self, pitch):
        empty = FrameSequence(frames=[], frame_rate=25.0, pitch=pitch,
                              home_team_id="home", away_team_id="away")
        fig = animate_sequence(empty)
        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# High-level convenience functions
# ---------------------------------------------------------------------------

class TestConvenienceFunctions:
    def test_plot_pitch_control_no_frame(self, simple_grid):
        fig = plot_pitch_control(simple_grid)
        assert isinstance(fig, go.Figure)

    def test_plot_pitch_control_with_frame(self, simple_grid, simple_frame):
        fig = plot_pitch_control(simple_grid, simple_frame)
        assert isinstance(fig, go.Figure)
        heat = [t for t in fig.data if isinstance(t, go.Heatmap)]
        assert len(heat) == 1

    def test_plot_voronoi_control(self, simple_voronoi, simple_frame):
        fig = plot_voronoi_control(simple_voronoi, simple_frame)
        assert isinstance(fig, go.Figure)
