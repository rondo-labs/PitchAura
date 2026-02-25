"""
Project: PitchAura
File Created: 2026-02-24
Author: Xingnan Zhu
File Name: _pitch_draw.py
Description:
    Internal pitch background rendering for Plotly figures.
    Converts a PitchSpec into a list of Plotly layout.shapes covering
    full FIFA-standard pitch markings: outer boundary, halfway line,
    centre circle, penalty areas, goal areas, penalty spots, and corner arcs.
    All coordinates are computed dynamically from PitchSpec dimensions,
    supporting both origin="center" and origin="bottom_left".
"""

from __future__ import annotations

import math

import plotly.graph_objects as go

from pitch_aura.types import PitchSpec

# ---------------------------------------------------------------------------
# FIFA standard measurements (metres)
# ---------------------------------------------------------------------------
_GOAL_WIDTH = 7.32
_PENALTY_AREA_HALF_WIDTH = _GOAL_WIDTH / 2 + 16.5   # 20.16 m from y-centre
_PENALTY_AREA_DEPTH = 16.5
_GOAL_AREA_HALF_WIDTH = _GOAL_WIDTH / 2 + 5.5        # 9.16 m from y-centre
_GOAL_AREA_DEPTH = 5.5
_PENALTY_SPOT_DEPTH = 11.0
_CENTRE_RADIUS = 9.15
_CORNER_RADIUS = 1.0


def _circle_path(cx: float, cy: float, r: float, n: int = 64) -> tuple[list[float], list[float]]:
    """Return (x, y) coordinate lists for a full circle."""
    angles = [2 * math.pi * i / n for i in range(n + 1)]
    xs = [cx + r * math.cos(a) for a in angles]
    ys = [cy + r * math.sin(a) for a in angles]
    return xs, ys


def _arc_path(
    cx: float,
    cy: float,
    r: float,
    a0_deg: float,
    a1_deg: float,
    n: int = 32,
) -> tuple[list[float], list[float]]:
    """Return (x, y) lists for an arc from a0_deg to a1_deg."""
    a0 = math.radians(a0_deg)
    a1 = math.radians(a1_deg)
    angles = [a0 + (a1 - a0) * i / n for i in range(n + 1)]
    xs = [cx + r * math.cos(a) for a in angles]
    ys = [cy + r * math.sin(a) for a in angles]
    return xs, ys


def _make_pitch_traces(pitch: PitchSpec, line_color: str) -> list[go.Scatter]:
    """Build all pitch marking traces as go.Scatter objects.

    Using Scatter traces (rather than layout.shapes) allows them to appear
    correctly over heatmap layers and supports transparent fills.

    Parameters:
        pitch:      Pitch specification.
        line_color: CSS color string for all pitch markings.

    Returns:
        List of ``go.Scatter`` traces to add to a figure.
    """
    x0, x1 = pitch.x_range
    y0, y1 = pitch.y_range
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0

    def line(xs: list[float], ys: list[float]) -> go.Scatter:
        return go.Scatter(
            x=xs, y=ys,
            mode="lines",
            line={"color": line_color, "width": 1.5},
            hoverinfo="skip",
            showlegend=False,
        )

    def dot(x: float, y: float) -> go.Scatter:
        return go.Scatter(
            x=[x], y=[y],
            mode="markers",
            marker={"color": line_color, "size": 5},
            hoverinfo="skip",
            showlegend=False,
        )

    traces: list[go.Scatter] = []

    # Outer boundary
    traces.append(line(
        [x0, x1, x1, x0, x0],
        [y0, y0, y1, y1, y0],
    ))

    # Halfway line
    traces.append(line([cx, cx], [y0, y1]))

    # Centre circle
    xs, ys = _circle_path(cx, cy, _CENTRE_RADIUS)
    traces.append(line(xs, ys))

    # Centre spot
    traces.append(dot(cx, cy))

    # Left penalty area
    pa_left_x0 = x0
    pa_left_x1 = x0 + _PENALTY_AREA_DEPTH
    traces.append(line(
        [pa_left_x0, pa_left_x1, pa_left_x1, pa_left_x0],
        [cy - _PENALTY_AREA_HALF_WIDTH, cy - _PENALTY_AREA_HALF_WIDTH,
         cy + _PENALTY_AREA_HALF_WIDTH, cy + _PENALTY_AREA_HALF_WIDTH],
    ))

    # Left goal area
    traces.append(line(
        [x0, x0 + _GOAL_AREA_DEPTH, x0 + _GOAL_AREA_DEPTH, x0],
        [cy - _GOAL_AREA_HALF_WIDTH, cy - _GOAL_AREA_HALF_WIDTH,
         cy + _GOAL_AREA_HALF_WIDTH, cy + _GOAL_AREA_HALF_WIDTH],
    ))

    # Left penalty spot + D-arc (arc outside penalty area)
    lp_spot_x = x0 + _PENALTY_SPOT_DEPTH
    traces.append(dot(lp_spot_x, cy))
    arc_xs, arc_ys = _arc_path(lp_spot_x, cy, _CENTRE_RADIUS, -53.0, 53.0)
    # Keep only points outside the penalty area
    arc_xs_clip = [x for x, y in zip(arc_xs, arc_ys) if x > pa_left_x1]
    arc_ys_clip = [y for x, y in zip(arc_xs, arc_ys) if x > pa_left_x1]
    if arc_xs_clip:
        traces.append(line(arc_xs_clip, arc_ys_clip))

    # Right penalty area
    pa_right_x1 = x1
    pa_right_x0 = x1 - _PENALTY_AREA_DEPTH
    traces.append(line(
        [pa_right_x0, pa_right_x1, pa_right_x1, pa_right_x0, pa_right_x0],
        [cy - _PENALTY_AREA_HALF_WIDTH, cy - _PENALTY_AREA_HALF_WIDTH,
         cy + _PENALTY_AREA_HALF_WIDTH, cy + _PENALTY_AREA_HALF_WIDTH,
         cy - _PENALTY_AREA_HALF_WIDTH],
    ))

    # Right goal area
    traces.append(line(
        [x1, x1 - _GOAL_AREA_DEPTH, x1 - _GOAL_AREA_DEPTH, x1],
        [cy - _GOAL_AREA_HALF_WIDTH, cy - _GOAL_AREA_HALF_WIDTH,
         cy + _GOAL_AREA_HALF_WIDTH, cy + _GOAL_AREA_HALF_WIDTH],
    ))

    # Right penalty spot + D-arc
    rp_spot_x = x1 - _PENALTY_SPOT_DEPTH
    traces.append(dot(rp_spot_x, cy))
    arc_xs, arc_ys = _arc_path(rp_spot_x, cy, _CENTRE_RADIUS, 127.0, 233.0)
    arc_xs_clip = [x for x, y in zip(arc_xs, arc_ys) if x < pa_right_x0]
    arc_ys_clip = [y for x, y in zip(arc_xs, arc_ys) if x < pa_right_x0]
    if arc_xs_clip:
        traces.append(line(arc_xs_clip, arc_ys_clip))

    # Corner arcs (4 corners)
    for cx_c, cy_c, a0, a1 in [
        (x0, y0, 0.0,   90.0),   # bottom-left
        (x1, y0, 90.0,  180.0),  # bottom-right
        (x1, y1, 180.0, 270.0),  # top-right
        (x0, y1, 270.0, 360.0),  # top-left
    ]:
        arc_xs, arc_ys = _arc_path(cx_c, cy_c, _CORNER_RADIUS, a0, a1)
        traces.append(line(arc_xs, arc_ys))

    return traces


def pitch_background(
    pitch: PitchSpec,
    *,
    bgcolor: str = "#1a472a",
    line_color: str = "white",
    width: int = 760,
    height: int = 520,
) -> go.Figure:
    """Create a blank Plotly figure with FIFA-standard pitch markings.

    Parameters:
        pitch:      Pitch specification (determines coordinate ranges).
        bgcolor:    Background fill colour (default: dark green ``"#1a472a"``).
        line_color: Colour for all pitch marking lines (default: ``"white"``).
        width:      Figure pixel width (default 760).  At the default height of
                    520 px this gives a plot area of 680×440 px, matching the
                    105 m × 68 m FIFA pitch aspect ratio (≈ 1.544).
        height:     Figure pixel height (default 520).

    Returns:
        A ``go.Figure`` with pitch markings, ready for overlay traces.
    """
    x0, x1 = pitch.x_range
    y0, y1 = pitch.y_range

    fig = go.Figure()

    for trace in _make_pitch_traces(pitch, line_color):
        fig.add_trace(trace)

    fig.update_layout(
        plot_bgcolor=bgcolor,
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            range=[x0, x1],
            showgrid=False,
            zeroline=False,
            showticklabels=True,
        ),
        yaxis=dict(
            range=[y0, y1],
            showgrid=False,
            zeroline=False,
            showticklabels=True,
        ),
        margin=dict(l=40, r=40, t=40, b=40),
        width=width,
        height=height,
        showlegend=False,
    )
    return fig
