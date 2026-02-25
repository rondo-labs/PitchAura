"""
Project: PitchAura
File Created: 2026-02-24
Author: Xingnan Zhu
File Name: voronoi.py
Description:
    Voronoi tessellation visualisation layer.
    Renders each player's Voronoi region as a team-coloured filled polygon
    using go.Scatter with fill="toself". Optionally overlays player positions
    and displays area labels. Composable via the fig parameter.
"""

from __future__ import annotations

import plotly.graph_objects as go

from pitch_aura.types import FrameRecord, PitchSpec, VoronoiResult

from ._pitch_draw import pitch_background
from .players import plot_players


def plot_voronoi(
    result: VoronoiResult,
    frame: FrameRecord,
    *,
    fig: go.Figure | None = None,
    pitch: PitchSpec | None = None,
    home_team_id: str | None = None,
    home_color: str = "#3b82f6",
    away_color: str = "#ef4444",
    opacity: float = 0.3,
    show_players: bool = True,
    show_areas: bool = False,
    show_pitch: bool = True,
    bgcolor: str = "#1a472a",
    line_color: str = "white",
) -> go.Figure:
    """Render Voronoi regions as team-coloured filled polygons.

    Parameters:
        result:        Voronoi tessellation result from
                       :class:`~pitch_aura.space.voronoi.VoronoiModel`.
        frame:         The :class:`~pitch_aura.types.FrameRecord` associated
                       with *result* (needed for team membership lookup).
        fig:           Existing figure to add traces to.
        pitch:         Pitch spec for background (defaults to 105×68 m).
        home_team_id:  Team ID for home side; determines fill colour.
        home_color:    Fill colour for home team regions (default blue).
        away_color:    Fill colour for away team regions (default red).
        opacity:       Polygon fill opacity (default 0.3).
        show_players:  Overlay player markers on top of Voronoi regions.
        show_areas:    Annotate each region centroid with the area in m².
        show_pitch:    Create pitch background when ``fig=None``.
        bgcolor:       Pitch background colour.
        line_color:    Pitch line colour.

    Returns:
        ``go.Figure`` with Voronoi polygon traces (and optionally players).
    """
    from pitch_aura.types import PitchSpec as _PitchSpec
    resolved_pitch = pitch if pitch is not None else _PitchSpec()

    if fig is None:
        fig = pitch_background(
            resolved_pitch, bgcolor=bgcolor, line_color=line_color,
        ) if show_pitch else go.Figure()

    pid_to_team = dict(zip(frame.player_ids, frame.team_ids))

    for pid, polygon in result.regions.items():
        if polygon is None or len(polygon) < 3:
            continue
        tid = pid_to_team.get(pid, "")
        if home_team_id is None:
            color = home_color
        else:
            color = home_color if tid == home_team_id else away_color

        area = result.areas.get(pid, 0.0)
        hover = f"{pid}<br>Team: {tid}<br>Area: {area:.1f} m²"

        xs = list(polygon[:, 0]) + [polygon[0, 0]]  # close polygon
        ys = list(polygon[:, 1]) + [polygon[0, 1]]

        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines",
            fill="toself",
            fillcolor=color,
            opacity=opacity,
            line=dict(color=color, width=1),
            name=f"{pid} ({tid})",
            hovertemplate=hover + "<extra></extra>",
            showlegend=False,
        ))

        if show_areas:
            cx_poly = float(polygon[:, 0].mean())
            cy_poly = float(polygon[:, 1].mean())
            fig.add_annotation(
                x=cx_poly, y=cy_poly,
                text=f"{area:.0f}",
                showarrow=False,
                font=dict(size=8, color="white"),
            )

    if show_players:
        fig = plot_players(
            frame, fig=fig,
            home_team_id=home_team_id,
            home_color=home_color,
            away_color=away_color,
            show_pitch=False,
        )

    return fig
