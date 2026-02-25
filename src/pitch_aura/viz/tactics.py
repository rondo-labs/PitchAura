"""
Project: PitchAura
File Created: 2026-02-24
Author: Xingnan Zhu
File Name: tactics.py
Description:
    Tactical overlay visualisation functions.
    Provides plot_pockets() for highlighting defensive line gaps and
    plot_passing_lane() for drawing the passing channel between two players.
    Both functions are composable via the fig parameter.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from pitch_aura.tactics.line_breaking import Pocket
from pitch_aura.types import FrameRecord, PitchSpec

from ._pitch_draw import pitch_background


def plot_pockets(
    pockets: list[Pocket],
    *,
    fig: go.Figure | None = None,
    pitch: PitchSpec | None = None,
    color: str = "rgba(255,215,0,0.35)",
    border_color: str = "gold",
    show_pitch: bool = True,
    bgcolor: str = "#1a472a",
    line_color: str = "white",
) -> go.Figure:
    """Render defensive line pockets as highlighted rectangles.

    Each :class:`~pitch_aura.tactics.line_breaking.Pocket` is drawn as a
    semi-transparent filled rectangle.  The rectangle spans the detected
    gap in the y-direction and a fixed depth of 3 m centred on the line's
    mean x-coordinate.

    Parameters:
        pockets:      List of pockets from
                      :func:`~pitch_aura.tactics.line_breaking.line_breaking_pockets`.
        fig:          Existing figure; creates a new one if ``None``.
        pitch:        Pitch spec for background (defaults to 105×68 m).
        color:        Fill colour of pocket rectangles (default gold).
        border_color: Border colour of pocket rectangles.
        show_pitch:   Create pitch background when ``fig=None``.
        bgcolor:      Pitch background colour.
        line_color:   Pitch line colour.

    Returns:
        ``go.Figure`` with pocket rectangles added.
    """
    from pitch_aura.types import PitchSpec as _PitchSpec
    resolved_pitch = pitch if pitch is not None else _PitchSpec()

    if fig is None:
        fig = pitch_background(
            resolved_pitch, bgcolor=bgcolor, line_color=line_color,
        ) if show_pitch else go.Figure()

    depth_half = 1.5  # ±1.5 m around line_depth

    for pocket in pockets:
        x_left = pocket.line_depth - depth_half
        x_right = pocket.line_depth + depth_half
        y_left = pocket.y_left
        y_right = pocket.y_right

        xs = [x_left, x_right, x_right, x_left, x_left]
        ys = [y_left, y_left, y_right, y_right, y_left]

        hover = (
            f"Pocket<br>Width: {pocket.width:.1f} m<br>"
            f"Left: {pocket.player_left}<br>Right: {pocket.player_right}"
        )

        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines",
            fill="toself",
            fillcolor=color,
            line=dict(color=border_color, width=1.5),
            hovertemplate=hover + "<extra></extra>",
            name=f"Pocket ({pocket.width:.1f}m)",
            showlegend=True,
        ))

    return fig


def plot_passing_lane(
    frame: FrameRecord,
    passer_id: str,
    receiver_id: str,
    *,
    lane_width: float = 2.0,
    obstructed: bool = False,
    fig: go.Figure | None = None,
    pitch: PitchSpec | None = None,
    open_color: str = "rgba(0,200,100,0.35)",
    blocked_color: str = "rgba(220,50,50,0.35)",
    show_pitch: bool = True,
    bgcolor: str = "#1a472a",
    line_color: str = "white",
) -> go.Figure:
    """Draw a passing lane rectangle between two players.

    The lane is rendered as a semi-transparent rectangle aligned with the
    vector from passer to receiver.  The colour indicates whether the lane
    is open (green) or obstructed (red).

    Parameters:
        frame:        Tracking frame containing both players.
        passer_id:    Player ID of the passer.
        receiver_id:  Player ID of the intended receiver.
        lane_width:   Width of the lane in metres (default 2 m).
        obstructed:   If ``True``, renders in *blocked_color*.
        fig:          Existing figure to add to.
        pitch:        Pitch spec for background.
        open_color:   Fill colour when lane is unobstructed (default green).
        blocked_color:Fill colour when lane is obstructed (default red).
        show_pitch:   Create pitch background when ``fig=None``.
        bgcolor:      Pitch background colour.
        line_color:   Pitch line colour.

    Returns:
        ``go.Figure`` with the lane rectangle added.
    """
    from pitch_aura.types import PitchSpec as _PitchSpec
    resolved_pitch = pitch if pitch is not None else _PitchSpec()

    if fig is None:
        fig = pitch_background(
            resolved_pitch, bgcolor=bgcolor, line_color=line_color,
        ) if show_pitch else go.Figure()

    pid_idx = {pid: i for i, pid in enumerate(frame.player_ids)}
    if passer_id not in pid_idx or receiver_id not in pid_idx:
        return fig

    p = frame.positions[pid_idx[passer_id]]
    r = frame.positions[pid_idx[receiver_id]]

    lane_vec = r - p
    lane_len = float(np.linalg.norm(lane_vec))
    if lane_len < 1e-6:
        return fig

    axis = lane_vec / lane_len
    perp = np.array([-axis[1], axis[0]]) * (lane_width / 2.0)

    # Four corners of the lane rectangle
    corners = np.array([
        p + perp,
        r + perp,
        r - perp,
        p - perp,
        p + perp,
    ])

    fill_color = blocked_color if obstructed else open_color
    border = "red" if obstructed else "lime"
    label = "Blocked" if obstructed else "Open"

    fig.add_trace(go.Scatter(
        x=corners[:, 0],
        y=corners[:, 1],
        mode="lines",
        fill="toself",
        fillcolor=fill_color,
        line=dict(color=border, width=1.5),
        name=f"Lane ({label})",
        hovertemplate=f"Passing lane<br>{passer_id} → {receiver_id}<br>Status: {label}<extra></extra>",
    ))

    return fig
