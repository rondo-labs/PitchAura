"""
Project: PitchAura
File Created: 2026-02-24
Author: Xingnan Zhu
File Name: tactics.py
Description:
    Tactical overlay visualisation functions.
    Provides plot_pockets() for highlighting defensive line gaps,
    plot_passing_lane() for drawing the passing channel between two players,
    plot_deformation_field() for spatial gravity deformation heatmaps,
    plot_gravity_timeseries() for SDI/NSG time-series charts,
    plot_flow_field() for directional drag vector arrows, and
    plot_interaction_matrix() for the N×N gravity interaction heatmap.
    All functions are composable via the fig parameter.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

import pandas as pd

from pitch_aura.tactics.gravity import DeformationGrid, DeformationVectorField
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


def plot_deformation_field(
    deformation: DeformationGrid,
    frame: FrameRecord | None = None,
    *,
    fig: go.Figure | None = None,
    colorscale: str = "RdBu",
    opacity: float = 0.75,
    show_pitch: bool = True,
    show_colorbar: bool = True,
    home_team_id: str | None = None,
    home_color: str = "#3b82f6",
    away_color: str = "#ef4444",
    player_names: dict[str, str] | None = None,
    bgcolor: str = "#1a472a",
    line_color: str = "white",
) -> go.Figure:
    """Render a :class:`DeformationGrid` as a diverging heatmap.

    Blue indicates zones where the attacking team gained control due to
    the player's movement; red indicates zones where control was lost.

    Parameters:
        deformation:   Deformation grid from :func:`spatial_drag_index`.
        frame:         Optional frame to overlay player positions.
        fig:           Existing figure; creates a new one if ``None``.
        colorscale:    Plotly diverging colorscale (default ``"RdBu"``).
        opacity:       Heatmap layer opacity.
        show_pitch:    Draw pitch markings when creating a new figure.
        show_colorbar: Show the colour legend bar.
        home_team_id:  Team ID of the home side; controls player colouring
                       when *frame* is provided.
        home_color:    Marker colour for home players (default blue).
        away_color:    Marker colour for away players (default red).
        player_names:  Optional mapping from player_id to display name,
                       forwarded to :func:`~pitch_aura.viz.players.plot_players`.
        bgcolor:       Pitch background colour.
        line_color:    Pitch line colour.

    Returns:
        ``go.Figure`` with the deformation heatmap added.
    """
    if fig is None:
        fig = pitch_background(
            deformation.pitch, bgcolor=bgcolor, line_color=line_color,
        ) if show_pitch else go.Figure()

    x_centers = (deformation.x_edges[:-1] + deformation.x_edges[1:]) / 2.0
    y_centers = (deformation.y_edges[:-1] + deformation.y_edges[1:]) / 2.0

    abs_max = max(float(np.abs(deformation.values).max()), 1e-6)

    fig.add_trace(go.Heatmap(
        x=x_centers,
        y=y_centers,
        z=deformation.values.T,
        colorscale=colorscale,
        zmin=-abs_max,
        zmax=abs_max,
        zmid=0.0,
        opacity=opacity,
        showscale=show_colorbar,
        colorbar=dict(title="ΔControl", thickness=12, len=0.6),
        hovertemplate=(
            "x: %{x:.1f}<br>y: %{y:.1f}<br>"
            "Δ: %{z:.3f}<extra></extra>"
        ),
    ))

    if frame is not None:
        from pitch_aura.viz.players import plot_players
        fig = plot_players(
            frame,
            fig=fig,
            show_pitch=False,
            home_team_id=home_team_id,
            home_color=home_color,
            away_color=away_color,
            player_names=player_names,
        )

    return fig


def plot_gravity_timeseries(
    df: pd.DataFrame,
    *,
    metric_col: str,
    label: str | None = None,
    fig: go.Figure | None = None,
    line_color: str = "#3b82f6",
) -> go.Figure:
    """Plot SDI or NSG over time as a line chart.

    Parameters:
        df:          DataFrame from :func:`spatial_drag_index` or
                     :func:`net_space_generated`.
        metric_col:  Column name to plot on the y-axis (e.g. ``"sdi_m2"``
                     or ``"nsg_m2"``).
        label:       Legend label for the line (defaults to *metric_col*).
        fig:         Existing figure; creates a new one if ``None``.
        line_color:  Line colour.

    Returns:
        ``go.Figure`` with the time-series trace added.
    """
    if fig is None:
        fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df[metric_col],
        mode="lines",
        name=label or metric_col,
        line=dict(color=line_color, width=2),
        hovertemplate="t: %{x:.2f}s<br>%{y:.1f} m²<extra></extra>",
    ))

    fig.update_layout(
        xaxis_title="Time (s)",
        yaxis_title=f"{label or metric_col} (m²)",
        template="plotly_white",
    )

    return fig


def plot_flow_field(
    flow: DeformationVectorField,
    frame: FrameRecord | None = None,
    *,
    fig: go.Figure | None = None,
    arrow_scale: float = 15.0,
    min_magnitude: float = 0.01,
    color: str = "rgba(255,255,255,0.7)",
    show_pitch: bool = True,
    home_team_id: str | None = None,
    home_color: str = "#3b82f6",
    away_color: str = "#ef4444",
    player_names: dict[str, str] | None = None,
    bgcolor: str = "#1a472a",
    line_color: str = "white",
) -> go.Figure:
    """Render a :class:`DeformationVectorField` as quiver arrows on the pitch.

    Each grid cell with magnitude ≥ *min_magnitude* is drawn as a short line
    segment (arrow) starting at the cell centre and pointing in the gradient
    direction.  Arrow length is proportional to magnitude × *arrow_scale*.

    All arrows are packed into a single ``go.Scatter`` trace (segments
    separated by ``None``) for rendering efficiency.

    Parameters:
        flow:          Vector field from :func:`deformation_flow_field`.
        frame:         Optional tracking frame to overlay player positions.
        fig:           Existing figure; creates a new one if ``None``.
        arrow_scale:   Multiplier for arrow length (metres per unit magnitude).
        min_magnitude: Minimum gradient magnitude to draw (skips near-zero cells).
        color:         Arrow colour (default semi-transparent white).
        show_pitch:    Draw pitch background when creating a new figure.
        home_team_id:  Team ID of the home side for player colouring.
        home_color:    Home player marker colour.
        away_color:    Away player marker colour.
        player_names:  Optional player ID → display name mapping.
        bgcolor:       Pitch background colour.
        line_color:    Pitch marking line colour.

    Returns:
        ``go.Figure`` with the flow-field arrows (and optional player overlay).
    """
    if fig is None:
        fig = pitch_background(
            flow.pitch, bgcolor=bgcolor, line_color=line_color,
        ) if show_pitch else go.Figure()

    nx, ny = flow.vectors.shape[:2]
    x_centers = (flow.x_edges[:-1] + flow.x_edges[1:]) / 2.0
    y_centers = (flow.y_edges[:-1] + flow.y_edges[1:]) / 2.0

    # Build arrow segments: start → end with None separator
    arrow_x: list[float | None] = []
    arrow_y: list[float | None] = []

    for i in range(nx):
        for j in range(ny):
            mag = float(flow.magnitudes[i, j])
            if mag < min_magnitude:
                continue
            cx = float(x_centers[i])
            cy = float(y_centers[j])
            vx = float(flow.vectors[i, j, 0]) * arrow_scale
            vy = float(flow.vectors[i, j, 1]) * arrow_scale
            arrow_x += [cx, cx + vx, None]
            arrow_y += [cy, cy + vy, None]

    if arrow_x:
        fig.add_trace(go.Scatter(
            x=arrow_x,
            y=arrow_y,
            mode="lines",
            line=dict(color=color, width=1.5),
            hoverinfo="skip",
            showlegend=False,
            name="drag direction",
        ))

    if frame is not None:
        from pitch_aura.viz.players import plot_players
        fig = plot_players(
            frame,
            fig=fig,
            show_pitch=False,
            home_team_id=home_team_id,
            home_color=home_color,
            away_color=away_color,
            player_names=player_names,
        )

    return fig


def plot_interaction_matrix(
    df: pd.DataFrame,
    *,
    metric_col: str = "total_nsg_m2",
    player_names: dict[str, str] | None = None,
    fig: go.Figure | None = None,
    colorscale: str = "Blues",
    title: str = "Gravity Interaction Matrix",
) -> go.Figure:
    """Render a gravity interaction matrix as an annotated heatmap.

    Pivots the :class:`~pandas.DataFrame` from
    :func:`~pitch_aura.tactics.gravity.gravity_interaction_matrix` into a
    square mover × beneficiary matrix and renders it with
    :class:`plotly.graph_objects.Heatmap`.

    Parameters:
        df:           DataFrame with ``[mover_id, beneficiary_id, <metric>]``
                      columns from :func:`gravity_interaction_matrix`.
        metric_col:   Column to use as cell values (default ``"total_nsg_m2"``).
        player_names: Optional player ID → display name mapping for axis labels.
        fig:          Existing figure; creates a new one if ``None``.
        colorscale:   Plotly colorscale name (default ``"Blues"``).
        title:        Figure title.

    Returns:
        ``go.Figure`` with the interaction matrix heatmap.
    """
    if fig is None:
        fig = go.Figure()

    if df.empty:
        return fig

    def _name(pid: str) -> str:
        return player_names[pid] if player_names and pid in player_names else pid

    # Pivot to square matrix
    pivot = df.pivot(index="mover_id", columns="beneficiary_id", values=metric_col).fillna(0.0)

    mover_labels = [_name(pid) for pid in pivot.index]
    ben_labels = [_name(pid) for pid in pivot.columns]

    z = pivot.values.tolist()

    fig.add_trace(go.Heatmap(
        z=z,
        x=ben_labels,
        y=mover_labels,
        colorscale=colorscale,
        colorbar=dict(title=metric_col, thickness=12),
        hovertemplate=(
            "Mover: %{y}<br>Beneficiary: %{x}<br>"
            + f"{metric_col}: " + "%{z:.1f} m²<extra></extra>"
        ),
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Beneficiary",
        yaxis_title="Mover",
        template="plotly_white",
    )

    return fig
