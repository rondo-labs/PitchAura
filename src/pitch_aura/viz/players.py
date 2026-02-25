"""
Project: PitchAura
File Created: 2026-02-24
Author: Xingnan Zhu
File Name: players.py
Description:
    Player and ball position layer for Plotly figures.
    Renders player positions as team-coloured scatter markers with optional
    velocity arrows and player ID labels. Goalkeepers use a distinct square
    marker. The ball is drawn as a gold star. Velocity vectors are drawn as
    thin line segments scaled by velocity_scale.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from pitch_aura.types import FrameRecord, PitchSpec

from ._pitch_draw import pitch_background


def plot_players(
    frame: FrameRecord,
    *,
    fig: go.Figure | None = None,
    pitch: PitchSpec | None = None,
    home_team_id: str | None = None,
    home_color: str = "#3b82f6",
    away_color: str = "#ef4444",
    show_velocity: bool = True,
    velocity_scale: float = 0.5,
    show_ball: bool = True,
    show_labels: bool = False,
    player_names: dict[str, str] | None = None,
    show_pitch: bool = True,
    bgcolor: str = "#1a472a",
    line_color: str = "white",
) -> go.Figure:
    """Render player positions and optionally velocity vectors on a pitch.

    Parameters:
        frame:          Tracking frame to visualise.
        fig:            Existing figure to add traces to.  Creates a new
                        figure with pitch background if ``None``.
        pitch:          Pitch spec for the background (inferred from frame
                        positions if ``None`` and no ``fig`` is given; falls
                        back to a default 105×68 m pitch).
        home_team_id:   Team ID of the home side; determines which team gets
                        *home_color*.  If ``None``, all players use
                        *home_color*.
        home_color:     Marker colour for home players (default blue).
        away_color:     Marker colour for away players (default red).
        show_velocity:  Draw velocity arrows as line segments.
        velocity_scale: Scaling factor for velocity vectors (metres per m/s).
                        E.g. 0.5 means a 10 m/s velocity draws as a 5 m line.
        show_ball:      Draw the ball position as a gold star.
        show_labels:    Annotate each player with their ID (or name if
                        *player_names* is provided).
        player_names:   Optional mapping from player_id to display name.
                        When provided, names are used in hover tooltips and
                        labels instead of raw IDs.
        show_pitch:     Create pitch background when ``fig=None``.
        bgcolor:        Background colour (used when ``fig=None``).
        line_color:     Pitch line colour (used when ``fig=None``).

    Returns:
        ``go.Figure`` with player and (optionally) ball traces added.
    """
    from pitch_aura.types import PitchSpec as _PitchSpec
    resolved_pitch = pitch if pitch is not None else _PitchSpec()

    if fig is None:
        fig = pitch_background(
            resolved_pitch, bgcolor=bgcolor, line_color=line_color,
        ) if show_pitch else go.Figure()

    if frame.n_players == 0:
        return fig

    positions = frame.positions          # (N, 2)
    velocities = frame.velocities        # (N, 2) or None
    player_ids = frame.player_ids
    team_ids = frame.team_ids
    is_gk = frame.is_goalkeeper if len(frame.is_goalkeeper) == frame.n_players else np.zeros(frame.n_players, bool)

    # Group players by team for single trace per team (better legend)
    teams_seen: dict[str, list[int]] = {}
    for i, tid in enumerate(team_ids):
        teams_seen.setdefault(tid, []).append(i)

    for tid, indices in teams_seen.items():
        if home_team_id is None:
            color = home_color
        else:
            color = home_color if tid == home_team_id else away_color

        idx_arr = np.array(indices)
        pos = positions[idx_arr]            # (M, 2)
        gk_flags = is_gk[idx_arr]          # (M,)

        # Split outfield vs GK for different symbols
        for gk_val, symbol in [(False, "circle"), (True, "square")]:
            sel = gk_flags == gk_val
            if not sel.any():
                continue
            sub_idx = idx_arr[sel]
            sub_pos = positions[sub_idx]
            sub_pids = [player_ids[i] for i in sub_idx]
            label = ("GK " if gk_val else "") + tid

            def _display(pid: str) -> str:
                return player_names[pid] if player_names and pid in player_names else pid

            hover = [
                f"{_display(pid)}<br>x: {sub_pos[j,0]:.1f}<br>y: {sub_pos[j,1]:.1f}"
                for j, pid in enumerate(sub_pids)
            ]
            display_labels = [_display(pid) for pid in sub_pids]

            fig.add_trace(go.Scatter(
                x=sub_pos[:, 0],
                y=sub_pos[:, 1],
                mode="markers+text" if show_labels else "markers",
                marker=dict(
                    color=color,
                    size=12,
                    symbol=symbol,
                    line=dict(color="white", width=1.5),
                ),
                text=display_labels if show_labels else None,
                textposition="top center",
                textfont=dict(size=9, color="white"),
                name=label,
                hovertext=hover,
                hoverinfo="text",
                legendgroup=tid,
            ))

    # Velocity arrows as thin line segments
    if show_velocity and velocities is not None and velocity_scale > 0:
        arrow_x: list[float | None] = []
        arrow_y: list[float | None] = []
        for i in range(frame.n_players):
            x0, y0 = positions[i]
            vx, vy = velocities[i]
            x1 = x0 + vx * velocity_scale
            y1 = y0 + vy * velocity_scale
            arrow_x += [x0, x1, None]
            arrow_y += [y0, y1, None]

        fig.add_trace(go.Scatter(
            x=arrow_x,
            y=arrow_y,
            mode="lines",
            line=dict(color="rgba(255,255,255,0.6)", width=1.5),
            hoverinfo="skip",
            showlegend=False,
            name="velocity",
        ))

    # Ball
    if show_ball:
        bx, by = float(frame.ball_position[0]), float(frame.ball_position[1])
        fig.add_trace(go.Scatter(
            x=[bx], y=[by],
            mode="markers",
            marker=dict(color="gold", size=14, symbol="star",
                        line=dict(color="black", width=1)),
            name="Ball",
            hovertemplate=f"Ball<br>x: {bx:.1f}<br>y: {by:.1f}<extra></extra>",
        ))

    return fig
