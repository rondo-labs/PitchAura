"""
Project: PitchAura
File Created: 2026-02-24
Author: Xingnan Zhu
File Name: animation.py
Description:
    Frame-by-frame animation of FrameSequence tracking data.
    Builds a Plotly figure with go.Frames, a timeline slider, and
    play/pause buttons. Optionally overlays pre-computed ProbabilityGrid
    heatmaps on each frame. The resulting figure can be displayed in
    Jupyter with fig.show() or exported as a shareable HTML file via
    fig.write_html().
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from pitch_aura.types import FrameRecord, FrameSequence, ProbabilityGrid

from ._pitch_draw import _make_pitch_traces, pitch_background


def _player_traces(
    frame: FrameRecord,
    home_team_id: str | None,
    home_color: str,
    away_color: str,
    velocity_scale: float,
) -> list[go.BaseTraceType]:
    """Build player + ball scatter traces for a single animation frame."""
    traces: list[go.BaseTraceType] = []
    if frame.n_players == 0:
        return traces

    positions = frame.positions
    velocities = frame.velocities
    player_ids = frame.player_ids
    team_ids = frame.team_ids
    is_gk = (
        frame.is_goalkeeper
        if len(frame.is_goalkeeper) == frame.n_players
        else np.zeros(frame.n_players, bool)
    )

    for i in range(frame.n_players):
        tid = team_ids[i]
        if home_team_id is None:
            color = home_color
        else:
            color = home_color if tid == home_team_id else away_color
        symbol = "square" if is_gk[i] else "circle"
        pid = player_ids[i]
        x, y = float(positions[i, 0]), float(positions[i, 1])
        traces.append(go.Scatter(
            x=[x], y=[y],
            mode="markers",
            marker=dict(color=color, size=11, symbol=symbol,
                        line=dict(color="white", width=1.5)),
            name=pid,
            hovertemplate=f"{pid}<br>x: {x:.1f}<br>y: {y:.1f}<extra></extra>",
            showlegend=False,
        ))

    # Velocity arrows
    if velocities is not None and velocity_scale > 0:
        arrow_x: list[float | None] = []
        arrow_y: list[float | None] = []
        for i in range(frame.n_players):
            x0, y0 = positions[i]
            vx, vy = velocities[i]
            arrow_x += [float(x0), float(x0 + vx * velocity_scale), None]
            arrow_y += [float(y0), float(y0 + vy * velocity_scale), None]
        traces.append(go.Scatter(
            x=arrow_x, y=arrow_y,
            mode="lines",
            line=dict(color="rgba(255,255,255,0.5)", width=1.2),
            hoverinfo="skip",
            showlegend=False,
        ))

    # Ball
    bx, by = float(frame.ball_position[0]), float(frame.ball_position[1])
    traces.append(go.Scatter(
        x=[bx], y=[by],
        mode="markers",
        marker=dict(color="gold", size=13, symbol="star",
                    line=dict(color="black", width=1)),
        name="Ball",
        hovertemplate=f"Ball<br>x: {bx:.1f}<br>y: {by:.1f}<extra></extra>",
        showlegend=False,
    ))

    return traces


def animate_sequence(
    frames: FrameSequence,
    *,
    grids: list[ProbabilityGrid] | None = None,
    step: int = 1,
    home_team_id: str | None = None,
    home_color: str = "#3b82f6",
    away_color: str = "#ef4444",
    velocity_scale: float = 0.5,
    frame_duration: int = 40,
    transition_duration: int = 0,
    bgcolor: str = "#1a472a",
    line_color: str = "white",
    colorscale: str = "RdBu_r",
    heatmap_opacity: float = 0.7,
) -> go.Figure:
    """Build an animated Plotly figure for a :class:`~pitch_aura.types.FrameSequence`.

    Creates one ``go.Frame`` per tracking frame (after applying *step*
    subsampling).  Each frame contains player positions and optionally a
    pitch control heatmap if *grids* is supplied.

    Parameters:
        frames:              Tracking data to animate.
        grids:               Pre-computed ``ProbabilityGrid`` list — must
                             have the same length as the subsampled frame list
                             (i.e. ``len(frames[::step])``).  If ``None``,
                             only player positions are shown.
        step:                Subsample factor — use every *step*-th frame
                             (default 1 = all frames).
        home_team_id:        Team ID for home side colour.
        home_color:          Marker colour for home players.
        away_color:          Marker colour for away players.
        velocity_scale:      Velocity arrow scale (metres per m/s).
        frame_duration:      Milliseconds per animation frame (default 40 ms
                             ≈ 25 fps real-time).
        transition_duration: Milliseconds for frame transition (default 0 for
                             instant switching).
        bgcolor:             Pitch background colour.
        line_color:          Pitch marking line colour.
        colorscale:          Colorscale for heatmap overlay.
        heatmap_opacity:     Heatmap layer opacity when *grids* is supplied.

    Returns:
        A ``go.Figure`` with animation frames, slider, and play/pause button.
        Call ``fig.show()`` or ``fig.write_html("clip.html")``.

    Raises:
        ValueError: If *grids* is provided but its length does not match the
                    number of subsampled frames.
    """
    selected = frames.frames[::step]
    if not selected:
        return pitch_background(frames.pitch, bgcolor=bgcolor, line_color=line_color)

    if grids is not None and len(grids) != len(selected):
        raise ValueError(
            f"grids length ({len(grids)}) must match subsampled frame count "
            f"({len(selected)})"
        )

    pitch = frames.pitch
    pitch_traces = _make_pitch_traces(pitch, line_color)

    # Build initial frame data
    init_frame = selected[0]
    init_traces = list(pitch_traces)
    if grids is not None:
        g = grids[0]
        init_traces.append(go.Heatmap(
            x=g.x_centers, y=g.y_centers, z=g.values.T,
            colorscale=colorscale, zmin=0.0, zmax=1.0, zmid=0.5,
            opacity=heatmap_opacity, showscale=True,
            colorbar=dict(title="Control", thickness=10, len=0.5),
        ))
    init_traces += _player_traces(init_frame, home_team_id, home_color, away_color, velocity_scale)

    fig = go.Figure(data=init_traces)

    # Build animation frames
    anim_frames = []
    for k, frame in enumerate(selected):
        frame_traces = []
        if grids is not None:
            g = grids[k]
            frame_traces.append(go.Heatmap(
                x=g.x_centers, y=g.y_centers, z=g.values.T,
                colorscale=colorscale, zmin=0.0, zmax=1.0, zmid=0.5,
                opacity=heatmap_opacity, showscale=False,
            ))
        frame_traces += _player_traces(frame, home_team_id, home_color, away_color, velocity_scale)

        anim_frames.append(go.Frame(
            data=frame_traces,
            name=str(k),
            layout=go.Layout(
                title_text=f"t = {frame.timestamp:.2f}s  |  period {frame.period}"
            ),
        ))

    fig.frames = anim_frames

    # Slider steps
    slider_steps = [
        dict(
            args=[[str(k)], dict(frame=dict(duration=frame_duration, redraw=True),
                                  mode="immediate", transition=dict(duration=transition_duration))],
            label=f"{selected[k].timestamp:.1f}s",
            method="animate",
        )
        for k in range(len(selected))
    ]

    x0, x1 = pitch.x_range
    y0, y1 = pitch.y_range

    fig.update_layout(
        plot_bgcolor=bgcolor,
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[x0, x1], showgrid=False, zeroline=False,
                   scaleanchor="y", scaleratio=1, constrain="domain"),
        yaxis=dict(range=[y0, y1], showgrid=False, zeroline=False),
        margin=dict(l=40, r=40, t=60, b=80),
        showlegend=False,
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=0,
            x=0.5,
            xanchor="center",
            yanchor="top",
            buttons=[
                dict(label="▶ Play",
                     method="animate",
                     args=[None, dict(frame=dict(duration=frame_duration, redraw=True),
                                      fromcurrent=True,
                                      transition=dict(duration=transition_duration))]),
                dict(label="⏸ Pause",
                     method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode="immediate",
                                        transition=dict(duration=0))]),
            ],
        )],
        sliders=[dict(
            active=0,
            currentvalue=dict(prefix="Frame: ", visible=True, xanchor="center"),
            pad=dict(b=10, t=10),
            steps=slider_steps,
        )],
        title=dict(
            text=f"t = {selected[0].timestamp:.2f}s  |  period {selected[0].period}",
            x=0.5,
            font=dict(color="white" if bgcolor.startswith("#1") else "black"),
        ),
    )

    return fig
