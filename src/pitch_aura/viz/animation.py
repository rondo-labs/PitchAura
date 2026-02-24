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


# Fixed number of dynamic traces per frame (must match _player_traces output).
# Order: home-outfield, home-GK, away-outfield, away-GK, velocity, ball.
_N_DYNAMIC = 6


def _player_traces(
    frame: FrameRecord,
    home_team_id: str | None,
    home_color: str,
    away_color: str,
    velocity_scale: float,
) -> list[go.BaseTraceType]:
    """Return exactly ``_N_DYNAMIC`` traces for one animation frame.

    Using a fixed trace count lets ``go.Frame`` reference the same indices
    every frame via ``traces=``, preventing stale "ghost" copies of the
    initial frame from persisting during playback.

    Trace slots (always present, x/y may be empty):
        0: home outfield players
        1: home GK
        2: away outfield players
        3: away GK
        4: velocity arrows
        5: ball
    """
    n = frame.n_players
    positions = frame.positions
    velocities = frame.velocities
    player_ids = frame.player_ids
    team_ids = frame.team_ids
    is_gk = (
        frame.is_goalkeeper
        if len(frame.is_goalkeeper) == n
        else np.zeros(n, bool)
    )

    traces: list[go.BaseTraceType] = []

    # 4 player slots: (is_home_side, is_goalkeeper_slot, color)
    slots = [
        (True,  False, home_color),
        (True,  True,  home_color),
        (False, False, away_color),
        (False, True,  away_color),
    ]
    for is_home_side, gk_slot, color in slots:
        if n == 0:
            sel = np.zeros(0, bool)
        elif home_team_id is None:
            team_mask = np.ones(n, bool)
            sel = team_mask & (is_gk == gk_slot)
        else:
            team_mask = np.array([t == home_team_id for t in team_ids], bool)
            if not is_home_side:
                team_mask = ~team_mask
            sel = team_mask & (is_gk == gk_slot)

        xs = positions[sel, 0].tolist() if sel.any() else []
        ys = positions[sel, 1].tolist() if sel.any() else []
        pids = [player_ids[i] for i in range(n) if sel[i]]
        symbol = "square" if gk_slot else "circle"

        traces.append(go.Scatter(
            x=xs, y=ys,
            mode="markers",
            marker=dict(color=color, size=11, symbol=symbol,
                        line=dict(color="white", width=1.5)),
            customdata=pids if pids else None,
            hovertemplate=(
                "%{customdata}<br>x: %{x:.1f}<br>y: %{y:.1f}<extra></extra>"
                if pids else None
            ),
            showlegend=False,
        ))

    # Velocity arrows (slot 4)
    arrow_x: list[float | None] = []
    arrow_y: list[float | None] = []
    if velocities is not None and velocity_scale > 0 and n > 0:
        for i in range(n):
            x0, y0 = float(positions[i, 0]), float(positions[i, 1])
            vx, vy = float(velocities[i, 0]), float(velocities[i, 1])
            arrow_x += [x0, x0 + vx * velocity_scale, None]
            arrow_y += [y0, y0 + vy * velocity_scale, None]
    traces.append(go.Scatter(
        x=arrow_x or [None], y=arrow_y or [None],
        mode="lines",
        line=dict(color="rgba(255,255,255,0.5)", width=1.2),
        hoverinfo="skip",
        showlegend=False,
    ))

    # Ball (slot 5)
    bx = float(frame.ball_position[0])
    by = float(frame.ball_position[1])
    traces.append(go.Scatter(
        x=[bx], y=[by],
        mode="markers",
        marker=dict(color="gold", size=13, symbol="star",
                    line=dict(color="black", width=1)),
        hovertemplate=f"Ball<br>x: {bx:.1f}<br>y: {by:.1f}<extra></extra>",
        showlegend=False,
    ))

    return traces  # always exactly _N_DYNAMIC traces


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
    n_pitch = len(pitch_traces)

    # Build initial figure data: static pitch + optional heatmap + dynamic players
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

    # Indices of the dynamic traces inside fig.data.
    # The heatmap (if present) sits at n_pitch; player/ball traces follow.
    heatmap_idx = n_pitch if grids is not None else None
    dyn_start = n_pitch + (1 if grids is not None else 0)
    dyn_indices = list(range(dyn_start, dyn_start + _N_DYNAMIC))

    # Build animation frames
    anim_frames = []
    for k, frame in enumerate(selected):
        frame_data: list[go.BaseTraceType] = []
        frame_trace_indices: list[int] = []

        if grids is not None:
            g = grids[k]
            frame_data.append(go.Heatmap(
                x=g.x_centers, y=g.y_centers, z=g.values.T,
                colorscale=colorscale, zmin=0.0, zmax=1.0, zmid=0.5,
                opacity=heatmap_opacity, showscale=False,
            ))
            frame_trace_indices.append(heatmap_idx)  # type: ignore[arg-type]

        frame_data += _player_traces(frame, home_team_id, home_color, away_color, velocity_scale)
        frame_trace_indices += dyn_indices

        anim_frames.append(go.Frame(
            data=frame_data,
            traces=frame_trace_indices,
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
