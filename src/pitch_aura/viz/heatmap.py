"""
Project: PitchAura
File Created: 2026-02-24
Author: Xingnan Zhu
File Name: heatmap.py
Description:
    Heatmap visualisation for ProbabilityGrid.
    Renders the pitch control probability surface as a Plotly Heatmap
    overlaid on a pitch background. Supports customisable colorscales,
    opacity, and composable layering via the fig parameter.
"""

from __future__ import annotations

import plotly.graph_objects as go

from pitch_aura.types import ProbabilityGrid

from ._pitch_draw import pitch_background


def plot_heatmap(
    grid: ProbabilityGrid,
    *,
    fig: go.Figure | None = None,
    colorscale: str = "RdBu_r",
    zmin: float = 0.0,
    zmax: float = 1.0,
    zmid: float = 0.5,
    opacity: float = 0.75,
    show_pitch: bool = True,
    show_colorbar: bool = True,
    bgcolor: str = "#1a472a",
    line_color: str = "white",
) -> go.Figure:
    """Render a :class:`~pitch_aura.types.ProbabilityGrid` as a heatmap.

    Parameters:
        grid:          Pitch control probability grid.
        fig:           Existing figure to add the heatmap to.  If ``None``,
                       a new figure with pitch background is created.
        colorscale:    Plotly colorscale name (default ``"RdBu_r"`` — red for
                       attacking team, blue for defending, white at 0.5).
        zmin:          Minimum colour scale value (default 0.0).
        zmax:          Maximum colour scale value (default 1.0).
        zmid:          Value mapped to the midpoint colour (default 0.5).
        opacity:       Heatmap layer opacity (default 0.75).
        show_pitch:    Draw pitch markings when creating a new figure.
        show_colorbar: Show the colour legend bar.
        bgcolor:       Pitch background colour (used when ``fig=None``).
        line_color:    Pitch line colour (used when ``fig=None``).

    Returns:
        ``go.Figure`` with the heatmap trace added.
    """
    if fig is None:
        fig = pitch_background(
            grid.pitch,
            bgcolor=bgcolor,
            line_color=line_color,
        ) if show_pitch else go.Figure()

    fig.add_trace(go.Heatmap(
        x=grid.x_centers,
        y=grid.y_centers,
        z=grid.values.T,        # Plotly Heatmap: z[row, col] → z[y, x]
        colorscale=colorscale,
        zmin=zmin,
        zmax=zmax,
        zmid=zmid,
        opacity=opacity,
        showscale=show_colorbar,
        colorbar=dict(title="Control", thickness=12, len=0.6),
        hovertemplate="x: %{x:.1f}<br>y: %{y:.1f}<br>P: %{z:.3f}<extra></extra>",
    ))

    return fig
