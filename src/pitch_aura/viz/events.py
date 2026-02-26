"""
Project: PitchAura
File Created: 2026-02-26
Author: Xingnan Zhu
File Name: events.py
Description:
    Event-based visualisation layer for Plotly figures.
    Provides spatial passing network, progressive action arrows,
    and zone heatmap plots.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go

from pitch_aura.types import PitchSpec

from ._pitch_draw import pitch_background

if TYPE_CHECKING:
    import pandas as pd

    from pitch_aura.events.passing_network import PassingNetwork


def _ensure_fig(
    fig: go.Figure | None,
    pitch: PitchSpec | None,
    bgcolor: str,
    line_color: str,
) -> go.Figure:
    """Return *fig* or create a new pitch background."""
    if fig is not None:
        return fig
    p = pitch if pitch is not None else PitchSpec()
    return pitch_background(p, bgcolor=bgcolor, line_color=line_color)


def plot_passing_network(
    network: PassingNetwork,
    *,
    pitch: PitchSpec | None = None,
    fig: go.Figure | None = None,
    node_size_scale: float = 1.0,
    edge_width_scale: float = 1.0,
    color: str = "#3b82f6",
    bgcolor: str = "#1a472a",
    line_color: str = "white",
) -> go.Figure:
    """Visualise a spatial passing network on the pitch.

    Node size scales with pass count, edge width with connection frequency.

    Parameters:
        network:          :class:`~pitch_aura.events.passing_network.PassingNetwork`.
        pitch:            Pitch specification for the background.
        fig:              Existing figure to add traces to.
        node_size_scale:  Multiplier for node marker sizes.
        edge_width_scale: Multiplier for edge line widths.
        color:            Default colour for nodes and edges.
        bgcolor:          Pitch background colour.
        line_color:       Pitch line colour.

    Returns:
        ``go.Figure`` with passing network overlay.
    """
    fig = _ensure_fig(fig, pitch, bgcolor, line_color)

    # Draw edges first (underneath nodes)
    if not network.edges.empty:
        max_count = network.edges["count"].max()
        for _, row in network.edges.iterrows():
            width = max(1.0, (row["count"] / max(max_count, 1)) * 5.0 * edge_width_scale)
            fig.add_trace(go.Scatter(
                x=[row["avg_start_x"], row["avg_end_x"]],
                y=[row["avg_start_y"], row["avg_end_y"]],
                mode="lines",
                line={"color": color, "width": width},
                opacity=0.6,
                showlegend=False,
                hoverinfo="text",
                text=f"{row['passer']} → {row['receiver']} ({int(row['count'])}x)",
            ))

    # Draw nodes
    if not network.nodes.empty:
        max_passes = network.nodes["pass_count"].max()
        sizes = (network.nodes["pass_count"] / max(max_passes, 1) * 20 + 8) * node_size_scale
        fig.add_trace(go.Scatter(
            x=network.nodes["avg_x"],
            y=network.nodes["avg_y"],
            mode="markers+text",
            marker={
                "size": sizes,
                "color": color,
                "line": {"width": 1, "color": "white"},
            },
            text=network.nodes["player_id"],
            textposition="top center",
            textfont={"size": 9, "color": "white"},
            showlegend=False,
            hoverinfo="text",
            hovertext=[
                f"{row['player_id']}: {int(row['pass_count'])} passes"
                for _, row in network.nodes.iterrows()
            ],
        ))

    return fig


def plot_progressive_passes(
    df: pd.DataFrame,
    *,
    pitch: PitchSpec | None = None,
    fig: go.Figure | None = None,
    color_progressive: str = "#22c55e",
    color_other: str = "#94a3b8",
    bgcolor: str = "#1a472a",
    line_color: str = "white",
) -> go.Figure:
    """Visualise progressive and non-progressive actions as arrows.

    Parameters:
        df:                 Output of :func:`~pitch_aura.events.progressive.progressive_actions`.
        pitch:              Pitch specification.
        fig:                Existing figure to add traces to.
        color_progressive:  Colour for progressive actions.
        color_other:        Colour for non-progressive actions.
        bgcolor:            Pitch background colour.
        line_color:         Pitch line colour.

    Returns:
        ``go.Figure`` with arrow traces.
    """
    fig = _ensure_fig(fig, pitch, bgcolor, line_color)

    if df.empty:
        return fig

    for is_prog, group_color, name in [
        (False, color_other, "Non-progressive"),
        (True, color_progressive, "Progressive"),
    ]:
        subset = df[df["is_progressive"] == is_prog]
        if subset.empty:
            continue

        # Draw arrows as lines with annotation arrows
        for _, row in subset.iterrows():
            fig.add_annotation(
                x=row["end_x"],
                y=row["end_y"],
                ax=row["start_x"],
                ay=row["start_y"],
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.2,
                arrowwidth=1.5,
                arrowcolor=group_color,
                opacity=0.7 if is_prog else 0.3,
            )

        # Legend entry via invisible scatter
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker={"color": group_color, "size": 8},
            name=name,
            showlegend=True,
        ))

    return fig


def plot_event_zones(
    df: pd.DataFrame,
    *,
    pitch: PitchSpec | None = None,
    fig: go.Figure | None = None,
    colorscale: str = "YlOrRd",
    bgcolor: str = "#1a472a",
    line_color: str = "white",
) -> go.Figure:
    """Visualise zone event counts as a heatmap overlay.

    Parameters:
        df:          Output of :func:`~pitch_aura.events.zones.zone_counts`.
        pitch:       Pitch specification.
        fig:         Existing figure to add traces to.
        colorscale:  Plotly colorscale name.
        bgcolor:     Pitch background colour.
        line_color:  Pitch line colour.

    Returns:
        ``go.Figure`` with zone heatmap.
    """
    fig = _ensure_fig(fig, pitch, bgcolor, line_color)

    if df.empty:
        return fig

    nx = df["zone_x"].nunique()
    ny = df["zone_y"].nunique()

    # Pivot to 2D matrix
    sorted_df = df.sort_values(["zone_x", "zone_y"])
    z = sorted_df["count"].values.reshape(nx, ny).T  # transpose for heatmap (y rows, x cols)
    x_centers = sorted_df.drop_duplicates("zone_x")["x_center"].values
    y_centers = sorted_df.drop_duplicates("zone_y")["y_center"].values

    fig.add_trace(go.Heatmap(
        x=x_centers,
        y=y_centers,
        z=z,
        colorscale=colorscale,
        opacity=0.6,
        showscale=True,
        colorbar={"title": "Count"},
        hovertemplate="x: %{x:.1f}m<br>y: %{y:.1f}m<br>count: %{z}<extra></extra>",
    ))

    return fig
