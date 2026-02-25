"""
PitchAura visualisation layer.

All functions return ``plotly.graph_objects.Figure`` and never call
``fig.show()`` — the caller decides how to display or export the result.

Low-level composable functions (accept ``fig`` parameter for layering)::

    from pitch_aura.viz._pitch_draw import pitch_background
    from pitch_aura.viz.heatmap import plot_heatmap
    from pitch_aura.viz.players import plot_players
    from pitch_aura.viz.voronoi import plot_voronoi
    from pitch_aura.viz.tactics import plot_pockets, plot_passing_lane
    from pitch_aura.viz.animation import animate_sequence

High-level one-shot convenience functions (re-exported here)::

    from pitch_aura.viz import plot_pitch_control, plot_voronoi_control

Requires the ``viz`` optional dependency group::

    pip install pitch-aura[viz]
"""

from __future__ import annotations

from pitch_aura.viz._pitch_draw import pitch_background
from pitch_aura.viz.animation import animate_sequence
from pitch_aura.viz.heatmap import plot_heatmap
from pitch_aura.viz.players import plot_players
from pitch_aura.viz.tactics import (
    plot_deformation_field,
    plot_gravity_timeseries,
    plot_passing_lane,
    plot_pockets,
)
from pitch_aura.viz.voronoi import plot_voronoi

try:
    import plotly.graph_objects as go
    _PLOTLY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PLOTLY_AVAILABLE = False


def plot_pitch_control(
    grid: object,
    frame: object | None = None,
    *,
    home_team_id: str | None = None,
    home_color: str = "#3b82f6",
    away_color: str = "#ef4444",
    colorscale: str = "RdBu_r",
    heatmap_opacity: float = 0.75,
    show_velocity: bool = True,
    bgcolor: str = "#1a472a",
    line_color: str = "white",
) -> "go.Figure":
    """One-shot pitch control visualisation: heatmap + optional player overlay.

    Parameters:
        grid:             :class:`~pitch_aura.types.ProbabilityGrid` to render.
        frame:            Optional :class:`~pitch_aura.types.FrameRecord` to
                          overlay player positions on top of the heatmap.
        home_team_id:     Team ID of the home side (for player colouring).
        home_color:       Home player marker colour.
        away_color:       Away player marker colour.
        colorscale:       Heatmap colorscale (default ``"RdBu_r"``).
        heatmap_opacity:  Heatmap layer opacity.
        show_velocity:    Show velocity arrows when *frame* is given.
        bgcolor:          Pitch background colour.
        line_color:       Pitch marking line colour.

    Returns:
        ``go.Figure`` ready for ``fig.show()`` or ``fig.write_html()``.
    """
    from pitch_aura.types import ProbabilityGrid, FrameRecord
    assert isinstance(grid, ProbabilityGrid)

    fig = plot_heatmap(
        grid,
        colorscale=colorscale,
        opacity=heatmap_opacity,
        bgcolor=bgcolor,
        line_color=line_color,
    )
    if frame is not None:
        assert isinstance(frame, FrameRecord)
        fig = plot_players(
            frame, fig=fig,
            home_team_id=home_team_id,
            home_color=home_color,
            away_color=away_color,
            show_velocity=show_velocity,
            show_pitch=False,
        )
    return fig


def plot_voronoi_control(
    result: object,
    frame: object,
    *,
    home_team_id: str | None = None,
    home_color: str = "#3b82f6",
    away_color: str = "#ef4444",
    opacity: float = 0.3,
    show_areas: bool = False,
    bgcolor: str = "#1a472a",
    line_color: str = "white",
) -> "go.Figure":
    """One-shot Voronoi control visualisation: regions + player overlay.

    Parameters:
        result:       :class:`~pitch_aura.types.VoronoiResult` to render.
        frame:        :class:`~pitch_aura.types.FrameRecord` for player positions
                      and team membership lookup.
        home_team_id: Team ID of the home side.
        home_color:   Home team fill colour.
        away_color:   Away team fill colour.
        opacity:      Polygon fill opacity.
        show_areas:   Annotate regions with area labels.
        bgcolor:      Pitch background colour.
        line_color:   Pitch marking line colour.

    Returns:
        ``go.Figure`` with Voronoi regions and player markers.
    """
    from pitch_aura.types import VoronoiResult, FrameRecord
    assert isinstance(result, VoronoiResult)
    assert isinstance(frame, FrameRecord)

    return plot_voronoi(
        result, frame,
        home_team_id=home_team_id,
        home_color=home_color,
        away_color=away_color,
        opacity=opacity,
        show_areas=show_areas,
        bgcolor=bgcolor,
        line_color=line_color,
    )


__all__ = [
    # Low-level
    "pitch_background",
    "plot_heatmap",
    "plot_players",
    "plot_voronoi",
    "plot_pockets",
    "plot_passing_lane",
    "plot_deformation_field",
    "plot_gravity_timeseries",
    "animate_sequence",
    # High-level
    "plot_pitch_control",
    "plot_voronoi_control",
]
