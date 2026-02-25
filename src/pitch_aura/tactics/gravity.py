"""
Project: PitchAura
File Created: 2026-02-25
Author: Xingnan Zhu
File Name: gravity.py
Description:
    Spatial gravity and deformation analysis via counterfactual pitch control.
    Quantifies how off-ball movement distorts defensive spatial structure by
    comparing actual pitch control against a hypothetical where the runner
    stays put.  Two key metrics:

    - **SDI (Spatial Drag Index)**: total area of defensive control dissolved
      by a player's movement (m²).
    - **NSG (Net Space Generated)**: weighted area of attacking control gained
      near a teammate due to the mover's run (m²).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pitch_aura._grid import make_grid
from pitch_aura.constants import DEFAULT_GRID_RESOLUTION
from pitch_aura.space.kinematic import KinematicControlModel
from pitch_aura.types import FrameRecord, FrameSequence, PitchSpec, ProbabilityGrid


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class DeformationGrid:
    """Spatial deformation field produced by counterfactual analysis.

    ``values[i, j]`` is the change in attacking-team control probability at
    grid cell ``(i, j)`` caused by a player's movement.  Positive values mean
    the attacker gained control; negative means the attacker lost control
    (defender freed up).
    """

    values: np.ndarray
    """Delta in control probability, shape ``(nx, ny)``, range ``(-1, 1)``."""
    x_edges: np.ndarray
    """Cell boundaries along x-axis, shape ``(nx + 1,)``."""
    y_edges: np.ndarray
    """Cell boundaries along y-axis, shape ``(ny + 1,)``."""
    pitch: PitchSpec
    timestamp: float
    player_id: str

    @property
    def cell_area(self) -> float:
        """Area of a single cell in m²."""
        dx = float(self.x_edges[1] - self.x_edges[0])
        dy = float(self.y_edges[1] - self.y_edges[0])
        return dx * dy


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _counterfactual_frame(
    frame: FrameRecord,
    player_id: str,
    frozen_position: np.ndarray,
) -> FrameRecord:
    """Create a copy of *frame* with *player_id* frozen at *frozen_position*.

    The player's velocity is zeroed (stationary in the counterfactual).
    All arrays are copied so the original frame is never mutated.

    Raises:
        ValueError: If *player_id* is not present in *frame*.
    """
    if player_id not in frame.player_ids:
        raise ValueError(
            f"player_id {player_id!r} not found in frame at t={frame.timestamp}"
        )

    idx = frame.player_ids.index(player_id)

    new_positions = frame.positions.copy()
    new_positions[idx] = frozen_position

    new_velocities: np.ndarray | None = None
    if frame.velocities is not None:
        new_velocities = frame.velocities.copy()
        new_velocities[idx] = 0.0

    return FrameRecord(
        timestamp=frame.timestamp,
        period=frame.period,
        ball_position=frame.ball_position.copy(),
        player_ids=list(frame.player_ids),
        team_ids=list(frame.team_ids),
        positions=new_positions,
        velocities=new_velocities,
        is_goalkeeper=frame.is_goalkeeper.copy() if frame.is_goalkeeper.size else frame.is_goalkeeper,
    )


def _resolve_model(
    control_model: object | None,
    pitch: PitchSpec,
    resolution: tuple[int, int],
) -> object:
    """Return *control_model* or build a default ``KinematicControlModel``."""
    if control_model is not None:
        return control_model
    return KinematicControlModel(pitch=pitch, resolution=resolution)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def spatial_drag_index(
    frames: FrameSequence,
    *,
    player_id: str,
    attacking_team_id: str,
    time_window: float = 5.0,
    resolution: tuple[int, int] = DEFAULT_GRID_RESOLUTION,
    control_model: object | None = None,
    pitch: PitchSpec | None = None,
    return_deformation: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, list[DeformationGrid]]:
    """Compute the Spatial Drag Index over a time window.

    For each frame after the first, the SDI measures how much defensive
    control area was dissolved by the player's movement compared to a
    counterfactual where the player stayed at their initial position.

    Parameters:
        frames:            Tracking data sequence.
        player_id:         ID of the player whose movement to evaluate.
        attacking_team_id: Team ID of the attacking side (used as
                           ``team_id`` when computing pitch control).
        time_window:       Analysis window in seconds from first frame.
        resolution:        Grid resolution ``(nx, ny)``.
        control_model:     Custom model with
                           ``control(frame, team_id=...) -> ProbabilityGrid``.
                           Defaults to :class:`KinematicControlModel`.
        pitch:             Pitch specification; inferred from *frames* if omitted.
        return_deformation: If ``True``, also return a list of
                            :class:`DeformationGrid` objects for visualisation.

    Returns:
        DataFrame with columns ``[timestamp, sdi_m2, displacement_m]``.
        If *return_deformation* is ``True``, returns a tuple of
        ``(DataFrame, list[DeformationGrid])``.

    Raises:
        ValueError: If *player_id* is not found in any frame.
    """
    empty_df = pd.DataFrame(columns=["timestamp", "sdi_m2", "displacement_m"])

    if not frames.frames:
        return (empty_df, []) if return_deformation else empty_df

    # Validate player exists
    all_pids = {pid for f in frames.frames for pid in f.player_ids}
    if player_id not in all_pids:
        raise ValueError(f"player_id {player_id!r} not found in any frame")

    # Time window filter
    t0 = frames.frames[0].timestamp
    t_end = t0 + time_window
    window_frames = [f for f in frames.frames if f.timestamp <= t_end]

    if not window_frames:
        return (empty_df, []) if return_deformation else empty_df

    # Resolve model
    resolved_pitch = pitch if pitch is not None else frames.pitch
    model = _resolve_model(control_model, resolved_pitch, resolution)

    # Find frozen position from first frame where player appears
    frozen_pos: np.ndarray | None = None
    for f in window_frames:
        if player_id in f.player_ids:
            idx = f.player_ids.index(player_id)
            frozen_pos = f.positions[idx].copy()
            break

    if frozen_pos is None:
        return (empty_df, []) if return_deformation else empty_df

    timestamps: list[float] = []
    sdi_values: list[float] = []
    displacements: list[float] = []
    deformations: list[DeformationGrid] = []

    for frame in window_frames:
        if player_id not in frame.player_ids:
            continue

        idx = frame.player_ids.index(player_id)
        current_pos = frame.positions[idx]
        displacement = float(np.linalg.norm(current_pos - frozen_pos))

        # Actual pitch control
        actual_grid: ProbabilityGrid = model.control(frame, team_id=attacking_team_id)  # type: ignore[union-attr]

        # Counterfactual: player frozen at t0 position
        cf_frame = _counterfactual_frame(frame, player_id, frozen_pos)
        cf_grid: ProbabilityGrid = model.control(cf_frame, team_id=attacking_team_id)  # type: ignore[union-attr]

        # Delta: positive = attacker gained control due to movement
        delta = actual_grid.values - cf_grid.values

        # SDI = total area where attacker gained control (defender dragged away)
        sdi = float(np.sum(np.maximum(0.0, delta)) * actual_grid.cell_area)

        timestamps.append(frame.timestamp)
        sdi_values.append(sdi)
        displacements.append(displacement)

        if return_deformation:
            deformations.append(DeformationGrid(
                values=delta,
                x_edges=actual_grid.x_edges,
                y_edges=actual_grid.y_edges,
                pitch=resolved_pitch,
                timestamp=frame.timestamp,
                player_id=player_id,
            ))

    if not timestamps:
        return (empty_df, []) if return_deformation else empty_df

    df = pd.DataFrame({
        "timestamp": timestamps,
        "sdi_m2": sdi_values,
        "displacement_m": displacements,
    })

    return (df, deformations) if return_deformation else df


def net_space_generated(
    frames: FrameSequence,
    *,
    mover_id: str,
    beneficiary_id: str,
    attacking_team_id: str,
    time_window: float = 5.0,
    zone_weights: np.ndarray | None = None,
    resolution: tuple[int, int] = DEFAULT_GRID_RESOLUTION,
    control_model: object | None = None,
    pitch: PitchSpec | None = None,
) -> pd.DataFrame:
    """Compute Net Space Generated for a beneficiary due to the mover's run.

    Measures how much attacking control increases across the pitch (or in
    weighted zones) because the mover ran instead of staying put.

    Parameters:
        frames:            Tracking data sequence.
        mover_id:          Player whose movement causes the deformation.
        beneficiary_id:    Teammate whose spatial gain we measure.
        attacking_team_id: Team ID of the attacking side.
        time_window:       Analysis window in seconds.
        zone_weights:      Optional weight array, shape ``(nx, ny)``.  Each
                           cell's control gain is multiplied by its weight
                           before summation.  ``None`` means uniform weight.
                           Use :func:`penalty_zone_weights` for a common
                           danger-zone mask.
        resolution:        Grid resolution ``(nx, ny)``.
        control_model:     Custom model (default :class:`KinematicControlModel`).
        pitch:             Pitch specification; inferred from *frames* if omitted.

    Returns:
        DataFrame with columns
        ``[timestamp, nsg_m2, beneficiary_x, beneficiary_y]``.

    Raises:
        ValueError: If *mover_id* or *beneficiary_id* is not found in any frame.
    """
    empty_df = pd.DataFrame(
        columns=["timestamp", "nsg_m2", "beneficiary_x", "beneficiary_y"]
    )

    if not frames.frames:
        return empty_df

    # Validate both players exist
    all_pids = {pid for f in frames.frames for pid in f.player_ids}
    if mover_id not in all_pids:
        raise ValueError(f"mover_id {mover_id!r} not found in any frame")
    if beneficiary_id not in all_pids:
        raise ValueError(f"beneficiary_id {beneficiary_id!r} not found in any frame")

    # Time window filter
    t0 = frames.frames[0].timestamp
    t_end = t0 + time_window
    window_frames = [f for f in frames.frames if f.timestamp <= t_end]

    if not window_frames:
        return empty_df

    # Resolve model
    resolved_pitch = pitch if pitch is not None else frames.pitch
    model = _resolve_model(control_model, resolved_pitch, resolution)

    # Frozen position of mover
    frozen_pos: np.ndarray | None = None
    for f in window_frames:
        if mover_id in f.player_ids:
            idx = f.player_ids.index(mover_id)
            frozen_pos = f.positions[idx].copy()
            break

    if frozen_pos is None:
        return empty_df

    timestamps: list[float] = []
    nsg_values: list[float] = []
    ben_xs: list[float] = []
    ben_ys: list[float] = []

    for frame in window_frames:
        if mover_id not in frame.player_ids:
            continue
        if beneficiary_id not in frame.player_ids:
            continue

        ben_idx = frame.player_ids.index(beneficiary_id)
        ben_pos = frame.positions[ben_idx]

        # Actual vs counterfactual pitch control
        actual_grid: ProbabilityGrid = model.control(frame, team_id=attacking_team_id)  # type: ignore[union-attr]
        cf_frame = _counterfactual_frame(frame, mover_id, frozen_pos)
        cf_grid: ProbabilityGrid = model.control(cf_frame, team_id=attacking_team_id)  # type: ignore[union-attr]

        delta = actual_grid.values - cf_grid.values
        gain = np.maximum(0.0, delta)

        if zone_weights is not None:
            gain = gain * zone_weights

        nsg = float(np.sum(gain) * actual_grid.cell_area)

        timestamps.append(frame.timestamp)
        nsg_values.append(nsg)
        ben_xs.append(float(ben_pos[0]))
        ben_ys.append(float(ben_pos[1]))

    if not timestamps:
        return empty_df

    return pd.DataFrame({
        "timestamp": timestamps,
        "nsg_m2": nsg_values,
        "beneficiary_x": ben_xs,
        "beneficiary_y": ben_ys,
    })


def penalty_zone_weights(
    pitch: PitchSpec | None = None,
    resolution: tuple[int, int] = DEFAULT_GRID_RESOLUTION,
    *,
    side: str = "right",
) -> np.ndarray:
    """Build a binary weight mask for the penalty area.

    Returns an ``(nx, ny)`` array with 1.0 inside the specified penalty
    area and 0.0 elsewhere.  The penalty area extends 16.5 m from the
    goal line and 20.15 m either side of centre (40.3 m total width).

    Parameters:
        pitch:      Pitch specification.  Defaults to standard 105×68 m.
        resolution: Grid resolution ``(nx, ny)``.
        side:       ``"right"`` for the right-hand goal (default) or
                    ``"left"`` for the left-hand goal.

    Returns:
        Weight array of shape ``(nx, ny)`` with binary values.

    Raises:
        ValueError: If *side* is not ``"right"`` or ``"left"``.
    """
    if side not in ("right", "left"):
        raise ValueError(f"side must be 'right' or 'left', got {side!r}")

    resolved_pitch = pitch if pitch is not None else PitchSpec()
    targets, x_edges, y_edges = make_grid(resolved_pitch, resolution)

    nx, ny = resolution
    x = targets[:, 0].reshape(nx, ny)
    y = targets[:, 1].reshape(nx, ny)

    x_min, x_max = resolved_pitch.x_range
    y_min, y_max = resolved_pitch.y_range
    y_mid = (y_min + y_max) / 2.0

    # Penalty area: 16.5m deep, 40.32m wide (20.16m each side of centre)
    pa_depth = 16.5
    pa_half_width = 20.16

    if side == "right":
        x_mask = x >= (x_max - pa_depth)
    else:
        x_mask = x <= (x_min + pa_depth)

    y_mask = np.abs(y - y_mid) <= pa_half_width

    weights = np.where(x_mask & y_mask, 1.0, 0.0)
    return weights
