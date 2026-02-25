"""
Project: PitchAura
File Created: 2026-02-25
Author: Xingnan Zhu
File Name: gravity.py
Description:
    Spatial gravity and deformation analysis via counterfactual pitch control.
    Quantifies how off-ball movement distorts defensive spatial structure by
    comparing actual pitch control against a hypothetical where the runner
    stays put.  Key metrics and features:

    - **SDI (Spatial Drag Index)**: total area of defensive control dissolved
      by a player's movement (m²), plus efficiency normalization.
    - **NSG (Net Space Generated)**: weighted area of attacking control gained
      near a teammate due to the mover's run (m²).
    - **gravity_profile**: match-level aggregation of SDI across a full sequence.
    - **deformation_recovery**: how quickly defensive structure recovers after
      peak deformation (half-life, recovery rate).
    - **DeformationVectorField / deformation_flow_field**: spatial gradient of
      the deformation field, showing the direction space was "dragged".
    - **gravity_interaction_matrix**: N×N who-creates-space-for-whom matrix
      across all attacking player pairs.
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


@dataclass(frozen=True, slots=True)
class DeformationVectorField:
    """Spatial gradient of a deformation field.

    Represents the direction and magnitude of spatial flow at each grid cell —
    i.e., the direction defenders were "dragged" by a player's movement.

    ``vectors[i, j]`` is the 2D gradient vector (∂delta/∂x, ∂delta/∂y) at
    cell ``(i, j)``, pointing from low→high deformation.
    """

    vectors: np.ndarray
    """Flow vectors, shape ``(nx, ny, 2)``.  vectors[i, j] = (dx, dy)."""
    magnitudes: np.ndarray
    """Gradient magnitudes, shape ``(nx, ny)``."""
    x_edges: np.ndarray
    """Cell boundaries along x-axis, shape ``(nx + 1,)``."""
    y_edges: np.ndarray
    """Cell boundaries along y-axis, shape ``(ny + 1,)``."""
    pitch: PitchSpec
    timestamp: float
    player_id: str


@dataclass(frozen=True, slots=True)
class RecoveryMetrics:
    """Defensive recovery statistics derived from an SDI timeseries.

    Describes how quickly the defensive structure recovers after peak
    deformation caused by a player's off-ball run.
    """

    peak_sdi_m2: float
    """Maximum SDI value observed in the timeseries (m²)."""
    peak_timestamp: float
    """Timestamp at which peak SDI occurred (s)."""
    half_life_s: float | None
    """Time after peak for SDI to drop to 50 % of peak (s).
    ``None`` if recovery never reaches the threshold."""
    recovery_rate_m2_per_s: float
    """Mean rate of SDI decrease post-peak (m²/s, always ≥ 0)."""
    full_recovery_s: float | None
    """Time after peak for SDI to drop below 10 % of peak (s).
    ``None`` if full recovery is not observed."""


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
        DataFrame with columns
        ``[timestamp, sdi_m2, displacement_m, sdi_efficiency]``.
        ``sdi_efficiency`` is SDI normalised by displacement (m²/m) and
        measures how much space is created per metre of running.
        If *return_deformation* is ``True``, returns a tuple of
        ``(DataFrame, list[DeformationGrid])``.

    Raises:
        ValueError: If *player_id* is not found in any frame.
    """
    empty_df = pd.DataFrame(columns=["timestamp", "sdi_m2", "displacement_m", "sdi_efficiency"])

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

    disp_arr = np.array(displacements)
    sdi_arr = np.array(sdi_values)
    efficiency = sdi_arr / np.maximum(disp_arr, 1e-6)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "sdi_m2": sdi_arr,
        "displacement_m": disp_arr,
        "sdi_efficiency": efficiency,
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


def gravity_profile(
    frames: FrameSequence,
    *,
    player_id: str,
    attacking_team_id: str,
    resolution: tuple[int, int] = DEFAULT_GRID_RESOLUTION,
    control_model: object | None = None,
    pitch: PitchSpec | None = None,
    min_displacement: float = 2.0,
) -> dict:
    """Aggregate gravity metrics for a player across an entire sequence.

    Computes SDI across all frames (no time-window truncation) and returns
    summary statistics useful for cross-player comparison or match-level
    reporting.

    Parameters:
        frames:            Tracking data sequence.
        player_id:         Player to analyse.
        attacking_team_id: Attacking team ID passed to the control model.
        resolution:        Grid resolution ``(nx, ny)``.
        control_model:     Custom model (default :class:`KinematicControlModel`).
        pitch:             Pitch specification; inferred from *frames* if omitted.
        min_displacement:  Displacement threshold (m) for counting a frame as
                           "significantly off starting position".

    Returns:
        Dict with keys:

        * ``total_sdi_m2``          — sum of SDI across all frames (m²).
        * ``peak_sdi_m2``           — maximum SDI in any single frame (m²).
        * ``mean_sdi_efficiency``   — mean SDI-per-metre ratio (m²/m).
        * ``total_displacement_m``  — maximum displacement from starting
          position observed across the sequence (m).
        * ``n_significant_frames``  — number of frames where displacement
          ≥ *min_displacement*.
    """
    empty: dict = {
        "total_sdi_m2": 0.0,
        "peak_sdi_m2": 0.0,
        "mean_sdi_efficiency": 0.0,
        "total_displacement_m": 0.0,
        "n_significant_frames": 0,
    }
    if not frames.frames:
        return empty

    # Cover all frames by using a very large time window
    t0 = frames.frames[0].timestamp
    t_last = frames.frames[-1].timestamp
    full_window = max(t_last - t0 + 1.0, 1.0)

    df = spatial_drag_index(
        frames,
        player_id=player_id,
        attacking_team_id=attacking_team_id,
        time_window=full_window,
        resolution=resolution,
        control_model=control_model,
        pitch=pitch,
    )

    if df.empty:
        return empty

    return {
        "total_sdi_m2": float(df["sdi_m2"].sum()),
        "peak_sdi_m2": float(df["sdi_m2"].max()),
        "mean_sdi_efficiency": float(df["sdi_efficiency"].mean()),
        "total_displacement_m": float(df["displacement_m"].max()),
        "n_significant_frames": int((df["displacement_m"] >= min_displacement).sum()),
    }


def deformation_recovery(
    sdi_df: pd.DataFrame,
    *,
    peak_threshold: float = 0.5,
    full_threshold: float = 0.1,
) -> RecoveryMetrics:
    """Compute defensive recovery metrics from an SDI timeseries.

    Pure post-processing on the :class:`~pandas.DataFrame` returned by
    :func:`spatial_drag_index`.  Locates the peak SDI, then measures how
    quickly the value decays afterward.

    Parameters:
        sdi_df:          DataFrame with ``[timestamp, sdi_m2]`` columns, as
                         returned by :func:`spatial_drag_index`.
        peak_threshold:  Fraction of peak SDI defining the "half-life" point
                         (default 0.5 → 50 %).
        full_threshold:  Fraction of peak SDI defining "full recovery"
                         (default 0.1 → 10 %).

    Returns:
        :class:`RecoveryMetrics` with peak info, half-life, recovery rate,
        and full-recovery time.  ``half_life_s`` and ``full_recovery_s`` are
        ``None`` if the threshold is never reached within the window.

    Raises:
        ValueError: If *sdi_df* is empty or missing required columns.
    """
    required = {"timestamp", "sdi_m2"}
    if sdi_df.empty or not required.issubset(sdi_df.columns):
        raise ValueError(
            f"sdi_df must be non-empty and contain columns {required}"
        )

    peak_idx = int(sdi_df["sdi_m2"].idxmax())
    peak_sdi = float(sdi_df.loc[peak_idx, "sdi_m2"])
    peak_ts = float(sdi_df.loc[peak_idx, "timestamp"])

    # Post-peak slice
    post = sdi_df.iloc[peak_idx:]
    post_ts = post["timestamp"].to_numpy()
    post_sdi = post["sdi_m2"].to_numpy()

    # Half-life: first timestamp where SDI < peak_threshold * peak_sdi
    half_level = peak_threshold * peak_sdi
    half_life_s: float | None = None
    below_half = np.where(post_sdi < half_level)[0]
    if len(below_half) > 0:
        half_life_s = float(post_ts[below_half[0]]) - peak_ts

    # Full recovery: first timestamp where SDI < full_threshold * peak_sdi
    full_level = full_threshold * peak_sdi
    full_recovery_s: float | None = None
    below_full = np.where(post_sdi < full_level)[0]
    if len(below_full) > 0:
        full_recovery_s = float(post_ts[below_full[0]]) - peak_ts

    # Recovery rate: mean decrease rate post-peak (m²/s)
    if len(post_ts) > 1:
        dt = float(post_ts[-1] - post_ts[0])
        dsdi = float(post_sdi[0] - post_sdi[-1])  # positive = decreasing
        recovery_rate = max(0.0, dsdi / max(dt, 1e-9))
    else:
        recovery_rate = 0.0

    return RecoveryMetrics(
        peak_sdi_m2=peak_sdi,
        peak_timestamp=peak_ts,
        half_life_s=half_life_s,
        recovery_rate_m2_per_s=recovery_rate,
        full_recovery_s=full_recovery_s,
    )


def deformation_flow_field(
    deformation: DeformationGrid,
) -> DeformationVectorField:
    """Compute the spatial gradient of a deformation field.

    The gradient ∇(delta) at each cell points in the direction of steepest
    increase in the deformation surface — i.e., the direction defenders were
    "pulled away from" by the attacker's movement.

    Uses :func:`numpy.gradient` (central differences at interior points,
    forward/backward at boundaries).

    Parameters:
        deformation: A :class:`DeformationGrid` from :func:`spatial_drag_index`.

    Returns:
        :class:`DeformationVectorField` with shape ``(nx, ny, 2)`` vectors
        and ``(nx, ny)`` magnitudes.
    """
    dx = float(deformation.x_edges[1] - deformation.x_edges[0])
    dy = float(deformation.y_edges[1] - deformation.y_edges[0])

    # np.gradient returns [∂f/∂axis0, ∂f/∂axis1] for a 2D array
    grad_x, grad_y = np.gradient(deformation.values, dx, dy)  # each (nx, ny)

    vectors = np.stack([grad_x, grad_y], axis=-1)          # (nx, ny, 2)
    magnitudes = np.sqrt(grad_x ** 2 + grad_y ** 2)        # (nx, ny)

    return DeformationVectorField(
        vectors=vectors,
        magnitudes=magnitudes,
        x_edges=deformation.x_edges.copy(),
        y_edges=deformation.y_edges.copy(),
        pitch=deformation.pitch,
        timestamp=deformation.timestamp,
        player_id=deformation.player_id,
    )


def gravity_interaction_matrix(
    frames: FrameSequence,
    *,
    attacking_team_id: str,
    time_window: float = 5.0,
    resolution: tuple[int, int] = DEFAULT_GRID_RESOLUTION,
    control_model: object | None = None,
    pitch: PitchSpec | None = None,
    zone_weights: np.ndarray | None = None,
) -> pd.DataFrame:
    """Build an N×N space-creation interaction matrix for a team.

    For each attacking player acting as *mover*, computes how much positive
    deformation (NSG) their movement generates across all other attacking
    players acting as *beneficiaries*.

    The deformation grid is computed once per (mover, frame), then the same
    gain array is accumulated for every beneficiary present in that frame.
    This is O(N) model evaluations per frame instead of O(N²).

    When *zone_weights* is ``None`` (default), each beneficiary receives the
    total positive deformation created by the mover in that frame — useful for
    ranking who creates the most space overall.  Pass a custom weight array to
    scope the calculation to a particular zone.

    Parameters:
        frames:            Tracking data sequence.
        attacking_team_id: Team ID of the attacking side.
        time_window:       Analysis window in seconds.
        resolution:        Grid resolution ``(nx, ny)``.
        control_model:     Custom model (default :class:`KinematicControlModel`).
        pitch:             Pitch specification; inferred from *frames* if omitted.
        zone_weights:      Optional ``(nx, ny)`` weight array applied to the
                           gain surface before summation.

    Returns:
        DataFrame with columns
        ``[mover_id, beneficiary_id, total_nsg_m2, peak_nsg_m2, mean_nsg_m2]``.
        Returns an empty DataFrame if fewer than 2 attacking players are found.
    """
    empty_df = pd.DataFrame(
        columns=["mover_id", "beneficiary_id", "total_nsg_m2", "peak_nsg_m2", "mean_nsg_m2"]
    )

    if not frames.frames:
        return empty_df

    resolved_pitch = pitch if pitch is not None else frames.pitch
    model = _resolve_model(control_model, resolved_pitch, resolution)

    t0 = frames.frames[0].timestamp
    t_end = t0 + time_window
    window_frames = [f for f in frames.frames if f.timestamp <= t_end]

    if not window_frames:
        return empty_df

    # Collect all attacking player IDs seen in the window
    attacking_pids: list[str] = []
    for frame in window_frames:
        for pid, tid in zip(frame.player_ids, frame.team_ids):
            if tid == attacking_team_id and pid not in attacking_pids:
                attacking_pids.append(pid)

    if len(attacking_pids) < 2:
        return empty_df

    # (mover_id, beneficiary_id) -> list[float] of per-frame NSG values
    nsg_records: dict[tuple[str, str], list[float]] = {}

    for mover_id in attacking_pids:
        # Frozen position from first appearance
        frozen_pos: np.ndarray | None = None
        for frame in window_frames:
            if mover_id in frame.player_ids:
                idx = frame.player_ids.index(mover_id)
                frozen_pos = frame.positions[idx].copy()
                break

        if frozen_pos is None:
            continue

        for frame in window_frames:
            if mover_id not in frame.player_ids:
                continue

            # Compute deformation once for this (mover, frame) pair
            actual_grid: ProbabilityGrid = model.control(frame, team_id=attacking_team_id)  # type: ignore[union-attr]
            cf_frame = _counterfactual_frame(frame, mover_id, frozen_pos)
            cf_grid: ProbabilityGrid = model.control(cf_frame, team_id=attacking_team_id)  # type: ignore[union-attr]

            delta = actual_grid.values - cf_grid.values
            gain = np.maximum(0.0, delta)

            if zone_weights is not None:
                gain = gain * zone_weights

            frame_nsg = float(np.sum(gain) * actual_grid.cell_area)

            # Record same NSG value for every beneficiary present in this frame
            for beneficiary_id in attacking_pids:
                if beneficiary_id == mover_id:
                    continue
                if beneficiary_id not in frame.player_ids:
                    continue

                key = (mover_id, beneficiary_id)
                if key not in nsg_records:
                    nsg_records[key] = []
                nsg_records[key].append(frame_nsg)

    if not nsg_records:
        return empty_df

    rows = []
    for (mover_id, beneficiary_id), nsg_list in nsg_records.items():
        arr = np.array(nsg_list)
        rows.append({
            "mover_id": mover_id,
            "beneficiary_id": beneficiary_id,
            "total_nsg_m2": float(arr.sum()),
            "peak_nsg_m2": float(arr.max()),
            "mean_nsg_m2": float(arr.mean()),
        })

    return pd.DataFrame(rows)
