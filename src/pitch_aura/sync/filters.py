"""
Project: PitchAura
File Created: 2026-02-23
Author: Xingnan Zhu
File Name: filters.py
Description:
    Signal smoothing filters for tracking data noise reduction.
    Provides moving-average and constant-velocity Kalman filters,
    both operating on per-player position time series extracted from
    a FrameSequence.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d

from pitch_aura.types import FrameRecord, FrameSequence


# ---------------------------------------------------------------------------
# Internal: per-player series extraction / reassembly
# ---------------------------------------------------------------------------

def _extract_player_series(
    frames: list[FrameRecord],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Extract per-player position time series from a frame list.

    Parameters:
        frames: Ordered list of :class:`FrameRecord`.

    Returns:
        Tuple of ``(player_ids, positions, mask)`` where:

        - ``player_ids``: sorted list of all unique player IDs.
        - ``positions``:  shape ``(T, P, 2)`` — position per frame per player.
          ``NaN`` where a player is absent.
        - ``mask``:       shape ``(T, P)`` bool — True where player is present.
    """
    all_pids: list[str] = sorted({pid for f in frames for pid in f.player_ids})
    pid_index = {pid: i for i, pid in enumerate(all_pids)}
    T = len(frames)
    P = len(all_pids)

    positions = np.full((T, P, 2), np.nan)
    mask = np.zeros((T, P), dtype=bool)

    for t, frame in enumerate(frames):
        for i, pid in enumerate(frame.player_ids):
            p = pid_index[pid]
            positions[t, p] = frame.positions[i]
            mask[t, p] = True

    return all_pids, positions, mask


# ---------------------------------------------------------------------------
# Smoothing algorithms
# ---------------------------------------------------------------------------

def _moving_average(
    positions: np.ndarray,
    mask: np.ndarray,
    window: int,
) -> np.ndarray:
    """Apply a uniform moving average along the time axis.

    NaN positions (absent players) are forward-filled before filtering
    and masked back to NaN afterward.

    Parameters:
        positions: ``(T, P, 2)`` array.
        mask:      ``(T, P)`` boolean presence mask.
        window:    Window size (must be odd for symmetric averaging).

    Returns:
        Smoothed ``(T, P, 2)`` array with NaN where the player was absent.
    """
    T, P, _ = positions.shape
    filled = positions.copy()

    # Forward-fill NaN per player per coordinate
    for p in range(P):
        for c in range(2):
            series = filled[:, p, c]
            nan_mask = np.isnan(series)
            if nan_mask.all():
                continue
            # ffill
            valid_idx = np.where(~nan_mask)[0]
            for i in range(T):
                if nan_mask[i]:
                    # nearest valid before
                    before = valid_idx[valid_idx < i]
                    if len(before):
                        filled[i, p, c] = filled[before[-1], p, c]
                    else:
                        # use nearest valid after
                        after = valid_idx[valid_idx > i]
                        if len(after):
                            filled[i, p, c] = filled[after[0], p, c]

    smoothed = uniform_filter1d(filled, size=window, axis=0, mode="nearest")

    # Restore NaN for absent frames
    absent = ~mask  # (T, P)
    smoothed[absent] = np.nan

    return smoothed


def _kalman_filter(
    positions: np.ndarray,
    mask: np.ndarray,
    dt: float,
    process_noise: float,
    measurement_noise: float,
) -> np.ndarray:
    """Constant-velocity Kalman filter on per-player position series.

    State vector: ``[x, y, vx, vy]``. Operates independently for each
    player using forward-pass filtering only (no RTS smoother).

    Parameters:
        positions:         ``(T, P, 2)`` position array.
        mask:              ``(T, P)`` boolean presence mask.
        dt:                Frame interval in seconds.
        process_noise:     Process noise covariance scalar ``q``.
        measurement_noise: Measurement noise covariance scalar ``r``.

    Returns:
        Smoothed ``(T, P, 2)`` array; NaN where the player was absent.
    """
    T, P, _ = positions.shape
    smoothed = np.full_like(positions, np.nan)

    # Constant-velocity state transition
    F = np.array([
        [1.0, 0.0, dt,  0.0],
        [0.0, 1.0, 0.0, dt ],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    H = np.array([[1.0, 0.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, 0.0]])
    Q = process_noise * np.eye(4)
    R = measurement_noise * np.eye(2)

    for p in range(P):
        # Find first valid observation to seed the filter
        first_valid = np.where(mask[:, p])[0]
        if len(first_valid) == 0:
            continue
        t0 = first_valid[0]

        x = np.zeros(4)
        x[:2] = positions[t0, p]
        P_cov = np.eye(4) * 1.0

        for t in range(t0, T):
            # Predict
            x = F @ x
            P_cov = F @ P_cov @ F.T + Q

            # Update (only when player is observed)
            if mask[t, p] and not np.any(np.isnan(positions[t, p])):
                z = positions[t, p]
                S = H @ P_cov @ H.T + R
                K = P_cov @ H.T @ np.linalg.inv(S)
                x = x + K @ (z - H @ x)
                P_cov = (np.eye(4) - K @ H) @ P_cov

            if mask[t, p]:
                smoothed[t, p] = x[:2]

    return smoothed


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def smooth(
    frames: FrameSequence,
    *,
    method: str = "moving_average",
    **kwargs: float,
) -> FrameSequence:
    """Smooth noisy tracking data coordinates.

    Applies the chosen filter independently to each player's position
    time series. Velocities in the output are recomputed via finite
    differences on the smoothed positions.

    Parameters:
        frames: Raw tracking data.
        method: ``"moving_average"`` or ``"kalman"``.
        **kwargs:
            For ``"moving_average"``: ``window`` (int, default 5) —
            number of frames to average over.

            For ``"kalman"``: ``process_noise`` (float, default 1.0) and
            ``measurement_noise`` (float, default 0.5) — covariance scalars
            for the constant-velocity state-space model.

    Returns:
        A new :class:`FrameSequence` with smoothed positions and
        recomputed velocities.

    Raises:
        ValueError: If ``method`` is unrecognised.
    """
    if method not in ("moving_average", "kalman"):
        raise ValueError(f"method must be 'moving_average' or 'kalman', got {method!r}")

    if not frames.frames:
        return FrameSequence(
            frames=[],
            frame_rate=frames.frame_rate,
            pitch=frames.pitch,
            home_team_id=frames.home_team_id,
            away_team_id=frames.away_team_id,
        )

    all_pids, positions, mask = _extract_player_series(frames.frames)
    dt = 1.0 / frames.frame_rate

    if method == "moving_average":
        window = int(kwargs.get("window", 5))
        smoothed = _moving_average(positions, mask, window)
    else:  # kalman
        q = float(kwargs.get("process_noise", 1.0))
        r = float(kwargs.get("measurement_noise", 0.5))
        smoothed = _kalman_filter(positions, mask, dt, q, r)

    new_frames = _rebuild_frames_with_dt(frames.frames, all_pids, smoothed, mask, dt)

    return FrameSequence(
        frames=new_frames,
        frame_rate=frames.frame_rate,
        pitch=frames.pitch,
        home_team_id=frames.home_team_id,
        away_team_id=frames.away_team_id,
    )


def _rebuild_frames_with_dt(
    frames: list[FrameRecord],
    all_pids: list[str],
    smoothed: np.ndarray,
    mask: np.ndarray,
    dt: float,
) -> list[FrameRecord]:
    """Like _rebuild_frames but uses the correct dt for velocity recomputation."""
    pid_index = {pid: i for i, pid in enumerate(all_pids)}
    T = len(frames)

    vel_series = np.zeros_like(smoothed)
    if T >= 2:
        vel_series[1:-1] = (smoothed[2:] - smoothed[:-2]) / (2.0 * dt)
        vel_series[0] = (smoothed[1] - smoothed[0]) / dt
        vel_series[-1] = (smoothed[-1] - smoothed[-2]) / dt

    result: list[FrameRecord] = []
    for t, frame in enumerate(frames):
        pids_here = frame.player_ids
        indices = [pid_index[pid] for pid in pids_here]
        valid = mask[t, indices]

        new_pos = smoothed[t, indices]
        new_vel = vel_series[t, indices]
        gk = frame.is_goalkeeper if len(frame.is_goalkeeper) == len(pids_here) else np.zeros(len(pids_here), bool)

        result.append(FrameRecord(
            timestamp=frame.timestamp,
            period=frame.period,
            ball_position=frame.ball_position.copy(),
            player_ids=[pid for pid, v in zip(pids_here, valid) if v],
            team_ids=[tid for tid, v in zip(frame.team_ids, valid) if v],
            positions=new_pos[valid],
            velocities=new_vel[valid],
            is_goalkeeper=gk[valid],
        ))

    return result
