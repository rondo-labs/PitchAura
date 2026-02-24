"""
Project: PitchAura
File Created: 2026-02-23
Author: Xingnan Zhu
File Name: alignment.py
Description:
    Temporal alignment between tracking data and event data.
    Produces one FrameRecord per event by finding the nearest tracking
    frame or by linearly interpolating between bounding frames.
"""

from __future__ import annotations

import numpy as np

from pitch_aura.types import EventRecord, FrameRecord, FrameSequence


def _interpolate_frame(
    frame_a: FrameRecord,
    frame_b: FrameRecord,
    t: float,
) -> FrameRecord:
    """Return a new FrameRecord at time ``t`` by linearly interpolating
    between ``frame_a`` (t_a <= t) and ``frame_b`` (t_b >= t).

    Players present in both frames get interpolated positions.
    Players present in only one frame retain that frame's position.
    """
    ta, tb = frame_a.timestamp, frame_b.timestamp
    alpha = (t - ta) / (tb - ta) if tb != ta else 0.0

    # Build a unified player list; prefer frame_a ordering
    pid_to_idx_a: dict[str, int] = {p: i for i, p in enumerate(frame_a.player_ids)}
    pid_to_idx_b: dict[str, int] = {p: i for i, p in enumerate(frame_b.player_ids)}

    all_pids = list(pid_to_idx_a.keys())
    for pid in frame_b.player_ids:
        if pid not in pid_to_idx_a:
            all_pids.append(pid)

    positions: list[np.ndarray] = []
    velocities_list: list[np.ndarray] = []
    team_ids: list[str] = []
    gk_flags: list[bool] = []
    has_vel = frame_a.velocities is not None and frame_b.velocities is not None

    for pid in all_pids:
        in_a = pid in pid_to_idx_a
        in_b = pid in pid_to_idx_b

        if in_a and in_b:
            ia, ib = pid_to_idx_a[pid], pid_to_idx_b[pid]
            pos = (1.0 - alpha) * frame_a.positions[ia] + alpha * frame_b.positions[ib]
            if has_vel:
                vel = (1.0 - alpha) * frame_a.velocities[ia] + alpha * frame_b.velocities[ib]  # type: ignore[index]
            else:
                vel = np.zeros(2)
            team_id = frame_a.team_ids[ia]
            gk = bool(frame_a.is_goalkeeper[ia])
        elif in_a:
            ia = pid_to_idx_a[pid]
            pos = frame_a.positions[ia].copy()
            vel = frame_a.velocities[ia].copy() if frame_a.velocities is not None else np.zeros(2)
            team_id = frame_a.team_ids[ia]
            gk = bool(frame_a.is_goalkeeper[ia])
        else:
            ib = pid_to_idx_b[pid]
            pos = frame_b.positions[ib].copy()
            vel = frame_b.velocities[ib].copy() if frame_b.velocities is not None else np.zeros(2)
            team_id = frame_b.team_ids[ib]
            gk = bool(frame_b.is_goalkeeper[ib])

        positions.append(pos)
        velocities_list.append(vel)
        team_ids.append(team_id)
        gk_flags.append(gk)

    # Interpolate ball position
    ball = (1.0 - alpha) * frame_a.ball_position + alpha * frame_b.ball_position

    return FrameRecord(
        timestamp=t,
        period=frame_a.period,
        ball_position=ball,
        player_ids=all_pids,
        team_ids=team_ids,
        positions=np.array(positions, dtype=np.float64),
        velocities=np.array(velocities_list, dtype=np.float64) if has_vel else None,
        is_goalkeeper=np.array(gk_flags, dtype=bool),
    )


def align(
    frames: FrameSequence,
    events: list[EventRecord],
    *,
    method: str = "nearest",
) -> FrameSequence:
    """Align tracking frames to event timestamps.

    Produces one :class:`FrameRecord` per event in the same period,
    either by picking the nearest existing frame or by linearly
    interpolating between the two bounding frames.

    Parameters:
        frames: Source tracking data (may span multiple periods).
        events: Event records to align against.
        method: ``"nearest"`` or ``"interpolate"``.

    Returns:
        A new :class:`FrameSequence` with one frame per event, ordered
        by event timestamp. Events outside the tracking time range are
        clamped to the first or last available frame.

    Raises:
        ValueError: If ``method`` is not ``"nearest"`` or ``"interpolate"``.
    """
    if method not in ("nearest", "interpolate"):
        raise ValueError(f"method must be 'nearest' or 'interpolate', got {method!r}")

    if not events or not frames.frames:
        return FrameSequence(
            frames=[],
            frame_rate=frames.frame_rate,
            pitch=frames.pitch,
            home_team_id=frames.home_team_id,
            away_team_id=frames.away_team_id,
        )

    timestamps = frames.timestamps  # (T,)
    result: list[FrameRecord] = []

    for event in events:
        t = event.timestamp

        if method == "nearest":
            idx = int(np.argmin(np.abs(timestamps - t)))
            result.append(frames.frames[idx])

        else:  # interpolate
            # Find the index of the last frame with timestamp <= t
            idx_right = int(np.searchsorted(timestamps, t, side="right"))
            idx_left = idx_right - 1

            if idx_right == 0:
                # Before the first frame — clamp
                result.append(frames.frames[0])
            elif idx_right >= len(frames.frames):
                # After the last frame — clamp
                result.append(frames.frames[-1])
            else:
                result.append(
                    _interpolate_frame(frames.frames[idx_left], frames.frames[idx_right], t)
                )

    return FrameSequence(
        frames=result,
        frame_rate=frames.frame_rate,
        pitch=frames.pitch,
        home_team_id=frames.home_team_id,
        away_team_id=frames.away_team_id,
    )
