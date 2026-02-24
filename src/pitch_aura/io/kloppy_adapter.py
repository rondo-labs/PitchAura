"""
Project: PitchAura
File Created: 2026-02-23
Author: Xingnan Zhu
File Name: kloppy_adapter.py
Description:
    Adapter for converting kloppy data models to pitch_aura types.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pitch_aura.types import EventRecord, FrameRecord, FrameSequence, PitchSpec

if TYPE_CHECKING:
    from kloppy import EventDataset, TrackingDataset
    from kloppy.domain.models.tracking import Frame, PlayerData


def _is_goalkeeper(player: object) -> bool:
    """Check if a kloppy Player has a goalkeeper position."""
    from kloppy.domain.models.position import PositionType

    pos = getattr(player, "starting_position", None)
    if pos is not None:
        return pos == PositionType.Goalkeeper
    return False


def _extract_frame(
    frame: Frame,
    pitch_length: float,
    pitch_width: float,
) -> FrameRecord | None:
    """Convert a single kloppy Frame to a FrameRecord.

    Returns None if the frame has no player data.
    """
    if not frame.players_data:
        return None

    player_ids: list[str] = []
    team_ids: list[str] = []
    positions: list[list[float]] = []
    speeds: list[float | None] = []
    gk_flags: list[bool] = []

    for player, pdata in frame.players_data.items():
        if pdata.coordinates is None:
            continue

        player_ids.append(str(player.player_id))
        team_ids.append(str(player.team.team_id))
        positions.append([
            pdata.coordinates.x * pitch_length,
            pdata.coordinates.y * pitch_width,
        ])
        speeds.append(pdata.speed)
        gk_flags.append(_is_goalkeeper(player))

    if not positions:
        return None

    # Ball coordinates
    ball_pos: np.ndarray
    if frame.ball_coordinates is not None:
        bx = frame.ball_coordinates.x * pitch_length
        by = frame.ball_coordinates.y * pitch_width
        bz = getattr(frame.ball_coordinates, "z", None)
        if bz is not None:
            ball_pos = np.array([bx, by, bz], dtype=np.float64)
        else:
            ball_pos = np.array([bx, by], dtype=np.float64)
    else:
        ball_pos = np.array([np.nan, np.nan], dtype=np.float64)

    return FrameRecord(
        timestamp=frame.timestamp.total_seconds(),
        period=frame.period.id,
        ball_position=ball_pos,
        player_ids=player_ids,
        team_ids=team_ids,
        positions=np.array(positions, dtype=np.float64),
        velocities=None,  # computed later via finite differences
        is_goalkeeper=np.array(gk_flags, dtype=bool),
    )


def _compute_velocities(
    frames: list[FrameRecord],
    frame_rate: float,
) -> None:
    """Compute velocities via finite differences and set them in-place.

    Uses central differences for interior frames and forward/backward
    differences for the first/last frames.
    """
    n = len(frames)
    if n < 2:
        return

    dt = 1.0 / frame_rate

    # Build a mapping from player_id to position per frame for alignment
    # Since player ordering may differ between frames, we align by player_id
    for i in range(n):
        curr = frames[i]
        n_players = curr.n_players
        vel = np.zeros_like(curr.positions)

        for p_idx in range(n_players):
            pid = curr.player_ids[p_idx]

            # Find same player in adjacent frames
            pos_prev: np.ndarray | None = None
            pos_next: np.ndarray | None = None

            if i > 0:
                prev_frame = frames[i - 1]
                try:
                    j = prev_frame.player_ids.index(pid)
                    pos_prev = prev_frame.positions[j]
                except ValueError:
                    pass

            if i < n - 1:
                next_frame = frames[i + 1]
                try:
                    j = next_frame.player_ids.index(pid)
                    pos_next = next_frame.positions[j]
                except ValueError:
                    pass

            if pos_prev is not None and pos_next is not None:
                # Central difference
                vel[p_idx] = (pos_next - pos_prev) / (2.0 * dt)
            elif pos_next is not None:
                # Forward difference
                vel[p_idx] = (pos_next - curr.positions[p_idx]) / dt
            elif pos_prev is not None:
                # Backward difference
                vel[p_idx] = (curr.positions[p_idx] - pos_prev) / dt
            # else: leave as zero

        curr.velocities = vel


def from_tracking(dataset: TrackingDataset) -> FrameSequence:
    """Convert a kloppy ``TrackingDataset`` to a :class:`FrameSequence`.

    Coordinates are scaled from the kloppy coordinate system to meters
    using the dataset's pitch dimensions. Velocities are computed via
    finite differences if not directly available.

    Parameters:
        dataset: A ``kloppy.TrackingDataset`` instance.

    Returns:
        Converted frame sequence with contiguous array layout.
    """
    meta = dataset.metadata
    pitch_dims = meta.pitch_dimensions

    pitch_length = pitch_dims.pitch_length or 105.0
    pitch_width = pitch_dims.pitch_width or 68.0

    # Determine team IDs
    teams = meta.teams
    home_team_id = str(teams[0].team_id) if teams else "home"
    away_team_id = str(teams[1].team_id) if len(teams) > 1 else "away"

    # Convert frames
    records: list[FrameRecord] = []
    for frame in dataset.records:
        rec = _extract_frame(frame, pitch_length, pitch_width)
        if rec is not None:
            records.append(rec)

    # Compute velocities via finite differences
    frame_rate = meta.frame_rate or 25.0
    _compute_velocities(records, frame_rate)

    return FrameSequence(
        frames=records,
        frame_rate=frame_rate,
        pitch=PitchSpec(
            length=pitch_length,
            width=pitch_width,
            origin="bottom_left",
        ),
        home_team_id=home_team_id,
        away_team_id=away_team_id,
    )


def from_events(dataset: EventDataset) -> list[EventRecord]:
    """Convert a kloppy ``EventDataset`` to a list of :class:`EventRecord`.

    Parameters:
        dataset: A ``kloppy.EventDataset`` instance.

    Returns:
        List of event records ordered by timestamp.
    """
    meta = dataset.metadata
    pitch_dims = meta.pitch_dimensions
    pitch_length = pitch_dims.pitch_length or 105.0
    pitch_width = pitch_dims.pitch_width or 68.0

    records: list[EventRecord] = []
    for event in dataset.records:
        coords: np.ndarray | None = None
        if event.coordinates is not None:
            coords = np.array(
                [
                    event.coordinates.x * pitch_length,
                    event.coordinates.y * pitch_width,
                ],
                dtype=np.float64,
            )

        records.append(
            EventRecord(
                timestamp=event.timestamp.total_seconds(),
                period=event.period.id,
                event_type=event.event_name,
                player_id=str(event.player.player_id) if event.player else None,
                team_id=str(event.team.team_id) if event.team else None,
                coordinates=coords,
            )
        )

    return records
