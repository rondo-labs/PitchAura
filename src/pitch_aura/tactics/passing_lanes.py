"""
Project: PitchAura
File Created: 2026-02-23
Author: Xingnan Zhu
File Name: passing_lanes.py
Description:
    Passing lane analysis: channel obstruction and lifespan.
    Represents a passing lane as a thin rectangle from passer to receiver
    and checks whether any opponent player's position falls inside it,
    then measures how long the lane stays unobstructed in a FrameSequence.
"""

from __future__ import annotations

import numpy as np

from pitch_aura.types import FrameRecord, FrameSequence


def _lane_obstructed(
    frame: FrameRecord,
    passer_id: str,
    receiver_id: str,
    defending_team_id: str,
    lane_width: float,
) -> bool:
    """Return True if any defender centre falls inside the passer-receiver lane.

    The lane is modelled as a rectangle of width *lane_width* centred on the
    segment from passer to receiver.  Obstruction is tested by projecting each
    defender's position onto the passer-receiver axis and checking both the
    along-axis range and the perpendicular offset.

    Parameters:
        frame:             Single tracking frame.
        passer_id:         Player ID of the passer.
        receiver_id:       Player ID of the intended receiver.
        defending_team_id: Team ID of the defending (obstructing) side.
        lane_width:        Width of the lane rectangle in metres.

    Returns:
        ``True`` if at least one defender is inside the lane.
    """
    pid_to_idx = {pid: i for i, pid in enumerate(frame.player_ids)}

    if passer_id not in pid_to_idx or receiver_id not in pid_to_idx:
        return False

    p = frame.positions[pid_to_idx[passer_id]]   # (2,)
    r = frame.positions[pid_to_idx[receiver_id]]  # (2,)

    lane_vec = r - p
    lane_len = float(np.linalg.norm(lane_vec))
    if lane_len < 1e-6:
        return False  # passer and receiver coincide

    axis = lane_vec / lane_len           # unit vector along lane
    perp = np.array([-axis[1], axis[0]]) # perpendicular unit vector

    half_w = lane_width / 2.0

    for i, tid in enumerate(frame.team_ids):
        if tid != defending_team_id:
            continue
        pid = frame.player_ids[i]
        if pid in (passer_id, receiver_id):
            continue
        d = frame.positions[i] - p
        along = float(np.dot(d, axis))
        lateral = abs(float(np.dot(d, perp)))
        # Defender must be between passer and receiver and within lane width
        if 0.0 <= along <= lane_len and lateral <= half_w:
            return True

    return False


def passing_lane_lifespan(
    frames: FrameSequence,
    *,
    passer_id: str,
    receiver_id: str,
    lane_width: float = 2.0,
) -> float:
    """Duration (seconds) that a passing lane remains continuously unobstructed.

    Identifies the defending team automatically as whichever team contains
    neither *passer_id* nor *receiver_id*.  If both players are on the same
    team, the opposite team is used as defenders.

    The measurement starts from the first frame in *frames* where both players
    are present and counts consecutive unobstructed frames.  The counter resets
    to zero on the first obstructed frame.

    Parameters:
        frames:      Tracking data window to analyse.
        passer_id:   Player ID of the passer.
        receiver_id: Player ID of the intended receiver.
        lane_width:  Width of the lane rectangle in metres (default 2 m).

    Returns:
        Total unobstructed duration in seconds (sum over all unobstructed
        frames, not just the initial run), or ``0.0`` if no valid frames found.

    Raises:
        ValueError: If *passer_id* or *receiver_id* is not found in any frame.
    """
    if not frames.frames:
        return 0.0

    dt = 1.0 / frames.frame_rate

    # Resolve defending team
    passer_team: str | None = None
    receiver_team: str | None = None
    for frame in frames.frames:
        pid_map = {pid: tid for pid, tid in zip(frame.player_ids, frame.team_ids)}
        if passer_id in pid_map:
            passer_team = pid_map[passer_id]
        if receiver_id in pid_map:
            receiver_team = pid_map[receiver_id]
        if passer_team is not None and receiver_team is not None:
            break

    if passer_team is None:
        raise ValueError(f"passer_id {passer_id!r} not found in any frame")
    if receiver_team is None:
        raise ValueError(f"receiver_id {receiver_id!r} not found in any frame")

    # Defending team = the team that is not the passer's team (use away/home logic)
    all_team_ids = {frames.home_team_id, frames.away_team_id}
    other_teams = all_team_ids - {passer_team}
    defending_team_id = other_teams.pop() if other_teams else receiver_team

    unobstructed_duration = 0.0
    for frame in frames.frames:
        if passer_id not in frame.player_ids or receiver_id not in frame.player_ids:
            continue
        if not _lane_obstructed(frame, passer_id, receiver_id, defending_team_id, lane_width):
            unobstructed_duration += dt

    return unobstructed_duration
