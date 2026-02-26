"""
Project: PitchAura
File Created: 2026-02-26
Author: Xingnan Zhu
File Name: progressive.py
Description:
    Progressive distance metrics for event data.
    Measures how far passes, carries, and other actions advance the ball
    toward the opponent's goal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pitch_aura.types import EventRecord, PitchSpec


def progressive_actions(
    events: list[EventRecord],
    *,
    pitch: PitchSpec,
    event_types: tuple[str, ...] = ("pass", "carry"),
    target_x: float | None = None,
    min_distance: float = 0.0,
) -> pd.DataFrame:
    """Compute progressive distance for actions with start and end coordinates.

    An action is *progressive* when it moves the ball closer to the target
    goal by at least ``max(min_distance, 0.25 * total_distance)`` meters.

    Parameters:
        events:       List of event records.
        pitch:        Pitch specification (used to locate goal line).
        event_types:  Event types to include (matched case-insensitively).
        target_x:     X-coordinate of the target goal line.  If ``None``,
                      defaults to the far end of the pitch
                      (``pitch.x_range[1]``).
        min_distance: Absolute minimum forward distance to qualify as
                      progressive (meters).

    Returns:
        DataFrame with columns ``timestamp``, ``period``, ``event_type``,
        ``player_id``, ``team_id``, ``start_x``, ``start_y``, ``end_x``,
        ``end_y``, ``distance``, ``progressive_distance``,
        ``is_progressive``.
    """
    cols = [
        "timestamp", "period", "event_type", "player_id", "team_id",
        "start_x", "start_y", "end_x", "end_y",
        "distance", "progressive_distance", "is_progressive",
    ]

    if not events:
        return pd.DataFrame(columns=cols)

    if target_x is None:
        target_x = pitch.x_range[1]

    target = np.array([target_x, pitch.y_range[0] + pitch.width / 2])

    allowed = {et.lower() for et in event_types}

    rows: list[dict] = []
    for ev in events:
        if ev.event_type.lower() not in allowed:
            continue
        if ev.coordinates is None or ev.end_coordinates is None:
            continue

        start = ev.coordinates
        end = ev.end_coordinates
        total_dist = float(np.linalg.norm(end - start))

        dist_start = float(np.linalg.norm(target - start))
        dist_end = float(np.linalg.norm(target - end))
        prog_dist = dist_start - dist_end

        threshold = max(min_distance, 0.25 * total_dist)
        is_prog = prog_dist >= threshold

        rows.append({
            "timestamp": ev.timestamp,
            "period": ev.period,
            "event_type": ev.event_type,
            "player_id": ev.player_id,
            "team_id": ev.team_id,
            "start_x": float(start[0]),
            "start_y": float(start[1]),
            "end_x": float(end[0]),
            "end_y": float(end[1]),
            "distance": total_dist,
            "progressive_distance": prog_dist,
            "is_progressive": is_prog,
        })

    if not rows:
        return pd.DataFrame(columns=cols)

    return pd.DataFrame(rows, columns=cols)
