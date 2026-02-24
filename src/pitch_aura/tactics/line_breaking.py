"""
Project: PitchAura
File Created: 2026-02-23
Author: Xingnan Zhu
File Name: line_breaking.py
Description:
    Defensive line pocket detection.
    Groups defending players into horizontal lines by clustering their
    depth (x) coordinates, then identifies gaps between adjacent players
    within each line that exceed a minimum width. Returns detected pockets
    as lightweight namedtuples (no ProbabilityGrid dependency).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pitch_aura.types import FrameRecord


@dataclass(frozen=True, slots=True)
class Pocket:
    """A gap in a defensive line through which a ball or run could pass.

    Attributes:
        line_depth:   Mean x-coordinate of the defensive line (m).
        y_left:       Left (lower-y) edge of the pocket (m).
        y_right:      Right (upper-y) edge of the pocket (m).
        width:        Pocket width in metres (``y_right - y_left``).
        player_left:  ID of the player to the left of the gap, or ``None``
                      if the gap starts at the pitch edge.
        player_right: ID of the player to the right of the gap, or ``None``
                      if the gap ends at the pitch edge.
    """

    line_depth: float
    y_left: float
    y_right: float
    width: float
    player_left: str | None
    player_right: str | None


def line_breaking_pockets(
    frame: FrameRecord,
    *,
    defending_team_id: str,
    min_pocket_width: float = 5.0,
    line_cluster_threshold: float = 3.0,
    min_line_players: int = 2,
) -> list[Pocket]:
    """Detect pockets between players in each defensive line.

    Defenders are grouped into horizontal lines by agglomerative clustering
    on their x-coordinate (depth): players within *line_cluster_threshold*
    metres of each other are placed on the same line.  Within each line,
    players are sorted by y-coordinate and gaps wider than *min_pocket_width*
    are returned as :class:`Pocket` objects.

    Parameters:
        frame:                  Single tracking frame.
        defending_team_id:      Team ID of the defending side.
        min_pocket_width:       Minimum gap width (m) to be reported (default 5 m).
        line_cluster_threshold: Max x-distance between two players for them to
                                be considered on the same line (default 3 m).
        min_line_players:       Minimum players needed to form a line (default 2).

    Returns:
        List of :class:`Pocket` instances sorted by ``(line_depth, y_left)``.
        Empty list if no qualifying gaps are found.
    """
    # Extract defender positions
    def_indices = [
        i for i, tid in enumerate(frame.team_ids) if tid == defending_team_id
    ]
    if len(def_indices) < min_line_players:
        return []

    def_pids = [frame.player_ids[i] for i in def_indices]
    def_pos = frame.positions[def_indices]  # (M, 2); x = depth, y = width

    x_coords = def_pos[:, 0]
    y_coords = def_pos[:, 1]

    # -----------------------------------------------------------------------
    # Agglomerative single-linkage clustering on x-coordinate
    # -----------------------------------------------------------------------
    order = np.argsort(x_coords)
    x_sorted = x_coords[order]
    y_sorted = y_coords[order]
    pid_sorted = [def_pids[i] for i in order]

    # Assign cluster labels
    labels = np.zeros(len(x_sorted), dtype=int)
    cluster_id = 0
    for k in range(1, len(x_sorted)):
        if x_sorted[k] - x_sorted[k - 1] > line_cluster_threshold:
            cluster_id += 1
        labels[k] = cluster_id

    # -----------------------------------------------------------------------
    # Detect pockets within each cluster
    # -----------------------------------------------------------------------
    pockets: list[Pocket] = []

    for cid in range(cluster_id + 1):
        mask = labels == cid
        if mask.sum() < min_line_players:
            continue

        line_depth = float(x_sorted[mask].mean())
        # Sort players within line by y-coordinate
        y_line = y_sorted[mask]
        p_line = [pid_sorted[i] for i, m in enumerate(mask) if m]
        sort_idx = np.argsort(y_line)
        y_line = y_line[sort_idx]
        p_line = [p_line[i] for i in sort_idx]

        # Gaps between adjacent players
        for j in range(len(y_line) - 1):
            gap = float(y_line[j + 1] - y_line[j])
            if gap >= min_pocket_width:
                pockets.append(Pocket(
                    line_depth=line_depth,
                    y_left=float(y_line[j]),
                    y_right=float(y_line[j + 1]),
                    width=gap,
                    player_left=p_line[j],
                    player_right=p_line[j + 1],
                ))

    pockets.sort(key=lambda p: (p.line_depth, p.y_left))
    return pockets
