"""
Project: PitchAura
File Created: 2026-02-26
Author: Xingnan Zhu
File Name: passing_network.py
Description:
    Spatial passing network construction from event data.
    Builds nodes (player average positions) and edges (pass connections)
    from pass events with coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pitch_aura.types import EventRecord


@dataclass(frozen=True, slots=True)
class PassingNetwork:
    """Spatial passing network with player nodes and pass edges.

    Attributes:
        nodes: DataFrame with columns ``player_id``, ``team_id``,
               ``avg_x``, ``avg_y``, ``pass_count``.
        edges: DataFrame with columns ``passer``, ``receiver``,
               ``avg_start_x``, ``avg_start_y``, ``avg_end_x``,
               ``avg_end_y``, ``count``, ``completion_rate``.
    """

    nodes: pd.DataFrame
    edges: pd.DataFrame


def _receiver_from_next(events: list[EventRecord], idx: int) -> str | None:
    """Infer receiver from the next event in the sequence."""
    if idx + 1 < len(events):
        nxt = events[idx + 1]
        if nxt.player_id is not None and nxt.player_id != events[idx].player_id:
            return nxt.player_id
    return None


def passing_network(
    events: list[EventRecord],
    *,
    team_id: str | None = None,
    event_type: str = "pass",
    min_passes: int = 1,
) -> PassingNetwork:
    """Build a spatial passing network from event data.

    Each pass event contributes to a passer–receiver edge.  The receiver
    is inferred from the next event's ``player_id`` if the pass was
    completed.  Nodes represent average positions of each player in pass
    actions.

    Parameters:
        events:     List of event records.
        team_id:    Filter to passes by this team (``None`` = all teams).
        event_type: Event type name to use (default ``"pass"``).
        min_passes: Minimum total passes for a player to appear as a node.

    Returns:
        :class:`PassingNetwork` with ``nodes`` and ``edges`` DataFrames.
    """
    node_cols = ["player_id", "team_id", "avg_x", "avg_y", "pass_count"]
    edge_cols = [
        "passer", "receiver",
        "avg_start_x", "avg_start_y", "avg_end_x", "avg_end_y",
        "count", "completion_rate",
    ]

    et_lower = event_type.lower()

    # Collect pass data
    pass_rows: list[dict] = []
    for i, ev in enumerate(events):
        if ev.event_type.lower() != et_lower:
            continue
        if team_id is not None and ev.team_id != team_id:
            continue
        if ev.player_id is None or ev.coordinates is None:
            continue

        is_complete = ev.result is not None and ev.result.lower() in (
            "complete", "success", "won",
        )

        receiver = _receiver_from_next(events, i) if is_complete else None

        end_xy = ev.end_coordinates if ev.end_coordinates is not None else ev.coordinates

        pass_rows.append({
            "passer": ev.player_id,
            "team_id": ev.team_id,
            "receiver": receiver,
            "start_x": float(ev.coordinates[0]),
            "start_y": float(ev.coordinates[1]),
            "end_x": float(end_xy[0]),
            "end_y": float(end_xy[1]),
            "is_complete": is_complete,
        })

    if not pass_rows:
        return PassingNetwork(
            nodes=pd.DataFrame(columns=node_cols),
            edges=pd.DataFrame(columns=edge_cols),
        )

    df = pd.DataFrame(pass_rows)

    # --- Build nodes ---
    # Player positions from passes they made
    player_stats: dict[str, dict] = {}
    for _, row in df.iterrows():
        pid = row["passer"]
        if pid not in player_stats:
            player_stats[pid] = {
                "team_id": row["team_id"],
                "xs": [],
                "ys": [],
            }
        player_stats[pid]["xs"].append(row["start_x"])
        player_stats[pid]["ys"].append(row["start_y"])

    node_rows = []
    for pid, stats in player_stats.items():
        count = len(stats["xs"])
        if count < min_passes:
            continue
        node_rows.append({
            "player_id": pid,
            "team_id": stats["team_id"],
            "avg_x": float(np.mean(stats["xs"])),
            "avg_y": float(np.mean(stats["ys"])),
            "pass_count": count,
        })

    nodes_df = pd.DataFrame(node_rows, columns=node_cols) if node_rows else pd.DataFrame(
        columns=node_cols,
    )

    # --- Build edges ---
    completed = df[df["receiver"].notna()].copy()
    if completed.empty:
        return PassingNetwork(
            nodes=nodes_df,
            edges=pd.DataFrame(columns=edge_cols),
        )

    edge_rows = []
    for (passer, receiver), group in completed.groupby(["passer", "receiver"]):
        count = len(group)
        if count < 1:
            continue
        # Completion rate: completed passes / total passes from this passer to this receiver
        total_between = len(
            df[(df["passer"] == passer) & (df["receiver"] == receiver)]
        )
        edge_rows.append({
            "passer": passer,
            "receiver": receiver,
            "avg_start_x": float(group["start_x"].mean()),
            "avg_start_y": float(group["start_y"].mean()),
            "avg_end_x": float(group["end_x"].mean()),
            "avg_end_y": float(group["end_y"].mean()),
            "count": count,
            "completion_rate": count / total_between if total_between > 0 else 0.0,
        })

    edges_df = pd.DataFrame(edge_rows, columns=edge_cols) if edge_rows else pd.DataFrame(
        columns=edge_cols,
    )

    return PassingNetwork(nodes=nodes_df, edges=edges_df)
