"""
Project: PitchAura
File Created: 2026-02-23
Author: Xingnan Zhu
File Name: test_line_breaking.py
Description:
    Tests for tactics.line_breaking_pockets() and the Pocket dataclass.
    Covers single-line gap detection, multi-line separation, threshold
    filtering, sorting, and edge cases (too few players, no gaps).
"""

from __future__ import annotations

import numpy as np
import pytest

from pitch_aura.tactics.line_breaking import Pocket, line_breaking_pockets
from pitch_aura.types import FrameRecord, PitchSpec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _frame(
    defender_positions: list[tuple[float, float]],
    defending_team: str = "away",
    attacker_positions: list[tuple[float, float]] | None = None,
) -> FrameRecord:
    """Build a minimal FrameRecord."""
    if attacker_positions is None:
        attacker_positions = [(0.0, 0.0)]  # single dummy attacker

    def_pids = [f"d{i}" for i in range(len(defender_positions))]
    att_pids = [f"a{i}" for i in range(len(attacker_positions))]
    player_ids = def_pids + att_pids
    team_ids = [defending_team] * len(defender_positions) + ["home"] * len(attacker_positions)
    positions = np.array(list(defender_positions) + list(attacker_positions), dtype=float)

    return FrameRecord(
        timestamp=0.0,
        period=1,
        ball_position=np.array([0.0, 0.0]),
        player_ids=player_ids,
        team_ids=team_ids,
        positions=positions,
        velocities=np.zeros((len(player_ids), 2)),
        is_goalkeeper=np.zeros(len(player_ids), dtype=bool),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLineBreakingPockets:
    def test_no_defenders_returns_empty(self):
        frame = _frame(defender_positions=[])
        pockets = line_breaking_pockets(frame, defending_team_id="away")
        assert pockets == []

    def test_single_defender_returns_empty(self):
        frame = _frame([(0.0, 0.0)])
        pockets = line_breaking_pockets(frame, defending_team_id="away")
        assert pockets == []

    def test_two_players_large_gap(self):
        # Two defenders on the same line (same x), 20 m apart in y
        frame = _frame([(30.0, -10.0), (30.0, 10.0)])
        pockets = line_breaking_pockets(frame, defending_team_id="away", min_pocket_width=5.0)
        assert len(pockets) == 1
        p = pockets[0]
        assert p.width == pytest.approx(20.0)
        assert p.y_left == pytest.approx(-10.0)
        assert p.y_right == pytest.approx(10.0)

    def test_gap_below_threshold_excluded(self):
        # 3 m gap, threshold 5 m → no pocket
        frame = _frame([(30.0, 0.0), (30.0, 3.0)])
        pockets = line_breaking_pockets(frame, defending_team_id="away", min_pocket_width=5.0)
        assert pockets == []

    def test_two_defensive_lines_detected(self):
        # Two clusters: x≈20 and x≈50 (> 3 m threshold)
        defenders = [(20.0, -10.0), (20.0, 10.0), (50.0, -8.0), (50.0, 8.0)]
        frame = _frame(defenders)
        pockets = line_breaking_pockets(
            frame, defending_team_id="away",
            min_pocket_width=5.0, line_cluster_threshold=3.0,
        )
        # Should find one gap per line
        assert len(pockets) == 2
        depths = {p.line_depth for p in pockets}
        assert len(depths) == 2  # two distinct lines

    def test_pocket_players_identified(self):
        frame = _frame([(30.0, -10.0), (30.0, 10.0)])
        pockets = line_breaking_pockets(frame, defending_team_id="away", min_pocket_width=5.0)
        assert len(pockets) == 1
        p = pockets[0]
        assert p.player_left is not None
        assert p.player_right is not None
        assert p.player_left != p.player_right

    def test_returns_pocket_dataclass(self):
        frame = _frame([(30.0, -10.0), (30.0, 10.0)])
        pockets = line_breaking_pockets(frame, defending_team_id="away", min_pocket_width=5.0)
        assert all(isinstance(p, Pocket) for p in pockets)

    def test_pockets_sorted_by_depth_then_y(self):
        # Three players forming 2 gaps on same line
        defenders = [(30.0, -15.0), (30.0, 0.0), (30.0, 15.0)]
        frame = _frame(defenders)
        pockets = line_breaking_pockets(frame, defending_team_id="away", min_pocket_width=5.0)
        assert len(pockets) == 2
        assert pockets[0].y_left < pockets[1].y_left

    def test_no_gap_between_tightly_packed_defenders(self):
        # Defenders 1 m apart; threshold 5 m → no pockets
        defenders = [(30.0, float(i)) for i in range(5)]
        frame = _frame(defenders)
        pockets = line_breaking_pockets(frame, defending_team_id="away", min_pocket_width=5.0)
        assert pockets == []

    def test_line_depth_is_mean_x(self):
        # Two defenders at x=29 and x=31 → cluster at mean 30
        frame = _frame([(29.0, -10.0), (31.0, 10.0)])
        pockets = line_breaking_pockets(
            frame, defending_team_id="away",
            min_pocket_width=5.0, line_cluster_threshold=5.0,
        )
        assert len(pockets) == 1
        assert pockets[0].line_depth == pytest.approx(30.0)
