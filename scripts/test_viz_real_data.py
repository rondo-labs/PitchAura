"""
Project: PitchAura
File Created: 2026-02-25
Author: Xingnan Zhu
File Name: test_viz_real_data.py
Description:
    End-to-end smoke test for all viz methods using the real
    Atlético Madrid vs Real Madrid tracking + event dataset.

    Run:
        uv run python scripts/test_viz_real_data.py

    Output HTML files land in  scripts/output/
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parent.parent
DATA_DIR   = REPO_ROOT / "data" / "Atlético Madrid_Real Madrid"
OUT_DIR    = Path(__file__).parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO_ROOT / "src"))

# ── pitch_aura imports ─────────────────────────────────────────────────────────
from pitch_aura.types import FrameRecord, FrameSequence, PitchSpec
from pitch_aura.space.voronoi import VoronoiModel
from pitch_aura.space.kinematic import KinematicControlModel
from pitch_aura.tactics.line_breaking import line_breaking_pockets
from pitch_aura.tactics.passing_lanes import passing_lane_lifespan
from pitch_aura.viz import (
    pitch_background,
    plot_players,
    plot_heatmap,
    plot_voronoi,
    plot_pockets,
    plot_passing_lane,
    animate_sequence,
    plot_pitch_control,
    plot_voronoi_control,
)

# ── constants ──────────────────────────────────────────────────────────────────
HOME_TEAM_ID = "275"   # Atlético Madrid (home)
AWAY_TEAM_ID = "262"   # Real Madrid CF   (away)
FRAME_RATE   = 10.0    # tracking is ~10 fps based on frame numbers
PITCH        = PitchSpec(length=105.0, width=68.0, origin="center")


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Load tracking data → list[FrameRecord]
# ══════════════════════════════════════════════════════════════════════════════

def load_match_meta() -> dict:
    with open(DATA_DIR / "2033526_match_data.json") as f:
        return json.load(f)


def build_player_lookup(meta: dict) -> tuple[dict, dict]:
    """Return (player_id → team_id, player_id → short_name)."""
    id_to_team = {str(p["id"]): str(p["team_id"]) for p in meta["players"]}
    id_to_name = {str(p["id"]): p["short_name"]   for p in meta["players"]}
    return id_to_team, id_to_name


def _parse_timestamp(ts: str | float) -> float:
    """Convert 'HH:MM:SS.ss' string or plain float to seconds."""
    if isinstance(ts, (int, float)):
        return float(ts)
    parts = ts.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    if len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    return float(ts)


def raw_rows_to_frame(
    row: dict,
    id_to_team: dict[str, str],
    id_to_name: dict[str, str],
) -> FrameRecord | None:
    """Convert one JSONL row → FrameRecord.  Returns None if no position data."""
    bd = row["ball_data"]
    if bd["x"] is None:
        return None

    player_rows = [p for p in row.get("player_data", []) if p["x"] is not None]
    if not player_rows:
        return None

    player_ids = [str(p["player_id"]) for p in player_rows]
    team_ids   = [id_to_team.get(pid, "unknown") for pid in player_ids]
    positions  = np.array([[p["x"], p["y"]] for p in player_rows], dtype=float)
    ball_pos   = np.array([bd["x"], bd["y"]], dtype=float)

    return FrameRecord(
        timestamp=_parse_timestamp(row["timestamp"]),
        period=int(row["period"]),
        ball_position=ball_pos,
        player_ids=player_ids,
        team_ids=team_ids,
        positions=positions,
    )


def load_frames(
    n_frames: int = 300,
    period: int = 1,
    start_offset: int = 0,
) -> list[FrameRecord]:
    """Load up to *n_frames* valid FrameRecords from the JSONL file."""
    meta       = load_match_meta()
    periods    = {p["name"]: p for p in meta["match_periods"]}
    period_key = f"period_{period}"
    start_frame = periods[period_key]["start_frame"] + start_offset

    id_to_team, id_to_name = build_player_lookup(meta)

    frames: list[FrameRecord] = []
    with open(DATA_DIR / "2033526_tracking_extrapolated.jsonl") as f:
        for line in f:
            row = json.loads(line)
            if row["frame"] < start_frame:
                continue
            if len(frames) >= n_frames:
                break
            fr = raw_rows_to_frame(row, id_to_team, id_to_name)
            if fr is not None:
                frames.append(fr)

    return frames


def add_finite_diff_velocities(frames: list[FrameRecord]) -> list[FrameRecord]:
    """Compute per-player velocity via central differences (in place)."""
    dt = 1.0 / FRAME_RATE
    out = []
    for i, fr in enumerate(frames):
        if i == 0 or i == len(frames) - 1:
            vels = np.zeros_like(fr.positions)
        else:
            prev = frames[i - 1]
            nxt  = frames[i + 1]
            # align by player_id
            pid_to_idx_prev = {pid: j for j, pid in enumerate(prev.player_ids)}
            pid_to_idx_nxt  = {pid: j for j, pid in enumerate(nxt.player_ids)}
            vels = np.zeros_like(fr.positions)
            for j, pid in enumerate(fr.player_ids):
                if pid in pid_to_idx_prev and pid in pid_to_idx_nxt:
                    p0 = prev.positions[pid_to_idx_prev[pid]]
                    p2 = nxt.positions[pid_to_idx_nxt[pid]]
                    vels[j] = (p2 - p0) / (2 * dt)

        out.append(FrameRecord(
            timestamp=fr.timestamp,
            period=fr.period,
            ball_position=fr.ball_position,
            player_ids=fr.player_ids,
            team_ids=fr.team_ids,
            positions=fr.positions,
            velocities=vels,
            is_goalkeeper=fr.is_goalkeeper,
        ))
    return out


def make_sequence(frames: list[FrameRecord]) -> FrameSequence:
    return FrameSequence(
        frames=frames,
        frame_rate=FRAME_RATE,
        pitch=PITCH,
        home_team_id=HOME_TEAM_ID,
        away_team_id=AWAY_TEAM_ID,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Individual viz tests
# ══════════════════════════════════════════════════════════════════════════════

def _save(fig, name: str) -> None:
    path = OUT_DIR / f"{name}.html"
    fig.write_html(str(path))
    print(f"  ✓  saved → {path.relative_to(REPO_ROOT)}")


def test_pitch_background() -> None:
    print("\n[1] pitch_background()")
    fig = pitch_background(PITCH)
    _save(fig, "01_pitch_background")


def test_plot_players(frame: FrameRecord) -> None:
    print("\n[2] plot_players()")
    fig = plot_players(
        frame,
        pitch=PITCH,
        home_team_id=HOME_TEAM_ID,
        show_velocity=True,
        show_labels=True,
    )
    _save(fig, "02_plot_players")


def test_plot_players_no_velocity(frame: FrameRecord) -> None:
    print("\n[3] plot_players() — no velocity arrows")
    fig = plot_players(
        frame,
        pitch=PITCH,
        home_team_id=HOME_TEAM_ID,
        show_velocity=False,
        show_labels=False,
    )
    _save(fig, "03_plot_players_no_vel")


def test_plot_heatmap(frame: FrameRecord) -> None:
    print("\n[4] plot_heatmap() — KinematicControlModel")
    t0 = time.perf_counter()
    model = KinematicControlModel(resolution=(50, 32), pitch=PITCH)
    grid  = model.control(frame, team_id=HOME_TEAM_ID)
    print(f"      KinematicControlModel.compute() took {time.perf_counter()-t0:.2f}s")
    fig = plot_heatmap(grid)
    _save(fig, "04_plot_heatmap")
    return grid


def test_plot_pitch_control(frame: FrameRecord, grid) -> None:
    print("\n[5] plot_pitch_control() — heatmap + players")
    fig = plot_pitch_control(
        grid,
        frame,
        home_team_id=HOME_TEAM_ID,
        show_velocity=True,
    )
    _save(fig, "05_plot_pitch_control")


def test_plot_voronoi(frame: FrameRecord) -> None:
    print("\n[6] plot_voronoi() — VoronoiModel")
    model  = VoronoiModel(pitch=PITCH)
    result = model.control(frame)
    fig = plot_voronoi(
        result,
        frame,
        home_team_id=HOME_TEAM_ID,
        show_areas=True,
    )
    _save(fig, "06_plot_voronoi")
    return result


def test_plot_voronoi_control(frame: FrameRecord, result) -> None:
    print("\n[7] plot_voronoi_control() — regions + players")
    fig = plot_voronoi_control(
        result,
        frame,
        home_team_id=HOME_TEAM_ID,
    )
    _save(fig, "07_plot_voronoi_control")


def test_plot_pockets(frame: FrameRecord) -> None:
    print("\n[8] plot_pockets() — line_breaking_pockets")
    pockets = line_breaking_pockets(
        frame,
        defending_team_id=AWAY_TEAM_ID,
        min_pocket_width=5.0,
    )
    print(f"      found {len(pockets)} pocket(s)")
    fig = plot_players(frame, pitch=PITCH, home_team_id=HOME_TEAM_ID, show_velocity=False)
    fig = plot_pockets(pockets, fig=fig, pitch=PITCH, show_pitch=False)
    _save(fig, "08_plot_pockets")


def test_plot_passing_lane(sequence: FrameSequence) -> None:
    print("\n[9] plot_passing_lane() — passing_lane_lifespan")
    # Use first frame as reference; pick ball carrier + a target player
    frame = sequence.frames[0]
    atl_positions = frame.team_positions(HOME_TEAM_ID)
    atl_ids = [pid for pid, tid in zip(frame.player_ids, frame.team_ids)
               if tid == HOME_TEAM_ID]

    if len(atl_ids) < 2:
        print("      skipped — not enough home players in frame")
        return

    passer_id = atl_ids[0]
    target_id = atl_ids[1]

    try:
        lifespan = passing_lane_lifespan(
            sequence,
            passer_id=passer_id,
            receiver_id=target_id,
        )
        print(f"      lifespan = {lifespan:.2f}s")
    except Exception as exc:
        print(f"      passing_lane_lifespan raised: {exc}")
        lifespan = None

    fig = plot_passing_lane(frame, passer_id, target_id, pitch=PITCH)
    _save(fig, "09_plot_passing_lane")


def test_animate_sequence(sequence: FrameSequence) -> None:
    print("\n[10] animate_sequence() — 50-frame clip")
    # Animate first 50 frames only (keep file size manageable)
    short_seq = sequence[:50]
    fig = animate_sequence(
        short_seq,
        home_team_id=HOME_TEAM_ID,
        frame_duration=100,
    )
    _save(fig, "10_animate_sequence")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 60)
    print("PitchAura — viz smoke test  (Atlético vs Real Madrid)")
    print("=" * 60)

    print("\nLoading 300 frames from period 1 …")
    t0 = time.perf_counter()
    raw_frames = load_frames(n_frames=300, period=1, start_offset=0)
    print(f"  loaded {len(raw_frames)} frames in {time.perf_counter()-t0:.2f}s")

    print("Computing finite-difference velocities …")
    frames = add_finite_diff_velocities(raw_frames)

    sequence = make_sequence(frames)
    frame    = frames[50]          # a representative mid-sequence frame

    print(f"\nSample frame summary:")
    print(f"  timestamp  = {frame.timestamp:.1f}s  period={frame.period}")
    print(f"  n_players  = {frame.n_players}")
    home_n = int(frame.team_mask(HOME_TEAM_ID).sum())
    away_n = int(frame.team_mask(AWAY_TEAM_ID).sum())
    print(f"  Atlético   = {home_n} players")
    print(f"  Real Madrid= {away_n} players")
    print(f"  ball       = ({frame.ball_position[0]:.1f}, {frame.ball_position[1]:.1f})")

    # ── run all tests ─────────────────────────────────────────────────────────
    test_pitch_background()
    test_plot_players(frame)
    test_plot_players_no_velocity(frame)
    grid   = test_plot_heatmap(frame)
    test_plot_pitch_control(frame, grid)
    result = test_plot_voronoi(frame)
    test_plot_voronoi_control(frame, result)
    test_plot_pockets(frame)
    test_plot_passing_lane(sequence)
    test_animate_sequence(sequence)

    print("\n" + "=" * 60)
    print(f"All outputs written to  {OUT_DIR.relative_to(REPO_ROOT)}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
