"""
Project: PitchAura
File Created: 2026-02-25
Author: Xingnan Zhu
File Name: viz_gravity_sdi_nsg.py
Description:
    Spatial Gravity & Deformation visualisation using real tracking data
    from Atlético Madrid vs Real Madrid.

    Demonstrates the two new metrics:
      - SDI (Spatial Drag Index):   how much defensive control is dissolved
                                    by a player's off-ball run.
      - NSG (Net Space Generated):  space gained near a teammate because
                                    of the mover's run.

    Four outputs:
      11_gravity_sdi_timeseries.html       — SDI curve for the top mover
      12_gravity_nsg_timeseries.html       — NSG curve (mover → beneficiary)
      13_gravity_deformation_peak.html     — deformation field at peak SDI
      14_gravity_pitch_control_compare.html— actual vs counterfactual control

    Run:
        uv run python scripts/viz_gravity_sdi_nsg.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

# ── paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR  = REPO_ROOT / "data" / "Atlético Madrid_Real Madrid"
OUT_DIR   = Path(__file__).parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO_ROOT / "src"))

# ── pitch_aura imports ────────────────────────────────────────────────────────
from pitch_aura.types import FrameRecord, FrameSequence, PitchSpec
from pitch_aura.space.kinematic import KinematicControlModel
from pitch_aura.tactics.gravity import (
    _counterfactual_frame,
    penalty_zone_weights,
    spatial_drag_index,
    net_space_generated,
)
from pitch_aura.viz import (
    plot_players,
    plot_deformation_field,
    plot_gravity_timeseries,
)
from pitch_aura.viz._pitch_draw import _make_pitch_traces

# ── constants ─────────────────────────────────────────────────────────────────
HOME_TEAM_ID = "275"   # Atlético Madrid
AWAY_TEAM_ID = "262"   # Real Madrid CF
FRAME_RATE   = 10.0
PITCH        = PitchSpec(length=105.0, width=68.0, origin="center")

# Lower resolution for speed; increase to (50, 32) for production quality
RESOLUTION   = (40, 27)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_match_meta() -> dict:
    with open(DATA_DIR / "2033526_match_data.json") as f:
        return json.load(f)


def build_player_lookup(meta: dict) -> tuple[dict[str, str], dict[str, str]]:
    """(player_id → team_id, player_id → short_name)"""
    id_to_team = {str(p["id"]): str(p["team_id"]) for p in meta["players"]}
    id_to_name = {str(p["id"]): p["short_name"]   for p in meta["players"]}
    return id_to_team, id_to_name


def _parse_timestamp(ts: str | float) -> float:
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


def raw_row_to_frame(
    row: dict,
    id_to_team: dict[str, str],
) -> FrameRecord | None:
    bd = row["ball_data"]
    if bd["x"] is None:
        return None
    player_rows = [p for p in row.get("player_data", []) if p["x"] is not None]
    if not player_rows:
        return None
    player_ids = [str(p["player_id"]) for p in player_rows]
    return FrameRecord(
        timestamp=_parse_timestamp(row["timestamp"]),
        period=int(row["period"]),
        ball_position=np.array([bd["x"], bd["y"]], dtype=float),
        player_ids=player_ids,
        team_ids=[id_to_team.get(pid, "unknown") for pid in player_ids],
        positions=np.array([[p["x"], p["y"]] for p in player_rows], dtype=float),
    )


def load_frames(
    meta: dict,
    n_frames: int = 250,
    period: int = 1,
    start_offset: int = 100,
) -> list[FrameRecord]:
    """Load up to *n_frames* valid FrameRecords from the JSONL file."""
    periods     = {p["name"]: p for p in meta["match_periods"]}
    start_frame = periods[f"period_{period}"]["start_frame"] + start_offset
    id_to_team, _ = build_player_lookup(meta)

    frames: list[FrameRecord] = []
    with open(DATA_DIR / "2033526_tracking_extrapolated.jsonl") as f:
        for line in f:
            row = json.loads(line)
            if row["frame"] < start_frame:
                continue
            if len(frames) >= n_frames:
                break
            fr = raw_row_to_frame(row, id_to_team)
            if fr is not None:
                frames.append(fr)
    return frames


def add_velocities(frames: list[FrameRecord]) -> list[FrameRecord]:
    dt  = 1.0 / FRAME_RATE
    out = []
    for i, fr in enumerate(frames):
        vels = np.zeros_like(fr.positions)
        if 0 < i < len(frames) - 1:
            prev = frames[i - 1]
            nxt  = frames[i + 1]
            prev_idx = {pid: j for j, pid in enumerate(prev.player_ids)}
            nxt_idx  = {pid: j for j, pid in enumerate(nxt.player_ids)}
            for j, pid in enumerate(fr.player_ids):
                if pid in prev_idx and pid in nxt_idx:
                    vels[j] = (nxt.positions[nxt_idx[pid]] -
                               prev.positions[prev_idx[pid]]) / (2 * dt)
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
# 2.  Player selection
# ══════════════════════════════════════════════════════════════════════════════

def find_active_home_players(
    frames: list[FrameRecord],
    min_appearance_ratio: float = 0.8,
) -> list[str]:
    counts: dict[str, int] = {}
    for fr in frames:
        for pid, tid in zip(fr.player_ids, fr.team_ids):
            if tid == HOME_TEAM_ID:
                counts[pid] = counts.get(pid, 0) + 1
    threshold = int(len(frames) * min_appearance_ratio)
    return [pid for pid, cnt in counts.items() if cnt >= threshold]


def compute_total_displacement(frames: list[FrameRecord], player_id: str) -> float:
    positions = []
    for fr in frames:
        if player_id in fr.player_ids:
            idx = fr.player_ids.index(player_id)
            positions.append(fr.positions[idx])
    if len(positions) < 2:
        return 0.0
    arr = np.array(positions)
    return float(np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1)))


def pick_runner_and_beneficiary(
    frames: list[FrameRecord],
    id_to_name: dict[str, str],
) -> tuple[str, str, str, str]:
    """Return (runner_id, runner_name, beneficiary_id, beneficiary_name)."""
    active = find_active_home_players(frames)
    if len(active) < 2:
        raise RuntimeError("Not enough consistently-present home players")
    displacements = {pid: compute_total_displacement(frames, pid) for pid in active}
    ranked = sorted(displacements, key=lambda p: displacements[p], reverse=True)
    runner_id, ben_id = ranked[0], ranked[1]
    return (
        runner_id, id_to_name.get(runner_id, runner_id),
        ben_id,    id_to_name.get(ben_id, ben_id),
    )


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Visualization helpers
# ══════════════════════════════════════════════════════════════════════════════

def _save(fig, name: str) -> None:
    path = OUT_DIR / f"{name}.html"
    fig.write_html(str(path))
    print(f"  ✓  saved → {path.relative_to(REPO_ROOT)}")


# ── viz 1: SDI timeseries ─────────────────────────────────────────────────────

def viz_sdi_timeseries(df_sdi, runner_name: str) -> None:
    print(f"\n[11] SDI timeseries — {runner_name}")
    fig = plot_gravity_timeseries(
        df_sdi,
        metric_col="sdi_m2",
        label=f"SDI — {runner_name}",
        line_color="#ef4444",
    )
    fig.add_scatter(
        x=df_sdi["timestamp"],
        y=df_sdi["displacement_m"],
        mode="lines",
        name="Displacement (m)",
        line=dict(color="#94a3b8", width=1, dash="dot"),
        yaxis="y2",
    )
    fig.update_layout(
        title=f"Spatial Drag Index — {runner_name}",
        yaxis2=dict(
            title="Displacement (m)",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        legend=dict(x=0.01, y=0.99),
    )
    _save(fig, "11_gravity_sdi_timeseries")
    peak_idx = df_sdi["sdi_m2"].idxmax()
    print(f"      peak SDI = {df_sdi['sdi_m2'].max():.1f} m²  "
          f"at t={df_sdi.loc[peak_idx, 'timestamp']:.1f}s")


# ── viz 2: NSG timeseries ─────────────────────────────────────────────────────

def viz_nsg_timeseries(
    df_nsg, runner_name: str, beneficiary_name: str, df_sdi,
) -> None:
    print(f"\n[12] NSG timeseries — {runner_name} → {beneficiary_name}")
    fig = plot_gravity_timeseries(
        df_nsg,
        metric_col="nsg_m2",
        label=f"NSG (penalty zone) — {runner_name} → {beneficiary_name}",
        line_color="#3b82f6",
    )
    fig.add_scatter(
        x=df_sdi["timestamp"],
        y=df_sdi["sdi_m2"],
        mode="lines",
        name=f"SDI — {runner_name}",
        line=dict(color="#ef4444", width=1.5, dash="dash"),
    )
    fig.update_layout(
        title=f"Net Space Generated: {runner_name} → {beneficiary_name}",
        yaxis_title="Space (m²)",
        legend=dict(x=0.01, y=0.99),
    )
    _save(fig, "12_gravity_nsg_timeseries")
    peak_idx = df_nsg["nsg_m2"].idxmax()
    print(f"      peak NSG = {df_nsg['nsg_m2'].max():.1f} m²  "
          f"at t={df_nsg.loc[peak_idx, 'timestamp']:.1f}s")


# ── viz 3: deformation field at peak SDI frame ───────────────────────────────

def viz_deformation_peak(
    df_sdi,
    deformations,
    sequence: FrameSequence,
    runner_name: str,
    id_to_name: dict[str, str],
) -> None:
    print(f"\n[13] Deformation field at peak SDI — {runner_name}")
    peak_idx    = int(df_sdi["sdi_m2"].idxmax())
    peak_ts     = df_sdi.loc[peak_idx, "timestamp"]
    peak_deform = deformations[peak_idx]
    peak_sdi    = df_sdi.loc[peak_idx, "sdi_m2"]
    peak_frame  = min(sequence.frames, key=lambda f: abs(f.timestamp - peak_ts))

    fig = plot_deformation_field(
        peak_deform,
        peak_frame,
        colorscale="RdBu",
        opacity=0.80,
        home_team_id=HOME_TEAM_ID,
        player_names=id_to_name,
    )
    fig.update_layout(
        title=(
            f"Deformation Field — {runner_name}<br>"
            f"<sup>t = {peak_ts:.1f}s | SDI = {peak_sdi:.1f} m² | "
            f"displacement = {df_sdi.loc[peak_idx, 'displacement_m']:.1f} m</sup>"
        ),
    )
    _save(fig, "13_gravity_deformation_peak")


# ── viz 4: actual vs counterfactual pitch control ─────────────────────────────

def viz_pitch_control_compare(
    df_sdi,
    sequence: FrameSequence,
    runner_id: str,
    runner_name: str,
    id_to_name: dict[str, str],
) -> None:
    print(f"\n[14] Actual vs counterfactual pitch control — {runner_name}")

    peak_idx    = int(df_sdi["sdi_m2"].idxmax())
    peak_ts     = df_sdi.loc[peak_idx, "timestamp"]
    peak_frame  = min(sequence.frames, key=lambda f: abs(f.timestamp - peak_ts))
    frozen_pos  = sequence.frames[0].positions[
        sequence.frames[0].player_ids.index(runner_id)
    ].copy()
    cf_frame    = _counterfactual_frame(peak_frame, runner_id, frozen_pos)

    model       = KinematicControlModel(resolution=RESOLUTION, pitch=PITCH)
    actual_grid = model.control(peak_frame, team_id=HOME_TEAM_ID)
    cf_grid     = model.control(cf_frame,   team_id=HOME_TEAM_ID)

    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"Actual  (t={peak_ts:.1f}s)",
            f"Counterfactual  ({runner_name} frozen at t₀)",
        ),
        horizontal_spacing=0.05,
    )

    pitch_traces = _make_pitch_traces(PITCH, "white")

    def _add_panel(grid, frame: FrameRecord, col: int) -> None:
        # Pitch markings
        for tr in pitch_traces:
            fig.add_trace(
                go.Scatter(**{**tr.to_plotly_json(), "showlegend": False}),
                row=1, col=col,
            )
        # Control heatmap
        fig.add_trace(
            go.Heatmap(
                x=grid.x_centers,
                y=grid.y_centers,
                z=grid.values.T,
                colorscale="RdBu_r",
                zmin=0.0,
                zmax=1.0,
                zmid=0.5,
                opacity=0.75,
                showscale=(col == 2),
                colorbar=dict(title="Control", thickness=12, len=0.55, x=1.01),
                hovertemplate="x: %{x:.1f}<br>y: %{y:.1f}<br>P: %{z:.3f}<extra></extra>",
            ),
            row=1, col=col,
        )
        # Players with names in hover
        player_fig = plot_players(
            frame,
            show_pitch=False,
            home_team_id=HOME_TEAM_ID,
            show_velocity=False,
            show_labels=False,
            player_names=id_to_name,
        )
        for tr in player_fig.data:
            fig.add_trace(tr, row=1, col=col)
        # Runner star
        if runner_id in frame.player_ids:
            idx = frame.player_ids.index(runner_id)
            rx, ry = frame.positions[idx]
            fig.add_trace(
                go.Scatter(
                    x=[rx], y=[ry],
                    mode="markers",
                    marker=dict(symbol="star", size=16, color="gold",
                                line=dict(color="black", width=1)),
                    name=runner_name,
                    showlegend=(col == 1),
                    hovertemplate=(
                        f"{runner_name}<br>"
                        f"x: {rx:.1f}<br>y: {ry:.1f}<extra></extra>"
                    ),
                ),
                row=1, col=col,
            )

    _add_panel(actual_grid, peak_frame, col=1)
    _add_panel(cf_grid,     cf_frame,   col=2)

    # ── fix axis ranges and pitch background colour ────────────────────────
    x0, x1 = PITCH.x_range
    y0, y1 = PITCH.y_range
    _axis = dict(range=[x0, x1], showgrid=False, zeroline=False, showticklabels=False)
    _axis_y = dict(range=[y0, y1], showgrid=False, zeroline=False, showticklabels=False)

    diff_area = actual_grid.total_area(0.5) - cf_grid.total_area(0.5)
    fig.update_layout(
        # Green pitch background for both panels
        plot_bgcolor="#1a472a",
        paper_bgcolor="#111111",
        # Axis ranges — keeps pitch marks exactly within bounds
        xaxis={**_axis, "scaleanchor": "y",  "scaleratio": 1},
        xaxis2={**_axis, "scaleanchor": "y2", "scaleratio": 1},
        yaxis=_axis_y,
        yaxis2=_axis_y,
        title=dict(
            text=(
                f"Pitch Control: Actual vs Counterfactual — {runner_name}<br>"
                f"<sup>Δ controlled area (>50%) = {diff_area:+.1f} m²</sup>"
            ),
            font=dict(color="white"),
        ),
        font=dict(color="white"),
        width=1420,
        height=540,
    )
    _save(fig, "14_gravity_pitch_control_compare")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 65)
    print("PitchAura — SDI / NSG gravity analysis  (Atlético vs Real Madrid)")
    print("=" * 65)

    # ── load ──────────────────────────────────────────────────────────────────
    print("\nLoading 250 frames from period 1 (offset 100) …")
    t0 = time.perf_counter()
    meta         = load_match_meta()
    _, id_to_name = build_player_lookup(meta)
    raw_frames   = load_frames(meta, n_frames=250, period=1, start_offset=100)
    frames       = add_velocities(raw_frames)
    sequence     = make_sequence(frames)
    print(f"  {len(frames)} frames loaded in {time.perf_counter()-t0:.2f}s")
    print(f"  time span: {frames[0].timestamp:.1f}s → {frames[-1].timestamp:.1f}s")

    # ── pick players ──────────────────────────────────────────────────────────
    runner_id, runner_name, ben_id, ben_name = pick_runner_and_beneficiary(
        frames, id_to_name,
    )
    print(f"\nSelected runner      : {runner_name}  (id={runner_id})")
    print(f"Selected beneficiary : {ben_name}  (id={ben_id})")

    time_window = frames[-1].timestamp - frames[0].timestamp + 1.0

    # ── SDI ───────────────────────────────────────────────────────────────────
    print(f"\nComputing SDI (resolution={RESOLUTION}, {len(frames)} frames) …")
    t0 = time.perf_counter()
    df_sdi, deformations = spatial_drag_index(
        sequence,
        player_id=runner_id,
        attacking_team_id=HOME_TEAM_ID,
        time_window=time_window,
        resolution=RESOLUTION,
        return_deformation=True,
    )
    print(f"  done in {time.perf_counter()-t0:.2f}s  |  {len(df_sdi)} frames analysed")

    # ── NSG (penalty zone weighted) ───────────────────────────────────────────
    print(f"\nComputing NSG (penalty zone weights, {len(frames)} frames) …")
    zone_w = penalty_zone_weights(PITCH, RESOLUTION, side="right")
    t0 = time.perf_counter()
    df_nsg = net_space_generated(
        sequence,
        mover_id=runner_id,
        beneficiary_id=ben_id,
        attacking_team_id=HOME_TEAM_ID,
        time_window=time_window,
        zone_weights=zone_w,
        resolution=RESOLUTION,
    )
    print(f"  done in {time.perf_counter()-t0:.2f}s  |  {len(df_nsg)} frames analysed")

    # ── visualisations ────────────────────────────────────────────────────────
    viz_sdi_timeseries(df_sdi, runner_name)
    viz_nsg_timeseries(df_nsg, runner_name, ben_name, df_sdi)
    viz_deformation_peak(df_sdi, deformations, sequence, runner_name, id_to_name)
    viz_pitch_control_compare(df_sdi, sequence, runner_id, runner_name, id_to_name)

    print("\n" + "=" * 65)
    print(f"All outputs written to  {OUT_DIR.relative_to(REPO_ROOT)}/")
    print("=" * 65)


if __name__ == "__main__":
    main()
