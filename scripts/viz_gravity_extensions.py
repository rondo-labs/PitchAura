"""
Project: PitchAura
File Created: 2026-02-25
Author: Xingnan Zhu
File Name: viz_gravity_extensions.py
Description:
    Demonstrates the 5 new spatial gravity extension features using real
    tracking data from Atlético Madrid vs Real Madrid.

    Six outputs:
      21_gravity_efficiency.html       — SDI + efficiency dual-axis + profile
      22_deformation_recovery.html     — SDI with peak & half-life annotated
      23_flow_field_peak.html          — drag vector arrows at peak SDI frame
      24_interaction_matrix.html       — N×N who-creates-space-for-whom
      25_vision_vs_base_sdi.html       — VisionAwareModel vs base SDI comparison
      26_flow_field_animation.html     — full temporal animation with time slider

    Run:
        uv run python scripts/viz_gravity_extensions.py
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
from pitch_aura.tactics.gravity import (
    deformation_flow_field,
    deformation_recovery,
    gravity_interaction_matrix,
    gravity_profile,
    spatial_drag_index,
)
from pitch_aura.cognitive.gravity_vision import VisionAwareControlModel
from pitch_aura.viz import (
    plot_deformation_field,
    plot_flow_field,
    plot_gravity_timeseries,
    plot_interaction_matrix,
)

# ── constants ─────────────────────────────────────────────────────────────────
HOME_TEAM_ID = "275"    # Atlético Madrid
AWAY_TEAM_ID = "262"    # Real Madrid CF
FRAME_RATE   = 10.0
PITCH        = PitchSpec(length=105.0, width=68.0, origin="center")

RESOLUTION      = (40, 27)   # standard resolution
RESOLUTION_FAST = (20, 14)   # fast resolution for interaction matrix


# ══════════════════════════════════════════════════════════════════════════════
# Data loading (same helpers as viz_gravity_sdi_nsg.py)
# ══════════════════════════════════════════════════════════════════════════════

def load_match_meta() -> dict:
    with open(DATA_DIR / "2033526_match_data.json") as f:
        return json.load(f)


def build_player_lookup(meta: dict) -> tuple[dict[str, str], dict[str, str]]:
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


def raw_row_to_frame(row: dict, id_to_team: dict[str, str]) -> FrameRecord | None:
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


def load_frames(meta: dict, n_frames: int = 200, period: int = 1,
                start_offset: int = 100) -> list[FrameRecord]:
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
            prev, nxt = frames[i - 1], frames[i + 1]
            prev_idx = {pid: j for j, pid in enumerate(prev.player_ids)}
            nxt_idx  = {pid: j for j, pid in enumerate(nxt.player_ids)}
            for j, pid in enumerate(fr.player_ids):
                if pid in prev_idx and pid in nxt_idx:
                    vels[j] = (nxt.positions[nxt_idx[pid]] -
                               prev.positions[prev_idx[pid]]) / (2 * dt)
        out.append(FrameRecord(
            timestamp=fr.timestamp, period=fr.period,
            ball_position=fr.ball_position,
            player_ids=fr.player_ids, team_ids=fr.team_ids,
            positions=fr.positions, velocities=vels,
            is_goalkeeper=fr.is_goalkeeper,
        ))
    return out


def make_sequence(frames: list[FrameRecord]) -> FrameSequence:
    return FrameSequence(
        frames=frames, frame_rate=FRAME_RATE, pitch=PITCH,
        home_team_id=HOME_TEAM_ID, away_team_id=AWAY_TEAM_ID,
    )


def find_active_home_players(frames: list[FrameRecord],
                              min_ratio: float = 0.8) -> list[str]:
    counts: dict[str, int] = {}
    for fr in frames:
        for pid, tid in zip(fr.player_ids, fr.team_ids):
            if tid == HOME_TEAM_ID:
                counts[pid] = counts.get(pid, 0) + 1
    thr = int(len(frames) * min_ratio)
    return [pid for pid, cnt in counts.items() if cnt >= thr]


def compute_total_displacement(frames: list[FrameRecord], pid: str) -> float:
    pos = [fr.positions[fr.player_ids.index(pid)]
           for fr in frames if pid in fr.player_ids]
    if len(pos) < 2:
        return 0.0
    arr = np.array(pos)
    return float(np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1)))


def pick_runner(frames: list[FrameRecord], id_to_name: dict) -> tuple[str, str]:
    active  = find_active_home_players(frames)
    ranked  = sorted(active, key=lambda p: compute_total_displacement(frames, p), reverse=True)
    rid     = ranked[0]
    return rid, id_to_name.get(rid, rid)


def _save(fig: go.Figure, name: str) -> None:
    path = OUT_DIR / f"{name}.html"
    fig.write_html(str(path))
    print(f"  ✓  saved → {path.relative_to(REPO_ROOT)}")


# ══════════════════════════════════════════════════════════════════════════════
# Feature 1 — Gravity Efficiency
# ══════════════════════════════════════════════════════════════════════════════

def viz_efficiency(df_sdi, runner_name: str, sequence: FrameSequence,
                   runner_id: str) -> None:
    print(f"\n[21] Gravity Efficiency — {runner_name}")

    # ── timeseries: SDI m² (left y) + efficiency m²/m (right y) ──────────────
    fig = plot_gravity_timeseries(
        df_sdi, metric_col="sdi_m2",
        label=f"SDI (m²) — {runner_name}",
        line_color="#ef4444",
    )
    fig.add_trace(go.Scatter(
        x=df_sdi["timestamp"],
        y=df_sdi["sdi_efficiency"],
        mode="lines",
        name="Efficiency (m²/m)",
        line=dict(color="#f59e0b", width=2, dash="dot"),
        yaxis="y2",
        hovertemplate="t: %{x:.2f}s<br>Efficiency: %{y:.2f} m²/m<extra></extra>",
    ))
    fig.update_layout(
        yaxis2=dict(
            title="SDI Efficiency (m²/m)",
            overlaying="y",
            side="right",
            showgrid=False,
            tickfont=dict(color="#f59e0b"),
            title_font=dict(color="#f59e0b"),
        ),
        legend=dict(x=0.01, y=0.99),
    )

    # ── gravity_profile summary as annotation ─────────────────────────────────
    profile = gravity_profile(
        sequence, player_id=runner_id, attacking_team_id=HOME_TEAM_ID,
        resolution=RESOLUTION,
    )
    ann_text = (
        f"<b>Gravity Profile — {runner_name}</b><br>"
        f"Total SDI: {profile['total_sdi_m2']:.0f} m²<br>"
        f"Peak SDI: {profile['peak_sdi_m2']:.1f} m²<br>"
        f"Mean efficiency: {profile['mean_sdi_efficiency']:.2f} m²/m<br>"
        f"Max displacement: {profile['total_displacement_m']:.1f} m<br>"
        f"Significant frames (≥2m): {profile['n_significant_frames']}"
    )
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.99, y=0.99,
        text=ann_text,
        showarrow=False,
        align="right",
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#94a3b8",
        borderwidth=1,
        font=dict(size=11),
        xanchor="right", yanchor="top",
    )
    fig.update_layout(
        title=f"Gravity Efficiency — {runner_name}",
    )
    _save(fig, "21_gravity_efficiency")

    peak_idx = df_sdi["sdi_m2"].idxmax()
    print(f"      peak SDI        = {df_sdi['sdi_m2'].max():.1f} m²")
    print(f"      peak efficiency = {df_sdi.loc[peak_idx, 'sdi_efficiency']:.2f} m²/m")
    print(f"      mean efficiency = {profile['mean_sdi_efficiency']:.2f} m²/m")


# ══════════════════════════════════════════════════════════════════════════════
# Feature 2 — Defensive Recovery
# ══════════════════════════════════════════════════════════════════════════════

def viz_recovery(df_sdi, runner_name: str) -> None:
    print(f"\n[22] Deformation Recovery — {runner_name}")

    rec = deformation_recovery(df_sdi)
    fig = plot_gravity_timeseries(
        df_sdi, metric_col="sdi_m2",
        label=f"SDI (m²) — {runner_name}",
        line_color="#ef4444",
    )

    # ── peak marker ───────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=[rec.peak_timestamp],
        y=[rec.peak_sdi_m2],
        mode="markers",
        marker=dict(symbol="star", size=14, color="gold",
                    line=dict(color="black", width=1)),
        name=f"Peak SDI = {rec.peak_sdi_m2:.1f} m²",
        hovertemplate=(
            f"Peak SDI<br>"
            f"t = {rec.peak_timestamp:.2f}s<br>"
            f"SDI = {rec.peak_sdi_m2:.1f} m²<extra></extra>"
        ),
    ))

    # ── half-life line ────────────────────────────────────────────────────────
    if rec.half_life_s is not None:
        half_ts   = rec.peak_timestamp + rec.half_life_s
        half_val  = rec.peak_sdi_m2 * 0.5
        fig.add_shape(
            type="line",
            x0=half_ts, x1=half_ts,
            y0=0, y1=half_val,
            line=dict(color="#3b82f6", width=1.5, dash="dash"),
        )
        fig.add_annotation(
            x=half_ts, y=half_val,
            text=f"  half-life<br>  +{rec.half_life_s:.1f}s",
            showarrow=False,
            font=dict(color="#3b82f6", size=11),
            xanchor="left",
        )

    # ── full recovery line ────────────────────────────────────────────────────
    if rec.full_recovery_s is not None:
        full_ts = rec.peak_timestamp + rec.full_recovery_s
        fig.add_vline(
            x=full_ts,
            line_width=1.5, line_dash="dot", line_color="#22c55e",
            annotation_text=f"Full recovery (+{rec.full_recovery_s:.1f}s)",
            annotation_position="top right",
            annotation_font=dict(color="#22c55e", size=10),
        )

    # ── 50% threshold reference line ─────────────────────────────────────────
    fig.add_hline(
        y=rec.peak_sdi_m2 * 0.5,
        line_width=1, line_dash="dot", line_color="rgba(59,130,246,0.4)",
    )

    # ── summary annotation ────────────────────────────────────────────────────
    hl_str  = f"{rec.half_life_s:.1f}s" if rec.half_life_s is not None else "—"
    fr_str  = f"{rec.full_recovery_s:.1f}s" if rec.full_recovery_s is not None else "—"
    rr_str  = f"{rec.recovery_rate_m2_per_s:.1f} m²/s"
    ann_text = (
        f"<b>Recovery Metrics</b><br>"
        f"Peak SDI: {rec.peak_sdi_m2:.1f} m² at t={rec.peak_timestamp:.1f}s<br>"
        f"Half-life (50%): {hl_str}<br>"
        f"Full recovery (10%): {fr_str}<br>"
        f"Recovery rate: {rr_str}"
    )
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.99, y=0.99,
        text=ann_text, showarrow=False, align="right",
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#94a3b8", borderwidth=1,
        font=dict(size=11), xanchor="right", yanchor="top",
    )
    fig.update_layout(title=f"Defensive Recovery — {runner_name}")
    _save(fig, "22_deformation_recovery")

    print(f"      peak SDI     = {rec.peak_sdi_m2:.1f} m²  at t={rec.peak_timestamp:.1f}s")
    print(f"      half-life    = {hl_str}")
    print(f"      full-recover = {fr_str}")
    print(f"      recovery rate= {rr_str}")


# ══════════════════════════════════════════════════════════════════════════════
# Feature 3 — Directional Flow Field
# ══════════════════════════════════════════════════════════════════════════════

def viz_flow_field(df_sdi, deformations, sequence: FrameSequence,
                   runner_name: str, id_to_name: dict) -> None:
    print(f"\n[23] Drag Vector Field at peak SDI — {runner_name}")

    peak_idx    = int(df_sdi["sdi_m2"].idxmax())
    peak_ts     = df_sdi.loc[peak_idx, "timestamp"]
    peak_deform = deformations[peak_idx]
    peak_frame  = min(sequence.frames, key=lambda f: abs(f.timestamp - peak_ts))

    # Compute vector field from deformation gradient
    flow = deformation_flow_field(peak_deform)

    # Deformation heatmap as background layer
    fig = plot_deformation_field(
        peak_deform, None,
        colorscale="RdBu",
        opacity=0.65,
        show_colorbar=True,
    )

    # Overlay flow vectors
    fig = plot_flow_field(
        flow, peak_frame,
        fig=fig,
        show_pitch=False,
        arrow_scale=20.0,
        min_magnitude=0.005,
        color="rgba(255,255,255,0.85)",
        home_team_id=HOME_TEAM_ID,
        player_names=id_to_name,
    )
    fig.update_layout(
        title=(
            f"Drag Vector Field — {runner_name}<br>"
            f"<sup>Arrows = direction space was dragged | "
            f"t = {peak_ts:.1f}s | SDI = {df_sdi.loc[peak_idx, 'sdi_m2']:.1f} m²</sup>"
        ),
    )
    _save(fig, "23_flow_field_peak")

    nonzero = int((flow.magnitudes >= 0.005).sum())
    print(f"      peak deformation at t={peak_ts:.1f}s")
    print(f"      {nonzero} active flow cells (magnitude ≥ 0.005)")


# ══════════════════════════════════════════════════════════════════════════════
# Feature 4 — Gravity Interaction Matrix
# ══════════════════════════════════════════════════════════════════════════════

def viz_interaction_matrix(sequence: FrameSequence, id_to_name: dict) -> None:
    print(f"\n[24] Gravity Interaction Matrix (home team)")

    t0 = time.perf_counter()
    time_window = (sequence.frames[-1].timestamp -
                   sequence.frames[0].timestamp + 1.0)

    df_matrix = gravity_interaction_matrix(
        sequence,
        attacking_team_id=HOME_TEAM_ID,
        time_window=time_window,
        resolution=RESOLUTION_FAST,      # lighter resolution for speed
    )
    elapsed = time.perf_counter() - t0
    print(f"      computed {len(df_matrix)} pairs in {elapsed:.1f}s")

    if df_matrix.empty:
        print("      (no pairs found — skipping)")
        return

    fig = plot_interaction_matrix(
        df_matrix,
        metric_col="total_nsg_m2",
        player_names=id_to_name,
        colorscale="Blues",
        title="Gravity Interaction Matrix — Atlético Madrid (Total NSG m²)",
    )
    _save(fig, "24_interaction_matrix")

    # Print top 5 mover→beneficiary pairs
    top5 = df_matrix.nlargest(5, "total_nsg_m2")
    print("      Top 5 mover → beneficiary pairs:")
    for _, row in top5.iterrows():
        m = id_to_name.get(row["mover_id"], row["mover_id"])
        b = id_to_name.get(row["beneficiary_id"], row["beneficiary_id"])
        print(f"        {m:20s} → {b:20s}  {row['total_nsg_m2']:.0f} m²")


# ══════════════════════════════════════════════════════════════════════════════
# Feature 5 — Vision-Aware Gravity vs Base
# ══════════════════════════════════════════════════════════════════════════════

def viz_vision_comparison(sequence: FrameSequence, runner_id: str,
                          runner_name: str) -> None:
    print(f"\n[25] Vision-Aware vs Base SDI — {runner_name}")

    time_window = (sequence.frames[-1].timestamp -
                   sequence.frames[0].timestamp + 1.0)

    # Base: standard KinematicControlModel
    t0 = time.perf_counter()
    df_base = spatial_drag_index(
        sequence, player_id=runner_id,
        attacking_team_id=HOME_TEAM_ID,
        time_window=time_window,
        resolution=RESOLUTION,
    )
    t_base = time.perf_counter() - t0

    # Vision-aware: VisionAwareControlModel (peripheral_penalty=0.3)
    vision_model = VisionAwareControlModel(
        pitch=PITCH,
        resolution=RESOLUTION,
        cone_half_angle=100.0,
        peripheral_penalty=0.3,   # defenders retain 30% control in blind spot
    )
    t0 = time.perf_counter()
    df_vision = spatial_drag_index(
        sequence, player_id=runner_id,
        attacking_team_id=HOME_TEAM_ID,
        time_window=time_window,
        resolution=RESOLUTION,
        control_model=vision_model,
    )
    t_vision = time.perf_counter() - t0

    # ── combined chart ────────────────────────────────────────────────────────
    fig = plot_gravity_timeseries(
        df_base, metric_col="sdi_m2",
        label=f"Base SDI (m²)",
        line_color="#94a3b8",
    )
    fig = plot_gravity_timeseries(
        df_vision, metric_col="sdi_m2",
        label="Vision-Aware SDI (m²)",
        line_color="#a855f7",
        fig=fig,
    )

    # Shade the difference region (vision > base = extra space due to blind spots)
    diff = df_vision["sdi_m2"].values - df_base["sdi_m2"].values
    ts   = df_base["timestamp"].values
    fig.add_trace(go.Scatter(
        x=np.concatenate([ts, ts[::-1]]),
        y=np.concatenate([
            np.maximum(df_vision["sdi_m2"].values, df_base["sdi_m2"].values),
            np.minimum(df_vision["sdi_m2"].values, df_base["sdi_m2"].values)[::-1],
        ]),
        fill="toself",
        fillcolor="rgba(168,85,247,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip",
        showlegend=False,
        name="Vision delta",
    ))

    mean_lift = float(np.mean(diff))
    peak_base   = df_base["sdi_m2"].max()
    peak_vision = df_vision["sdi_m2"].max()

    ann_text = (
        f"<b>Vision-Aware vs Base</b><br>"
        f"Base peak SDI:   {peak_base:.1f} m²<br>"
        f"Vision peak SDI: {peak_vision:.1f} m²<br>"
        f"Δ peak: {peak_vision - peak_base:+.1f} m²<br>"
        f"Mean Δ: {mean_lift:+.2f} m²<br>"
        f"peripheral_penalty = 0.3"
    )
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.99, y=0.99,
        text=ann_text, showarrow=False, align="right",
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#94a3b8", borderwidth=1,
        font=dict(size=11), xanchor="right", yanchor="top",
    )
    fig.update_layout(
        title=(
            f"Vision-Aware vs Base SDI — {runner_name}<br>"
            f"<sup>Purple = blind-spot amplification lifts attacker's effective SDI</sup>"
        ),
        legend=dict(x=0.01, y=0.99),
    )
    _save(fig, "25_vision_vs_base_sdi")

    print(f"      base SDI peak     = {peak_base:.1f} m²  ({t_base:.1f}s)")
    print(f"      vision SDI peak   = {peak_vision:.1f} m²  ({t_vision:.1f}s)")
    print(f"      mean lift         = {mean_lift:+.2f} m²")


# ══════════════════════════════════════════════════════════════════════════════
# Feature 3b — Flow Field Full Animation (time slider)
# ══════════════════════════════════════════════════════════════════════════════

def viz_flow_field_animation(
    df_sdi,
    deformations,
    sequence: FrameSequence,
    runner_name: str,
    id_to_name: dict,
    stride: int = 5,
) -> None:
    """Animated deformation heatmap + flow arrows over the full time window.

    Uses Plotly's animation API: every *stride*-th frame becomes one
    animation step.  A play button and time slider let the viewer scrub
    through the full run.  The pitch background is static; only the
    heatmap, arrows, and player markers are animated.
    """
    from pitch_aura.viz._pitch_draw import pitch_background

    print(f"\n[26] Flow Field Animation — {runner_name}  (stride={stride})")
    t0 = time.perf_counter()

    anim_indices = list(range(0, len(deformations), stride))
    n_anim = len(anim_indices)

    # Shared grid cell centres (same resolution for every frame)
    d0 = deformations[0]
    x_centers = (d0.x_edges[:-1] + d0.x_edges[1:]) / 2.0
    y_centers = (d0.y_edges[:-1] + d0.y_edges[1:]) / 2.0
    nx, ny = d0.values.shape

    # ── helpers ───────────────────────────────────────────────────────────────

    def _arrow_data(flow) -> tuple[list, list]:
        """Build Scatter x/y lists (segments separated by None)."""
        xc = (flow.x_edges[:-1] + flow.x_edges[1:]) / 2.0
        yc = (flow.y_edges[:-1] + flow.y_edges[1:]) / 2.0
        ax: list = []
        ay: list = []
        for i in range(nx):
            for j in range(ny):
                mag = float(flow.magnitudes[i, j])
                if mag < 0.005:
                    continue
                cx, cy = float(xc[i]), float(yc[j])
                vx = float(flow.vectors[i, j, 0]) * 20.0
                vy = float(flow.vectors[i, j, 1]) * 20.0
                ax += [cx, cx + vx, None]
                ay += [cy, cy + vy, None]
        return ax, ay

    def _player_data(frame_rec) -> tuple[list, list, list, list]:
        xs, ys, colors, texts = [], [], [], []
        for k, (pid, tid) in enumerate(zip(frame_rec.player_ids, frame_rec.team_ids)):
            xs.append(float(frame_rec.positions[k, 0]))
            ys.append(float(frame_rec.positions[k, 1]))
            colors.append("#3b82f6" if tid == HOME_TEAM_ID else "#ef4444")
            texts.append(id_to_name.get(pid, pid))
        return xs, ys, colors, texts

    # ── build base figure (pitch + placeholder animated traces) ───────────────

    fig = pitch_background(PITCH, bgcolor="#1a472a", line_color="white")
    n_static = len(fig.data)   # number of pitch traces — animated traces come after

    # First animation frame initialises the three animated traces
    flow0 = deformation_flow_field(deformations[anim_indices[0]])
    ax0, ay0 = _arrow_data(flow0)
    px0, py0, pc0, pt0 = _player_data(sequence.frames[anim_indices[0]])
    abs_max0 = max(float(np.abs(deformations[anim_indices[0]].values).max()), 1e-6)

    fig.add_trace(go.Heatmap(                           # trace n_static
        x=x_centers, y=y_centers,
        z=deformations[anim_indices[0]].values.T,
        colorscale="RdBu",
        zmin=-abs_max0, zmax=abs_max0, zmid=0.0,
        opacity=0.65, showscale=True,
        colorbar=dict(title="ΔControl", thickness=12, len=0.6, x=1.01),
        hovertemplate="x: %{x:.1f}<br>y: %{y:.1f}<br>Δ: %{z:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(                           # trace n_static + 1
        x=ax0, y=ay0,
        mode="lines",
        line=dict(color="rgba(255,255,255,0.8)", width=1.5),
        hoverinfo="skip", showlegend=False, name="drag direction",
    ))
    fig.add_trace(go.Scatter(                           # trace n_static + 2
        x=px0, y=py0,
        mode="markers+text",
        text=pt0, textposition="top center",
        textfont=dict(size=8, color="white"),
        marker=dict(size=8, color=pc0, line=dict(color="white", width=1)),
        hoverinfo="text", showlegend=False, name="players",
    ))

    animated_traces = [n_static, n_static + 1, n_static + 2]

    # ── build Plotly animation frames ─────────────────────────────────────────

    plotly_frames = []
    slider_steps = []

    for idx in anim_indices:
        deform = deformations[idx]
        flow   = deformation_flow_field(deform)
        ax, ay = _arrow_data(flow)
        px, py, pc, pt = _player_data(sequence.frames[idx])

        sdi_val  = float(df_sdi.loc[idx, "sdi_m2"]) if idx < len(df_sdi) else 0.0
        eff_val  = float(df_sdi.loc[idx, "sdi_efficiency"]) if idx < len(df_sdi) else 0.0
        ts       = deform.timestamp
        abs_max  = max(float(np.abs(deform.values).max()), 1e-6)

        plotly_frames.append(go.Frame(
            data=[
                go.Heatmap(
                    z=deform.values.T,
                    zmin=-abs_max, zmax=abs_max,
                ),
                go.Scatter(x=ax, y=ay),
                go.Scatter(
                    x=px, y=py,
                    marker=dict(color=pc),
                    text=pt,
                ),
            ],
            traces=animated_traces,
            name=str(idx),
            layout=go.Layout(
                title_text=(
                    f"Drag Vector Field — {runner_name} | "
                    f"t = {ts:.1f}s | SDI = {sdi_val:.1f} m² | "
                    f"efficiency = {eff_val:.2f} m²/m"
                )
            ),
        ))

        slider_steps.append({
            "args": [[str(idx)], {
                "frame": {"duration": 120, "redraw": True},
                "mode": "immediate",
                "transition": {"duration": 0},
            }],
            "label": f"{ts:.1f}s",
            "method": "animate",
        })

    fig.frames = plotly_frames

    # ── layout: play button + slider ──────────────────────────────────────────

    fig.update_layout(
        title=(
            f"Deformation Field & Drag Vectors — {runner_name}<br>"
            f"<sup>Blue = attacker gained | Red = defender freed | "
            f"Arrows = gradient direction of deformation</sup>"
        ),
        updatemenus=[{
            "buttons": [
                {
                    "args": [None, {
                        "frame": {"duration": 120, "redraw": True},
                        "fromcurrent": True,
                        "transition": {"duration": 0},
                    }],
                    "label": "▶ Play",
                    "method": "animate",
                },
                {
                    "args": [[None], {
                        "frame": {"duration": 0, "redraw": False},
                        "mode": "immediate",
                        "transition": {"duration": 0},
                    }],
                    "label": "⏸ Pause",
                    "method": "animate",
                },
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 70},
            "showactive": False,
            "type": "buttons",
            "x": 0.1, "xanchor": "right",
            "y": 0, "yanchor": "top",
        }],
        sliders=[{
            "active": 0,
            "steps": slider_steps,
            "x": 0.1, "len": 0.9,
            "pad": {"b": 10, "t": 50},
            "currentvalue": {
                "prefix": "Time: ",
                "visible": True,
                "xanchor": "center",
                "font": {"color": "white", "size": 13},
            },
            "transition": {"duration": 0},
            "bgcolor": "rgba(255,255,255,0.1)",
            "bordercolor": "rgba(255,255,255,0.3)",
            "tickcolor": "rgba(255,255,255,0.5)",
            "font": {"color": "rgba(255,255,255,0.7)", "size": 9},
        }],
        paper_bgcolor="#111111",
        font=dict(color="white"),
    )

    _save(fig, "26_flow_field_animation")
    elapsed = time.perf_counter() - t0
    print(f"      {n_anim} animation frames (stride={stride})  built in {elapsed:.1f}s")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 65)
    print("PitchAura — Gravity Extensions  (Atlético vs Real Madrid)")
    print("=" * 65)

    # ── load ──────────────────────────────────────────────────────────────────
    print("\nLoading 200 frames from period 1 (offset 100) …")
    t0 = time.perf_counter()
    meta           = load_match_meta()
    _, id_to_name  = build_player_lookup(meta)
    raw_frames     = load_frames(meta, n_frames=200, period=1, start_offset=100)
    frames         = add_velocities(raw_frames)
    sequence       = make_sequence(frames)
    print(f"  {len(frames)} frames  |  "
          f"{frames[0].timestamp:.1f}s → {frames[-1].timestamp:.1f}s  |  "
          f"loaded in {time.perf_counter()-t0:.2f}s")

    # ── runner selection ──────────────────────────────────────────────────────
    runner_id, runner_name = pick_runner(frames, id_to_name)
    print(f"  runner: {runner_name}  (id={runner_id})")

    time_window = frames[-1].timestamp - frames[0].timestamp + 1.0

    # ── shared SDI computation (with deformations) ────────────────────────────
    print(f"\nComputing SDI+deformations ({RESOLUTION}, {len(frames)} frames) …")
    t0 = time.perf_counter()
    df_sdi, deformations = spatial_drag_index(
        sequence,
        player_id=runner_id,
        attacking_team_id=HOME_TEAM_ID,
        time_window=time_window,
        resolution=RESOLUTION,
        return_deformation=True,
    )
    print(f"  done in {time.perf_counter()-t0:.2f}s  |  {len(df_sdi)} frames")

    # ── visualisations ────────────────────────────────────────────────────────
    viz_efficiency(df_sdi, runner_name, sequence, runner_id)
    viz_recovery(df_sdi, runner_name)
    viz_flow_field(df_sdi, deformations, sequence, runner_name, id_to_name)
    viz_interaction_matrix(sequence, id_to_name)
    viz_vision_comparison(sequence, runner_id, runner_name)
    viz_flow_field_animation(df_sdi, deformations, sequence, runner_name, id_to_name, stride=5)

    print("\n" + "=" * 65)
    print(f"All outputs → {OUT_DIR.relative_to(REPO_ROOT)}/")
    print("=" * 65)


if __name__ == "__main__":
    main()
