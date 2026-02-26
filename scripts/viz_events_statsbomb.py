"""
Project: PitchAura
File Created: 2026-02-26
Author: Xingnan Zhu
File Name: viz_events_statsbomb.py
Description:
    Event-based spatial analysis demo using the Atlético Madrid vs Real Madrid
    StatsBomb CSV dataset.  Demonstrates all five new event analysis functions:

      1. progressive_actions()  → plot_progressive_passes()
      2. passing_network()      → plot_passing_network()
      3. zone_counts()          → plot_event_zones()
      4. event_density()        → plot_heatmap()
      5. batch_event_control()  → plot_pitch_control()  (freeze frames from shots)

    StatsBomb coordinate system: 120 × 80 (origin = bottom-left).
    Both teams attack left→right in their own half (StatsBomb normalises
    all events so each team always attacks toward x=120).

    Run:
        uv run python scripts/viz_events_statsbomb.py

    Output HTML files → scripts/output/
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = (
    REPO_ROOT
    / "data"
    / "Atlético Madrid_Real Madrid"
    / "Atlético Madrid_Real Madrid_4007332.csv"
)
OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO_ROOT / "src"))

# ── pitch_aura imports ─────────────────────────────────────────────────────────
from pitch_aura.types import EventRecord, FrameRecord, PitchSpec
from pitch_aura.events.progressive import progressive_actions
from pitch_aura.events.passing_network import passing_network
from pitch_aura.events.zones import zone_counts, event_density
from pitch_aura.events.snapshot import batch_event_control
from pitch_aura.viz.events import (
    plot_passing_network,
    plot_progressive_passes,
    plot_event_zones,
)
from pitch_aura.viz.heatmap import plot_heatmap

# ── match constants ────────────────────────────────────────────────────────────
TEAM_ATM = "212"   # Atlético Madrid
TEAM_RMA = "220"   # Real Madrid
PITCH = PitchSpec(length=120.0, width=80.0, origin="bottom_left")


# ══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════════════

def _parse_timestamp(ts: str) -> float:
    """把 '00:43:21.450' 或 '00:43' 等格式转为秒数."""
    parts = str(ts).split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])


def _build_freeze_frame(
    ff_rows: pd.DataFrame,
    event_row: pd.Series,
    timestamp: float,
    period: int,
    event_team_id: str,
) -> FrameRecord | None:
    """把长表格式的 freeze_frame 行转为 FrameRecord."""
    ff_valid = ff_rows.dropna(subset=["freeze_frame_x", "freeze_frame_y"])
    if ff_valid.empty:
        return None

    player_ids: list[str] = []
    team_ids: list[str] = []
    positions: list[list[float]] = []

    for _, row in ff_valid.iterrows():
        pid = str(int(row["freeze_frame_player_id"]))
        # StatsBomb: freeze_frame_teammate=True → 同队; False → 对手
        is_teammate = bool(row["freeze_frame_teammate"])
        tid = event_team_id if is_teammate else (
            TEAM_RMA if event_team_id == TEAM_ATM else TEAM_ATM
        )
        player_ids.append(pid)
        team_ids.append(tid)
        positions.append([float(row["freeze_frame_x"]), float(row["freeze_frame_y"])])

    if not positions:
        return None

    # 球的位置 = 事件发生位置
    bx = event_row.get("location_x", np.nan)
    by = event_row.get("location_y", np.nan)
    ball_pos = np.array([
        float(bx) if pd.notna(bx) else np.nan,
        float(by) if pd.notna(by) else np.nan,
    ])

    return FrameRecord(
        timestamp=timestamp,
        period=period,
        ball_position=ball_pos,
        player_ids=player_ids,
        team_ids=team_ids,
        positions=np.array(positions, dtype=np.float64),
        velocities=None,
        is_goalkeeper=np.zeros(len(player_ids), dtype=bool),
    )


def load_events(df: pd.DataFrame) -> list[EventRecord]:
    """从 StatsBomb CSV 构建 EventRecord 列表（一行=一个事件）.

    StatsBomb 的 freeze frame 是长表（每个球员一行，共用同一个 event_id），
    这里先按 event_id 聚合 freeze frame 行，再为每个唯一事件建立 EventRecord.
    """
    # 分离 freeze frame 行（同一 id 的多行）
    # 正常事件行：freeze_frame_player_id 为空，或者是第一次出现该 id
    # freeze frame 行：freeze_frame_player_id 非空

    # 先把所有 id 唯一的事件行取出
    # （StatsBomb CSV 中，第一次出现某 event_id 的行是事件本身，后续行是 freeze frame）
    event_rows = df.groupby("id", sort=False).first().reset_index()

    # 构建 freeze_frame 的 group dict: id → sub-dataframe
    ff_groups = (
        df[df["freeze_frame_x"].notna()]
        .groupby("id", sort=False)
    )

    records: list[EventRecord] = []

    for _, row in event_rows.iterrows():
        event_id = row["id"]
        ts = _parse_timestamp(row["timestamp"])
        period = int(row["period"])
        event_type = str(row["event_type_name"])
        player_id = str(int(row["player_id"])) if pd.notna(row.get("player_id")) else None
        team_id = str(int(row["team_id"])) if pd.notna(row.get("team_id")) else None

        # 起点坐标
        coords: np.ndarray | None = None
        if pd.notna(row.get("location_x")) and pd.notna(row.get("location_y")):
            coords = np.array([float(row["location_x"]), float(row["location_y"])])

        # 终点坐标
        end_coords: np.ndarray | None = None
        if pd.notna(row.get("end_location_x")) and pd.notna(row.get("end_location_y")):
            end_coords = np.array([
                float(row["end_location_x"]), float(row["end_location_y"]),
            ])

        # Result: outcome_name 为空 → complete（Pass）
        result: str | None = None
        if event_type == "Pass":
            raw_outcome = row.get("outcome_name")
            result = str(raw_outcome).lower() if pd.notna(raw_outcome) else "complete"
        elif pd.notna(row.get("outcome_name")):
            result = str(row["outcome_name"]).lower()

        # Qualifiers: 把 True 的 boolean 列收集成 tuple
        QUALIFIER_COLS = [
            "pass_cross", "pass_cut_back", "pass_switch", "through_ball",
            "under_pressure", "counterpress",
        ]
        qualifiers = tuple(
            col.upper()
            for col in QUALIFIER_COLS
            if str(row.get(col, "")).lower() == "true"
        )

        # Freeze frame（仅 Shot 事件有）
        freeze_frame: FrameRecord | None = None
        if event_id in ff_groups.groups and team_id is not None:
            ff_df = ff_groups.get_group(event_id)
            freeze_frame = _build_freeze_frame(ff_df, row, ts, period, team_id)

        records.append(EventRecord(
            timestamp=ts,
            period=period,
            event_type=event_type,
            player_id=player_id,
            team_id=team_id,
            coordinates=coords,
            end_coordinates=end_coords,
            result=result,
            qualifiers=qualifiers,
            freeze_frame=freeze_frame,
        ))

    return records


def save(fig, name: str) -> None:
    path = OUT_DIR / name
    fig.write_html(str(path))
    print(f"  ✓ saved → {path.relative_to(REPO_ROOT)}")


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 60)
    print("PitchAura · Event Spatial Analysis Demo")
    print("Atlético Madrid 0 – 1 Real Madrid  |  La Liga 2024/25")
    print("=" * 60)

    # ── 1. 读取数据 ────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    df = pd.read_csv(DATA_FILE)
    events = load_events(df)
    t1 = time.perf_counter()

    n_with_ff = sum(1 for e in events if e.freeze_frame is not None)
    print(f"\n已加载 {len(events)} 个事件（{n_with_ff} 个含 freeze frame）  [{t1-t0:.2f}s]")

    atm_events = [e for e in events if e.team_id == TEAM_ATM]
    rma_events = [e for e in events if e.team_id == TEAM_RMA]
    print(f"  马德里竞技 {len(atm_events)} 个  |  皇家马德里 {len(rma_events)} 个")

    # ══════════════════════════════════════════════════════════════════════════
    # 2. Progressive Actions — 马竞的推进性传球 + carry
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[1/5] Progressive Actions ...")
    for team_id, label, fname in [
        (TEAM_ATM, "Atlético Madrid", "events_01_progressive_atm.html"),
        (TEAM_RMA, "Real Madrid",     "events_01_progressive_rma.html"),
    ]:
        team_evs = [e for e in events if e.team_id == team_id]
        df_prog = progressive_actions(
            team_evs,
            pitch=PITCH,
            event_types=("Pass", "Carries"),
            target_x=120.0,   # 进攻方向球门线
        )
        n_prog = df_prog["is_progressive"].sum()
        print(f"  {label}: {len(df_prog)} 个动作，其中推进性 {n_prog} 个")

        fig = plot_progressive_passes(
            df_prog,
            pitch=PITCH,
            color_progressive="#22c55e",
            color_other="#94a3b8",
        )
        fig.update_layout(
            title=f"{label} — 推进性动作（绿色=推进，灰色=非推进）",
            title_font_size=14,
        )
        save(fig, fname)

    # ══════════════════════════════════════════════════════════════════════════
    # 3. Passing Network — 马竞的传球网络
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[2/5] Passing Networks ...")
    for team_id, label, color, fname in [
        (TEAM_ATM, "Atlético Madrid", "#e74c3c", "events_02_network_atm.html"),
        (TEAM_RMA, "Real Madrid",     "#3498db", "events_02_network_rma.html"),
    ]:
        team_evs = [e for e in events if e.team_id == team_id]
        net = passing_network(team_evs, team_id=team_id, event_type="Pass", min_passes=2)
        print(f"  {label}: {len(net.nodes)} 节点，{len(net.edges)} 边")

        fig = plot_passing_network(net, pitch=PITCH, color=color, node_size_scale=1.2)
        fig.update_layout(
            title=f"{label} — 空间传球网络（节点大小=传球次数，边宽=连线频率）",
            title_font_size=14,
        )
        save(fig, fname)

    # ══════════════════════════════════════════════════════════════════════════
    # 4. Zone Counts — 全场事件区域分布
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[3/5] Zone Counts ...")
    for team_id, label, fname in [
        (TEAM_ATM, "Atlético Madrid", "events_03_zones_atm.html"),
        (TEAM_RMA, "Real Madrid",     "events_03_zones_rma.html"),
    ]:
        team_evs = [
            e for e in events
            if e.team_id == team_id and e.event_type in ("Pass", "Carries", "Shot")
        ]
        df_zones = zone_counts(team_evs, pitch=PITCH, nx=6, ny=4)
        print(f"  {label}: 共 {df_zones['count'].sum()} 个事件分布在 6×4 区域")

        fig = plot_event_zones(df_zones, pitch=PITCH, colorscale="YlOrRd")
        fig.update_layout(
            title=f"{label} — 传球/带球/射门区域分布（传球+带球+射门）",
            title_font_size=14,
        )
        save(fig, fname)

    # ══════════════════════════════════════════════════════════════════════════
    # 5. Event Density — KDE 热力图（传球密度）
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[4/5] Event Density (KDE) ...")
    for team_id, label, fname in [
        (TEAM_ATM, "Atlético Madrid", "events_04_density_atm.html"),
        (TEAM_RMA, "Real Madrid",     "events_04_density_rma.html"),
    ]:
        team_evs = [
            e for e in events
            if e.team_id == team_id and e.event_type == "Pass"
        ]
        grid = event_density(
            team_evs, pitch=PITCH, resolution=(60, 40), sigma=6.0,
        )
        fig = plot_heatmap(
            grid,
            colorscale="Hot",
            opacity=0.8,
            bgcolor="#1a472a",
            line_color="white",
        )
        fig.update_layout(
            title=f"{label} — 传球密度热力图（高斯核 σ=6m）",
            title_font_size=14,
        )
        save(fig, fname)

    # ══════════════════════════════════════════════════════════════════════════
    # 6. Freeze Frame Pitch Control — 射门瞬间的空间控制
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[5/5] Freeze Frame Pitch Control (shots) ...")

    # 分别处理两队的射门 freeze frame
    from pitch_aura.viz import plot_pitch_control

    for team_id, label in [
        (TEAM_ATM, "Atlético Madrid"),
        (TEAM_RMA, "Real Madrid"),
    ]:
        shot_evs = [
            e for e in events
            if e.event_type == "Shot"
            and e.team_id == team_id
            and e.freeze_frame is not None
        ]
        if not shot_evs:
            print(f"  {label}: 无 freeze frame 射门，跳过")
            continue

        results = batch_event_control(shot_evs, team_id=team_id, pitch=PITCH)
        print(f"  {label}: 共 {len(results)} 个射门 freeze frame")

        for i, (ev, grid) in enumerate(results[:3]):   # 只保存前 3 个
            minute = int(ev.timestamp // 60)
            fig = plot_pitch_control(
                grid,
                home_team_id=team_id,
                colorscale="RdBu_r" if team_id == TEAM_ATM else "RdBu",
            )
            # 叠加 freeze frame 球员位置
            ff = ev.freeze_frame
            if ff is not None:
                from pitch_aura.viz.players import plot_players
                fig = plot_players(
                    ff,
                    fig=fig,
                    home_team_id=team_id,
                    home_color="#e74c3c" if team_id == TEAM_ATM else "#3498db",
                    away_color="#3498db" if team_id == TEAM_ATM else "#e74c3c",
                    show_velocity=False,
                    show_pitch=False,
                )

            tag = "atm" if team_id == TEAM_ATM else "rma"
            fname = f"events_05_freeze_frame_{tag}_{i+1:02d}_min{minute:03d}.html"
            fig.update_layout(
                title=f"{label} — 射门 freeze frame Pitch Control（第 {minute}' 分）",
                title_font_size=14,
            )
            save(fig, fname)

    # ── 完成 ────────────────────────────────────────────────────────────────────
    t_total = time.perf_counter() - t0
    print(f"\n{'='*60}")
    print(f"全部完成！输出目录: scripts/output/  [{t_total:.1f}s]")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
