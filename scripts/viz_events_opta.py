"""
Project: PitchAura
File Created: 2026-02-26
Author: Xingnan Zhu
File Name: viz_events_opta.py
Description:
    Event-based spatial analysis demo using the Sunderland 3-0 West Ham
    Opta JSON dataset (JSONP wrapper format).

    Opta 坐标系：0–100 百分比 → 缩放到 105×68m
    Opta 无 freeze frame，因此本脚本演示 Layer 1 的全部功能：

      1. progressive_actions()  → plot_progressive_passes()（进攻性传球）
      2. passing_network()      → plot_passing_network()（传球网络）
      3. zone_counts()          → plot_event_zones()（区域事件分布）
      4. event_density()        → plot_heatmap()（KDE 热力图）

    Opta 主要 typeId：
      1=Pass, 13=Miss, 14=Post, 15=Attempt Saved, 16=Goal,
      4=Foul, 7=Tackle, 3=Take On, 12=Clearance

    Run:
        uv run python scripts/viz_events_opta.py

    Output HTML → scripts/output/
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = (
    REPO_ROOT
    / "data"
    / "Sunderland_EPL_2025_26"
    / "20250816_Sunderland_3_0_West_Ham.json"
)
OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO_ROOT / "src"))

from pitch_aura.types import EventRecord, PitchSpec
from pitch_aura.events.progressive import progressive_actions
from pitch_aura.events.passing_network import passing_network
from pitch_aura.events.zones import zone_counts, event_density
from pitch_aura.viz.events import (
    plot_passing_network,
    plot_progressive_passes,
    plot_event_zones,
)
from pitch_aura.viz.heatmap import plot_heatmap

# ── 常量 ───────────────────────────────────────────────────────────────────────
PITCH = PitchSpec(length=105.0, width=68.0, origin="bottom_left")
# Opta 坐标是 0–100 百分比，缩放因子
SCALE_X = 105.0 / 100.0
SCALE_Y = 68.0 / 100.0

# Opta typeId → 可读名称
OPTA_TYPE = {
    1: "Pass", 2: "Offside Pass", 3: "Take On", 4: "Foul", 5: "Out",
    6: "Corner", 7: "Tackle", 8: "Interception", 9: "Turnover",
    10: "Save", 12: "Clearance", 13: "Miss", 14: "Post",
    15: "Attempt Saved", 16: "Goal", 17: "Card",
    37: "Ball Recovery", 42: "Block", 44: "Keeper Pick Up",
    49: "Ball Touch", 61: "Goal Kick", 67: "Aerial",
}

# Opta qualifier IDs
Q_END_X  = 140   # 传球终点 x
Q_END_Y  = 141   # 传球终点 y
Q_CROSS  = 2     # 是否为 cross（值为 1 时）
Q_CORNER = 5     # corner 旗
Q_THRU   = 72    # Throughball 类型（值含 "Throughball"）
Q_HEAD   = 72    # —— 同字段，根据值判断


# ══════════════════════════════════════════════════════════════════════════════
# 解析工具
# ══════════════════════════════════════════════════════════════════════════════

def _load_json(path: Path) -> dict:
    """读取 Opta JSONP 文件，剥离 W<hash>(...) 包装."""
    raw = path.read_text(encoding="utf-8-sig")
    m = re.match(r'^W[a-f0-9]+\((.+)\)$', raw.strip(), re.DOTALL)
    return json.loads(m.group(1) if m else raw)


def _get_qual(event: dict, qualifier_id: int) -> str | None:
    """从事件中取指定 qualifierId 的值（无该 qualifier 时返回 None）."""
    for q in event.get("qualifier", []):
        if q["qualifierId"] == qualifier_id:
            return q.get("value")
    return None


def _has_qual(event: dict, qualifier_id: int) -> bool:
    """检查事件是否含某 qualifier（无论值是什么）."""
    return any(q["qualifierId"] == qualifier_id for q in event.get("qualifier", []))


def build_events(raw_events: list[dict]) -> list[EventRecord]:
    """把 Opta 原始事件列表转为 EventRecord 列表.

    - 坐标从 0–100 百分比缩放到 105×68m
    - 传球终点从 qualifier 140/141 提取
    - result：outcome=1 → "complete"，outcome=0 → "incomplete"
    - qualifiers：收集语义标签（cross、throughball 等）
    """
    records: list[EventRecord] = []

    for ev in raw_events:
        type_id   = ev.get("typeId", 0)
        event_type = OPTA_TYPE.get(type_id, f"type_{type_id}")
        period    = ev.get("periodId", 1)
        ts        = ev.get("timeMin", 0) * 60.0 + ev.get("timeSec", 0.0)
        player_id = ev.get("playerId")
        team_id   = ev.get("contestantId")
        outcome   = ev.get("outcome", -1)

        # 起点坐标
        raw_x = ev.get("x")
        raw_y = ev.get("y")
        coords: np.ndarray | None = None
        if raw_x is not None and raw_y is not None:
            coords = np.array([raw_x * SCALE_X, raw_y * SCALE_Y])

        # 终点坐标（传球专用）
        end_coords: np.ndarray | None = None
        if type_id == 1:  # Pass
            ex = _get_qual(ev, Q_END_X)
            ey = _get_qual(ev, Q_END_Y)
            if ex is not None and ey is not None:
                end_coords = np.array([float(ex) * SCALE_X, float(ey) * SCALE_Y])

        # result
        result: str | None = None
        if outcome == 1:
            result = "complete"
        elif outcome == 0:
            result = "incomplete"

        # qualifiers
        qual_tags: list[str] = []
        if _has_qual(ev, Q_CROSS):
            qual_tags.append("CROSS")
        thru_val = _get_qual(ev, Q_THRU)
        if thru_val and "throughball" in thru_val.lower():
            qual_tags.append("THROUGH_BALL")
        if _has_qual(ev, Q_CORNER):
            qual_tags.append("CORNER")

        records.append(EventRecord(
            timestamp=ts,
            period=period,
            event_type=event_type,
            player_id=str(player_id) if player_id else None,
            team_id=str(team_id) if team_id else None,
            coordinates=coords,
            end_coordinates=end_coords,
            result=result,
            qualifiers=tuple(qual_tags),
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
    print("=" * 62)
    print("PitchAura · Event Spatial Analysis Demo (Opta)")
    print("Sunderland 3 – 0 West Ham United  |  EPL 2025/26")
    print("=" * 62)

    # ── 1. 加载数据 ────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    data = _load_json(DATA_FILE)

    # 队伍信息
    contestants = {
        c["id"]: c["name"]
        for c in data["matchInfo"]["contestant"]
    }
    TEAM_SUN = next(
        c["id"] for c in data["matchInfo"]["contestant"] if c["position"] == "home"
    )
    TEAM_WHU = next(
        c["id"] for c in data["matchInfo"]["contestant"] if c["position"] == "away"
    )

    raw_events = data["liveData"]["event"]
    events = build_events(raw_events)
    t1 = time.perf_counter()

    print(f"\n已加载 {len(events)} 个事件  [{t1-t0:.2f}s]")
    sun_evs = [e for e in events if e.team_id == TEAM_SUN]
    whu_evs = [e for e in events if e.team_id == TEAM_WHU]
    print(f"  Sunderland:      {len(sun_evs)} 个")
    print(f"  West Ham United: {len(whu_evs)} 个")

    # Opta 攻方向：home 队攻 x 从小到大（0→100），away 队相反
    # Sunderland 进攻方向 target_x=105，West Ham 进攻方向 target_x=0
    TARGET = {TEAM_SUN: 105.0, TEAM_WHU: 0.0}

    # ══════════════════════════════════════════════════════════════════════════
    # 2. Progressive Actions — 仅传球（Opta 无 carry 事件）
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[1/4] Progressive Actions (传球) ...")
    for team_id, fname in [
        (TEAM_SUN, "opta_01_progressive_sun.html"),
        (TEAM_WHU, "opta_01_progressive_whu.html"),
    ]:
        team_name = contestants[team_id]
        team_evs = [e for e in events if e.team_id == team_id]
        df_prog = progressive_actions(
            team_evs,
            pitch=PITCH,
            event_types=("Pass",),
            target_x=TARGET[team_id],
        )
        n_prog = int(df_prog["is_progressive"].sum())
        n_total = len(df_prog)
        pct = n_prog / n_total * 100 if n_total else 0
        print(f"  {team_name}: {n_total} 次传球含终点坐标，推进性 {n_prog} 次（{pct:.0f}%）")

        fig = plot_progressive_passes(df_prog, pitch=PITCH)
        fig.update_layout(
            title=f"{team_name} — 推进性传球（绿色=推进，灰色=非推进）",
            title_font_size=14,
        )
        save(fig, fname)

    # ══════════════════════════════════════════════════════════════════════════
    # 3. Passing Network
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[2/4] Passing Networks ...")
    for team_id, color, fname in [
        (TEAM_SUN, "#e74c3c", "opta_02_network_sun.html"),
        (TEAM_WHU, "#3498db", "opta_02_network_whu.html"),
    ]:
        team_name = contestants[team_id]
        team_evs = [e for e in events if e.team_id == team_id]
        net = passing_network(team_evs, team_id=team_id, event_type="Pass", min_passes=2)
        print(f"  {team_name}: {len(net.nodes)} 节点，{len(net.edges)} 边")

        fig = plot_passing_network(net, pitch=PITCH, color=color, node_size_scale=1.3)
        fig.update_layout(
            title=f"{team_name} — 空间传球网络",
            title_font_size=14,
        )
        save(fig, fname)

    # ══════════════════════════════════════════════════════════════════════════
    # 4. Zone Counts
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[3/4] Zone Counts ...")
    for team_id, fname in [
        (TEAM_SUN, "opta_03_zones_sun.html"),
        (TEAM_WHU, "opta_03_zones_whu.html"),
    ]:
        team_name = contestants[team_id]
        team_evs = [
            e for e in events
            if e.team_id == team_id
            and e.event_type in ("Pass", "Tackle", "Interception", "Clearance")
        ]
        df_zones = zone_counts(team_evs, pitch=PITCH, nx=6, ny=4)
        print(
            f"  {team_name}: {df_zones['count'].sum()} 个事件"
            f"（传球+铲球+拦截+解围）分布在 6×4 区域"
        )

        fig = plot_event_zones(df_zones, pitch=PITCH, colorscale="Blues")
        fig.update_layout(
            title=f"{team_name} — 传球+铲球+拦截+解围区域分布",
            title_font_size=14,
        )
        save(fig, fname)

    # ══════════════════════════════════════════════════════════════════════════
    # 5. Event Density — 传球密度 KDE（区分完成/未完成）
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[4/4] Event Density (KDE) ...")
    for team_id, fname_ok, fname_fail in [
        (TEAM_SUN,
         "opta_04_density_sun_complete.html",
         "opta_04_density_sun_incomplete.html"),
        (TEAM_WHU,
         "opta_04_density_whu_complete.html",
         "opta_04_density_whu_incomplete.html"),
    ]:
        team_name = contestants[team_id]
        team_passes = [e for e in events if e.team_id == team_id and e.event_type == "Pass"]
        complete   = [e for e in team_passes if e.result == "complete"]
        incomplete = [e for e in team_passes if e.result == "incomplete"]

        for evs_sub, label, fname, cscale in [
            (complete,   "完成传球", fname_ok,   "YlGn"),
            (incomplete, "丢失传球", fname_fail, "OrRd"),
        ]:
            grid = event_density(evs_sub, pitch=PITCH, resolution=(60, 40), sigma=5.0)
            fig  = plot_heatmap(grid, colorscale=cscale, opacity=0.8,
                                bgcolor="#1a472a", line_color="white")
            fig.update_layout(
                title=f"{team_name} — {label}密度（{len(evs_sub)} 次）",
                title_font_size=14,
            )
            save(fig, fname)

    # ── 完成 ────────────────────────────────────────────────────────────────────
    t_total = time.perf_counter() - t0
    print(f"\n{'='*62}")
    print(f"全部完成！输出目录: scripts/output/  [{t_total:.1f}s]")
    print(f"{'='*62}")


if __name__ == "__main__":
    main()
