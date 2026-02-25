# PitchAura

**A spatial analytics computation engine for football/soccer tracking data.**

[![PyPI version](https://img.shields.io/pypi/v/pitch-aura.svg)](https://pypi.org/project/pitch-aura/)
[![Python](https://img.shields.io/pypi/pyversions/pitch-aura.svg)](https://pypi.org/project/pitch-aura/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

PitchAura transforms raw player tracking coordinates into spatial control matrices, Voronoi territories, tactical metrics, and cognitive vision models — built on NumPy/SciPy with no mandatory heavy dependencies.

---

## Features

| Module | What it does |
|--------|-------------|
| **Space / Pitch Control** | Kinematic control model (Spearman 2018) + Voronoi tessellation |
| **Tactics** | Space creation, passing lane lifespan, defensive line-breaking pockets |
| **Cognitive** | Player vision cones, blind-spot maps, `VisionModel` |
| **Sync** | Frame alignment across tracking streams + moving-average / Kalman filtering |
| **I/O** | kloppy adapter (`from_tracking`, `from_events`) |
| **Viz** | Plotly-based pitch control heatmaps, Voronoi plots, animations |

---

## Installation

```bash
pip install pitch-aura
```

### Optional extras

```bash
# kloppy I/O adapter
pip install "pitch-aura[kloppy]"

# Plotly visualisation
pip install "pitch-aura[viz]"

# Everything
pip install "pitch-aura[kloppy,viz]"
```

**Runtime requirements:** Python ≥ 3.12, NumPy ≥ 1.24, SciPy ≥ 1.10, pandas ≥ 2.0.

---

## Quick Start

### Build a FrameRecord manually

```python
import numpy as np
import pitch_aura as pa

frame = pa.FrameRecord(
    frame_id=1,
    timestamp=0.04,
    home_positions=np.array([[30.0, 0.0], [20.0, 10.0]]),   # (N, 2) metres
    away_positions=np.array([[-25.0, 5.0], [-15.0, -8.0]]),
    home_velocities=np.array([[2.0, 0.5], [1.0, -1.0]]),
    away_velocities=np.array([[-1.5, 0.0], [-2.0, 1.0]]),
    home_team_id="home",
    ball_position=np.array([5.0, 0.0]),
)
```

### Kinematic pitch control

```python
model = pa.KinematicControlModel()
grid = model.compute(frame)          # returns ProbabilityGrid
print(grid.values.shape)             # (68, 105) by default
```

### Voronoi territories

```python
voronoi = pa.VoronoiModel()
result = voronoi.compute(frame)      # returns VoronoiResult
# result.regions: list of (player_idx, team, polygon_vertices) tuples
```

### Load from kloppy

```python
# pip install "pitch-aura[kloppy]"
import kloppy
dataset = kloppy.tracab.load(...)

sequence = pa.from_tracking(dataset)   # FrameSequence
frame = sequence.frames[0]             # FrameRecord
```

---

## Tactics

```python
from pitch_aura.tactics.space_creation import space_creation
from pitch_aura.tactics.passing_lanes import passing_lane_lifespan
from pitch_aura.tactics.line_breaking import line_breaking_pockets

# How much space each attacking player is creating
scores = space_creation(frame, model)

# How many frames a passing lane stays open
lifespan = passing_lane_lifespan(sequence, passer_idx=7, receiver_idx=9, model=model)

# Pockets in the defensive line
pockets = line_breaking_pockets(frame)
for p in pockets:
    print(f"Pocket at depth {p.line_depth:.1f}m, y={p.y_left:.1f}–{p.y_right:.1f}m")
```

---

## Cognitive Models

```python
from pitch_aura.cognitive.vision_cone import player_heading, vision_cone_mask
from pitch_aura import VisionModel

# Per-frame vision cone mask for a single player
heading = player_heading(frame, player_idx=0, team="home")
mask = vision_cone_mask(frame, player_idx=0, team="home", grid=grid)

# Full VisionModel: blind-spot pressure surface
vision = VisionModel()
pressure = vision.blind_spot_pressure(frame, grid)
```

---

## Sync Utilities

```python
from pitch_aura.sync.alignment import align
from pitch_aura.sync.filters import smooth

# Align two FrameSequences by timestamp
aligned_home, aligned_away = align(seq_home, seq_away)

# Smooth positions with a Kalman filter
smoothed = smooth(sequence, method="kalman")
```

---

## Visualisation

Requires `pip install "pitch-aura[viz]"`. All functions return `plotly.graph_objects.Figure` and never call `.show()`.

```python
from pitch_aura.viz import plot_pitch_control, plot_voronoi_control

# Pitch control heatmap with player overlay
fig = plot_pitch_control(grid, frame, home_team_id="home")
fig.write_html("pitch_control.html")

# Voronoi territories
from pitch_aura.viz import plot_voronoi_control
fig = plot_voronoi_control(result, frame, home_team_id="home")
fig.show()

# Frame-by-frame animation
from pitch_aura.viz import animate_sequence
fig = animate_sequence(sequence, model)
fig.write_html("match_animation.html")
```

### Composable low-level API

```python
from pitch_aura.viz._pitch_draw import pitch_background
from pitch_aura.viz.heatmap import plot_heatmap
from pitch_aura.viz.players import plot_players
from pitch_aura.viz.tactics import plot_pockets, plot_passing_lane

fig = pitch_background()
fig = plot_heatmap(grid, fig=fig)
fig = plot_players(frame, fig=fig, show_pitch=False)
fig = plot_pockets(pockets, fig=fig)
```

---

## API Reference

### Core types

| Class | Description |
|-------|-------------|
| `FrameRecord` | Single tracking frame: positions, velocities, team IDs, ball |
| `FrameSequence` | Ordered sequence of frames + `PitchSpec` |
| `ProbabilityGrid` | `(H, W)` control probability array with coordinate metadata |
| `VoronoiResult` | Clipped Voronoi regions per player |
| `PitchSpec` | Pitch dimensions and coordinate convention |
| `EventRecord` | Discrete event (pass, shot, etc.) aligned to a frame |

### Models

| Class / function | Description |
|-----------------|-------------|
| `KinematicControlModel` | Spearman (2018) time-to-intercept pitch control |
| `VoronoiModel` | Mirror-augmented Voronoi with Sutherland-Hodgman clipping |
| `VisionModel` | Sigmoid vision cone + blind-spot pressure surface |

---

## Acknowledgements

The kinematic pitch control model is based on:

> Spearman, W. (2018). *Beyond Expected Goals*. MIT Sloan Sports Analytics Conference.

---

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).
