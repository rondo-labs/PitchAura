# Pitch Control & Voronoi

## Overview

PitchAura provides two complementary ways to model spatial dominance on a football pitch:

- **Kinematic pitch control** — a physics-based model of *who can reach which area first*, accounting for player speeds and directions.
- **Voronoi tessellation** — a purely geometric model that assigns each pitch cell to the nearest player.

---

## Kinematic control model

Based on Spearman (2018), the `KinematicControlModel` estimates for every grid cell the probability that a given team can control the ball if it arrives there.

### Key idea

For each player, the model computes a *time-to-intercept* (TTI) — how long it would take to reach a cell given current position, velocity, and reaction time. The team that can collectively reach a cell earliest controls it with high probability (via a sigmoid function).

### Usage

```python
import pitch_aura as pa

model = pa.KinematicControlModel(
    grid_shape=(68, 105),    # rows × cols (default)
    max_player_speed=13.0,   # m/s (default)
    reaction_time=0.7,       # seconds (default)
    tti_sigma=0.45,          # sigmoid steepness (default)
)

grid = model.compute(frame)   # → ProbabilityGrid
```

### ProbabilityGrid

```python
grid.values        # np.ndarray shape (H, W), home control probability 0–1
grid.x_coords      # np.ndarray shape (W,), x positions of grid columns
grid.y_coords      # np.ndarray shape (H,), y positions of grid rows
grid.pitch         # PitchSpec with pitch dimensions
```

Away control at any cell is simply `1 - grid.values[i, j]`.

### Custom pitch dimensions

```python
spec = pa.PitchSpec(length=105.0, width=68.0)  # metres
model = pa.KinematicControlModel(pitch=spec)
```

### Batch computation over a sequence

```python
grids = [model.compute(f) for f in sequence.frames]
# or more efficiently:
import numpy as np
control_stack = np.stack([model.compute(f).values for f in sequence.frames])
# shape: (n_frames, H, W)
mean_control = control_stack.mean(axis=0)
```

---

## Voronoi model

The `VoronoiModel` assigns each pitch cell to the nearest player (Euclidean distance), clipped to the pitch boundary.

### Usage

```python
voronoi = pa.VoronoiModel()
result = voronoi.compute(frame)   # → VoronoiResult
```

### VoronoiResult

```python
result.regions   # list of (player_idx, team_id, polygon_vertices)
                 # polygon_vertices: np.ndarray shape (M, 2), metres
result.areas     # dict mapping (player_idx, team_id) → area in m²
```

### Implementation notes

- Uses mirror-point augmentation to ensure boundary cells are correctly assigned.
- Polygon clipping uses the Sutherland-Hodgman algorithm against the pitch rectangle.

---

## Choosing between the two models

| | Kinematic | Voronoi |
|---|---|---|
| Accounts for velocity | Yes | No |
| Accounts for reaction time | Yes | No |
| Computational cost | Higher | Lower |
| Use case | Realistic pressure maps | Quick territorial overview |

For tactical dashboards requiring real-time or near-real-time computation, Voronoi is a useful lightweight alternative. For research or detailed analysis, kinematic control is more accurate.
