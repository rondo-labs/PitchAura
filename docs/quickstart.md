# Quick Start

This page walks through the most common workflows in PitchAura.

## Build a frame manually

`FrameRecord` is the core unit — a single tracking snapshot.
Positions and velocities are `(N, 2)` NumPy arrays in metres.

```python
import numpy as np
import pitch_aura as pa

frame = pa.FrameRecord(
    frame_id=1,
    timestamp=0.04,
    home_positions=np.array([[30.0, 0.0], [20.0, 10.0]]),
    away_positions=np.array([[-25.0, 5.0], [-15.0, -8.0]]),
    home_velocities=np.array([[2.0, 0.5], [1.0, -1.0]]),
    away_velocities=np.array([[-1.5, 0.0], [-2.0, 1.0]]),
    home_team_id="home",
    ball_position=np.array([5.0, 0.0]),
)
```

## Kinematic pitch control

```python
model = pa.KinematicControlModel()
grid = model.compute(frame)   # → ProbabilityGrid

print(grid.values.shape)      # (68, 105)  — home team control probability
```

`grid.values[i, j]` is the probability (0–1) that the home team controls
the pitch cell at row `i`, column `j`.

## Voronoi territories

```python
voronoi = pa.VoronoiModel()
result = voronoi.compute(frame)   # → VoronoiResult

for player_idx, team, polygon in result.regions:
    print(f"{team} player {player_idx}: {len(polygon)} vertices")
```

## Load from kloppy

```python
# pip install "pitch-aura[kloppy]"
import kloppy
dataset = kloppy.tracab.load(meta_data="meta.xml", raw_data="tracking.txt")

sequence = pa.from_tracking(dataset)   # → FrameSequence
frame = sequence.frames[0]
```

## Tactics in one minute

```python
from pitch_aura.tactics.space_creation import space_creation
from pitch_aura.tactics.passing_lanes import passing_lane_lifespan
from pitch_aura.tactics.line_breaking import line_breaking_pockets

# Space created by each home player (higher = more threatening)
scores = space_creation(frame, model)

# How many frames a lane from player 7 to player 9 stays open
lifespan = passing_lane_lifespan(sequence, passer_idx=7, receiver_idx=9, model=model)

# Gaps in the defensive line
pockets = line_breaking_pockets(frame)
for p in pockets:
    print(f"Pocket at depth {p.line_depth:.1f} m, y={p.y_left:.1f}–{p.y_right:.1f} m")
```

## Visualisation

```python
# pip install "pitch-aura[viz]"
from pitch_aura.viz import plot_pitch_control

fig = plot_pitch_control(grid, frame, home_team_id="home")
fig.write_html("pitch_control.html")
```

All viz functions return a `plotly.graph_objects.Figure` and never call `.show()`.

---

Next steps: browse the [User Guide](guides/pitch_control.md) for detailed explanations
of each module.
