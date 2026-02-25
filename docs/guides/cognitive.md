# Cognitive Models

PitchAura's cognitive module models what players can and cannot *see* on the pitch, and how visual awareness modifies pitch control.

---

## Vision cone

A player's vision cone is the angular sector in front of them. Cells inside the cone are visible; cells behind the player are in a blind spot.

### Player heading

`player_heading` infers the direction a player is facing from their velocity vector (or falls back to a default when stationary).

```python
from pitch_aura.cognitive.vision_cone import player_heading

heading_rad = player_heading(frame, player_idx=0, team="home")
# → float, angle in radians (0 = right, π/2 = up)
```

### Vision cone mask

`vision_cone_mask` returns a grid mask (same shape as `ProbabilityGrid`) where each cell's value is the visibility weight for the given player. The transition at the cone boundary is smooth (sigmoid).

```python
from pitch_aura.cognitive.vision_cone import vision_cone_mask

mask = vision_cone_mask(
    frame,
    player_idx=0,
    team="home",
    grid=grid,
    half_angle=80.0,   # degrees (default)
    sigma=10.0,        # sigmoid steepness at boundary
)
# → np.ndarray shape (H, W), values 0–1
```

---

## VisionModel — blind-spot pressure

`VisionModel` aggregates vision cone information across all defending players to produce a *blind-spot pressure* surface: cells where the defending team has poor collective visibility.

```python
import pitch_aura as pa

vision = pa.VisionModel(half_angle=80.0)
pressure = vision.blind_spot_pressure(frame, grid)
# → np.ndarray shape (H, W)
# High values = areas with poor defensive visibility (attacking opportunities)
```

The pressure surface uses max-pooling across all defenders: a cell is "visible" if *any* defender can see it.

---

## VisionAwareControlModel

`VisionAwareControlModel` is a drop-in replacement for `KinematicControlModel` that weights pitch control by the defending team's visual awareness. Areas in the defensive blind spot receive inflated attacking control values.

```python
from pitch_aura.cognitive.gravity_vision import VisionAwareControlModel

va_model = VisionAwareControlModel(
    half_angle=80.0,
    vision_weight=0.3,   # how much vision modulates control (0 = pure kinematic)
)

grid = va_model.compute(frame)   # → ProbabilityGrid, same interface as KinematicControlModel
```

### Swapping models

Because `VisionAwareControlModel` exposes the same `.compute()` interface, it works anywhere `KinematicControlModel` is accepted:

```python
from pitch_aura.tactics.space_creation import space_creation

scores = space_creation(frame, control_model=va_model)
```

---

## Combining with gravity

```python
from pitch_aura.tactics.gravity import spatial_drag_index

df = spatial_drag_index(sequence, control_model=va_model)
```
