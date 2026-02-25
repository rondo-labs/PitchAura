# Tactics & Spatial Gravity

## Classic tactical metrics

### Space creation

`space_creation` quantifies how much pressure-free space each attacking player is generating — measured as the rate of change in the home team's total pitch control.

```python
from pitch_aura.tactics.space_creation import space_creation

scores = space_creation(frame, model)
# → np.ndarray shape (N_home,), one score per home player
```

Higher values indicate the player is making runs or positioning that pulls defenders and opens territory.

---

### Passing lane lifespan

`passing_lane_lifespan` counts for how many consecutive frames a direct passing corridor between two players remains open (i.e., the home team controls the intermediate cells).

```python
from pitch_aura.tactics.passing_lanes import passing_lane_lifespan

frames_open = passing_lane_lifespan(
    sequence,
    passer_idx=7,
    receiver_idx=9,
    model=model,
    threshold=0.5,   # home control threshold to count as "open"
)
print(f"Lane open for {frames_open} frames")
```

---

### Line-breaking pockets

`line_breaking_pockets` identifies exploitable gaps in the defensive line — spatial intervals along the defensive shape where there is no defender coverage.

```python
from pitch_aura.tactics.line_breaking import line_breaking_pockets, Pocket

pockets = line_breaking_pockets(frame)

for p in pockets:
    print(f"Gap at depth {p.line_depth:.1f} m")
    print(f"  y range: {p.y_left:.1f} – {p.y_right:.1f} m")
    print(f"  width:   {p.width:.1f} m")
```

Each `Pocket` exposes: `line_depth`, `y_left`, `y_right`, `width`.

---

## Spatial gravity & deformation

The gravity module extends the tactical layer with a physics-inspired model of how player positions *deform* the effective pitch space.

### Spatial drag index

`spatial_drag_index` measures how much a player's movement is being resisted by surrounding defensive pressure. A high SDI means the player is moving into congested areas.

```python
from pitch_aura.tactics.gravity import spatial_drag_index

df = spatial_drag_index(sequence)
# → pd.DataFrame with columns: frame_id, player_idx, team, sdi, sdi_efficiency
```

`sdi_efficiency` normalises SDI against the player's velocity magnitude.

---

### Net space generated

`net_space_generated` measures the aggregate change in controlled area for the home team across a sequence.

```python
from pitch_aura.tactics.gravity import net_space_generated

nsg = net_space_generated(sequence, model)
# → pd.DataFrame with columns: frame_id, net_space_m2
```

---

### Gravity profile

`gravity_profile` computes a time-series of the "gravitational pull" intensity across all players — a summary measure of how compressed/stretched the overall shape is.

```python
from pitch_aura.tactics.gravity import gravity_profile

profile = gravity_profile(sequence)
# → pd.DataFrame with columns: frame_id, gravity_mean, gravity_max
```

---

### Deformation recovery

`deformation_recovery` estimates how quickly the defensive shape recovers after a deformation event (e.g., a successful dribble or positional overload).

```python
from pitch_aura.tactics.gravity import deformation_recovery

metrics = deformation_recovery(sequence, event_frame_id=42)
# → RecoveryMetrics(recovery_frames, recovery_time_s, peak_deformation)
```

---

### Deformation flow field

`deformation_flow_field` returns the vector field of spatial deformation across the pitch — how every grid cell is "pushed" by player movements.

```python
from pitch_aura.tactics.gravity import deformation_flow_field

field = deformation_flow_field(frame)
# → DeformationVectorField with .dx, .dy arrays of shape (H, W)
```

---

### Gravity interaction matrix

`gravity_interaction_matrix` produces a player×player matrix of mutual gravitational influence — how strongly each pair of players is affecting each other's spatial freedom.

```python
from pitch_aura.tactics.gravity import gravity_interaction_matrix

matrix = gravity_interaction_matrix(frame)
# → np.ndarray shape (N_total, N_total)
```

---

### Penalty zone weights

`penalty_zone_weights` applies spatial discounting to gravity calculations near the penalty areas, reflecting the higher tactical value of those zones.

```python
from pitch_aura.tactics.gravity import penalty_zone_weights

weights = penalty_zone_weights(grid)   # shape (H, W)
```
