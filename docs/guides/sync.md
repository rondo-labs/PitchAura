# Sync Utilities

Tracking data often comes from multiple sources (e.g., separate home/away cameras) with slightly misaligned timestamps. The sync module provides tools to align and smooth tracking sequences.

---

## Frame alignment

`align` synchronises two `FrameSequence` objects by timestamp, producing a pair of sequences that share the same frame grid.

```python
from pitch_aura.sync.alignment import align

aligned_home, aligned_away = align(seq_home, seq_away)

# Both sequences now have the same number of frames
assert len(aligned_home.frames) == len(aligned_away.frames)
```

**How it works:** `align` finds the intersection of timestamp ranges and linearly interpolates positions to a common time grid. Frames with timestamps outside the overlapping window are discarded.

---

## Smoothing

`smooth` applies temporal filtering to positions (and optionally velocities) in a `FrameSequence`.

```python
from pitch_aura.sync.filters import smooth

# Moving average (window = 5 frames)
smoothed_ma = smooth(sequence, method="moving_average", window=5)

# Kalman filter (constant-velocity model)
smoothed_kf = smooth(sequence, method="kalman")
```

### Moving average

Simple uniform window smoothing. Fast and interpretable. Useful for removing high-frequency jitter.

- Parameter: `window` (int, default 5)

### Kalman filter

A constant-velocity Kalman filter with state `[x, y, vx, vy]`. Handles missing frames and non-uniform time steps more gracefully than a fixed window.

- No additional parameters required.
- Velocities in the output frames are the Kalman-estimated velocities (not finite-difference derivatives).

```python
# Access smoothed velocities
f = smoothed_kf.frames[10]
print(f.home_velocities)   # Kalman-estimated velocities
```

---

## Notes on FrameSequence

Both `align` and `smooth` return a new `FrameSequence` — the original is never modified.

```python
import pitch_aura as pa

sequence = pa.FrameSequence(frames=[...], pitch=pa.PitchSpec())
smoothed = smooth(sequence, method="kalman")

# original unchanged
assert sequence.frames[0].home_positions is not smoothed.frames[0].home_positions
```
