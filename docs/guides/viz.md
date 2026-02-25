# Visualization

Requires `pip install "pitch-aura[viz]"` (Plotly + Kaleido).

All visualization functions:

- Return a `plotly.graph_objects.Figure`.
- Never call `.show()` — that's the caller's responsibility.
- Accept an optional `fig=None` parameter for composability (pass an existing figure to layer traces onto it).

---

## High-level API

### Pitch control heatmap

```python
from pitch_aura.viz import plot_pitch_control

fig = plot_pitch_control(grid, frame, home_team_id="home")
fig.write_html("pitch_control.html")
```

Renders a green/red heatmap of home team control probability with player markers overlaid.

### Voronoi territories

```python
from pitch_aura.viz import plot_voronoi_control

fig = plot_voronoi_control(result, frame, home_team_id="home")
fig.show()
```

### Frame animation

```python
from pitch_aura.viz import animate_sequence

fig = animate_sequence(sequence, model)
fig.write_html("match_animation.html")
```

Produces a slider-controlled Plotly animation with one frame per `FrameRecord`.

---

## Composable low-level API

For custom layouts, build up figures layer by layer:

```python
from pitch_aura.viz._pitch_draw import pitch_background
from pitch_aura.viz.heatmap import plot_heatmap
from pitch_aura.viz.players import plot_players
from pitch_aura.viz.voronoi import plot_voronoi

fig = pitch_background()                          # FIFA pitch markings
fig = plot_heatmap(grid, fig=fig)                 # control heatmap
fig = plot_players(frame, fig=fig, show_pitch=False)  # player markers
```

### Tactics overlays

```python
from pitch_aura.viz.tactics import (
    plot_pockets,
    plot_passing_lane,
    plot_deformation_field,
    plot_gravity_timeseries,
    plot_flow_field,
    plot_interaction_matrix,
)

# Mark line-breaking pockets
fig = plot_pockets(pockets, fig=fig)

# Draw a passing lane between two players
fig = plot_passing_lane(frame, passer_idx=7, receiver_idx=9, fig=fig)

# Deformation vector field
fig = plot_deformation_field(field, fig=fig)

# Gravity time-series chart
fig = plot_gravity_timeseries(gravity_df)

# Flow field (streamlines)
fig = plot_flow_field(field)

# Interaction matrix heatmap
fig = plot_interaction_matrix(matrix, frame)
```

---

## Saving figures

```python
# HTML (interactive, no extra deps)
fig.write_html("output.html")

# Static PNG/PDF (requires kaleido)
fig.write_image("output.png")
fig.write_image("output.pdf")
```

---

## Pitch background

The `pitch_background()` function draws a full FIFA-spec pitch with:

- Pitch outline and centre circle
- Penalty areas and goal areas
- Goal mouths
- Corner arcs and centre spot

All markings are drawn as `go.Scatter` traces (not layout shapes) so that heatmap layers render behind them correctly.

```python
from pitch_aura.viz._pitch_draw import pitch_background

fig = pitch_background(
    line_color="white",
    pitch_color="#2d7a27",
    width=800,
    height=600,
)
```
