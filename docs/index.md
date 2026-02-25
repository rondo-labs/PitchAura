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
| **Tactics** | Space creation, passing lane lifespan, line-breaking pockets, spatial gravity & deformation |
| **Cognitive** | Player vision cones, blind-spot pressure maps, `VisionAwareControlModel` |
| **Sync** | Frame alignment across tracking streams + moving-average / Kalman filtering |
| **I/O** | kloppy adapter (`from_tracking`, `from_events`) |
| **Viz** | Plotly-based heatmaps, Voronoi plots, deformation fields, animations |

---

## Installation

```bash
pip install pitch-aura
```

With optional extras:

```bash
pip install "pitch-aura[kloppy]"     # kloppy I/O adapter
pip install "pitch-aura[viz]"         # Plotly visualisation
pip install "pitch-aura[kloppy,viz]"  # Everything
```

**Requirements:** Python ≥ 3.12, NumPy ≥ 1.24, SciPy ≥ 1.10, pandas ≥ 2.0.

---

## Getting started

Head to the [Quick Start](quickstart.md) page for hands-on examples, or browse the User Guide for in-depth coverage of each module.

---

## Acknowledgements

The kinematic pitch control model is based on:

> Spearman, W. (2018). *Beyond Expected Goals*. MIT Sloan Sports Analytics Conference.

## License

GNU General Public License v3.0 — see [LICENSE](https://github.com/rondo-labs/PitchAura/blob/main/LICENSE).
