"""
Project: PitchAura
File Created: 2026-02-24
Author: Xingnan Zhu
File Name: _pitch_draw.py
Description:
    Internal pitch background rendering for Plotly figures.
    Re-exported from pitch-core for backward compatibility.
"""

from pitch_core.viz._pitch_draw import (  # noqa: F401
    _arc_path,
    _circle_path,
    _make_pitch_traces,
    pitch_background,
)

__all__ = ["pitch_background"]
