"""
Project: PitchAura
File Created: 2026-02-23
Author: Xingnan Zhu
File Name: kloppy_adapter.py
Description:
    Adapter for converting kloppy data models to pitch_aura types.
    Re-exported from pitch-core for backward compatibility.
"""

from pitch_core.io.kloppy_adapter import (  # noqa: F401
    _compute_velocities,
    _extract_frame,
    from_events,
    from_tracking,
)

__all__ = ["from_events", "from_tracking"]
