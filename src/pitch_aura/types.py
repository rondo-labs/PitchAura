"""
Project: PitchAura
File Created: 2026-02-23
Author: Xingnan Zhu
File Name: types.py
Description:
    Core data models for pitch_aura.
    Shared types are re-exported from pitch-core; PitchAura-specific
    types (VoronoiResult) are defined here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Re-export shared types from pitch-core
from pitch_core.types import (  # noqa: F401
    EventRecord,
    FrameRecord,
    FrameSequence,
    PitchSpec,
    ProbabilityGrid,
)


@dataclass(slots=True)
class VoronoiResult:
    """Result of a Voronoi tessellation on a single frame."""

    regions: dict[str, np.ndarray]
    """Mapping from player_id to polygon vertices, each shape ``(M, 2)``."""
    areas: dict[str, float]
    """Mapping from player_id to controlled area in m²."""
    team_areas: dict[str, float]
    """Mapping from team_id to total controlled area in m²."""
    timestamp: float


__all__ = [
    "EventRecord",
    "FrameRecord",
    "FrameSequence",
    "PitchSpec",
    "ProbabilityGrid",
    "VoronoiResult",
]
