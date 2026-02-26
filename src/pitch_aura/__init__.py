"""PitchAura — A spatial analytics computation engine for football/soccer.

Transforms raw tracking coordinates into tactical spatial matrices
and evaluation metrics.
"""

from importlib.metadata import PackageNotFoundError, version

from pitch_aura.cognitive.blind_spots import VisionModel
from pitch_aura.events import (
    PassingNetwork,
    batch_event_control,
    event_control,
    event_density,
    passing_network,
    progressive_actions,
    zone_counts,
)
from pitch_aura.io.kloppy_adapter import from_events, from_tracking
from pitch_aura.space.kinematic import KinematicControlModel
from pitch_aura.space.voronoi import VoronoiModel
from pitch_aura.tactics.line_breaking import Pocket
from pitch_aura.types import (
    EventRecord,
    FrameRecord,
    FrameSequence,
    PitchSpec,
    ProbabilityGrid,
    VoronoiResult,
)

try:
    __version__ = version("pitch-aura")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = [
    # Data models
    "EventRecord",
    "FrameRecord",
    "FrameSequence",
    "PassingNetwork",
    "PitchSpec",
    "ProbabilityGrid",
    "VoronoiResult",
    # I/O
    "from_events",
    "from_tracking",
    # Spatial models
    "KinematicControlModel",
    "VoronoiModel",
    # Cognitive
    "VisionModel",
    # Tactics
    "Pocket",
    # Events
    "batch_event_control",
    "event_control",
    "event_density",
    "passing_network",
    "progressive_actions",
    "zone_counts",
    "__version__",
]
