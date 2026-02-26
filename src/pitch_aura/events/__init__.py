"""Event-based spatial analysis for PitchAura.

Layer 1 — works with any event data (coordinates only):

    progressive_actions   Spatial progression metrics for passes/carries.
    passing_network       Spatial passing network (nodes + edges).
    zone_counts           Coarse zone event frequency.
    event_density         Smooth KDE event density surface.

Layer 2 — requires freeze-frame data (e.g. StatsBomb):

    event_control         Pitch control for a single event snapshot.
    batch_event_control   Pitch control for multiple event snapshots.
"""

from pitch_aura.events.passing_network import PassingNetwork, passing_network
from pitch_aura.events.progressive import progressive_actions
from pitch_aura.events.snapshot import batch_event_control, event_control
from pitch_aura.events.zones import event_density, zone_counts

__all__ = [
    "PassingNetwork",
    "batch_event_control",
    "event_control",
    "event_density",
    "passing_network",
    "progressive_actions",
    "zone_counts",
]
