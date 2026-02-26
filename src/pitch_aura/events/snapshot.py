"""
Project: PitchAura
File Created: 2026-02-26
Author: Xingnan Zhu
File Name: snapshot.py
Description:
    Freeze-frame spatial analysis utilities.
    Enables running existing pitch control and Voronoi models on
    event data that carries freeze-frame player snapshots (e.g. StatsBomb).
"""

from __future__ import annotations

from pitch_aura.types import EventRecord, PitchSpec, ProbabilityGrid, VoronoiResult


def event_control(
    event: EventRecord,
    *,
    team_id: str | None = None,
    control_model: object | None = None,
    pitch: PitchSpec | None = None,
) -> ProbabilityGrid | VoronoiResult:
    """Compute spatial control for an event's freeze frame.

    By default uses :class:`~pitch_aura.space.kinematic.KinematicControlModel`
    which returns a :class:`ProbabilityGrid`.  A custom *control_model*
    may return any type.

    Parameters:
        event:         An :class:`EventRecord` with a non-``None``
                       ``freeze_frame``.
        team_id:       Attacking team identifier.  Required for the default
                       :class:`KinematicControlModel`.  If the event has a
                       ``team_id`` and this is ``None``, the event's team is
                       used.
        control_model: Custom spatial model with a ``control(frame, ...)``
                       method.  If ``None``,
                       :class:`KinematicControlModel` is used.
        pitch:         Pitch specification.  Required when *control_model*
                       is ``None``.

    Returns:
        Spatial control result (type depends on *control_model*).

    Raises:
        ValueError: If the event has no ``freeze_frame``, or if required
                    parameters are missing.
    """
    if event.freeze_frame is None:
        raise ValueError(
            f"Event at t={event.timestamp}s has no freeze_frame; "
            "cannot compute spatial control"
        )

    frame = event.freeze_frame

    if control_model is not None:
        return control_model.control(frame)  # type: ignore[union-attr]

    # Resolve team_id
    resolved_team = team_id if team_id is not None else event.team_id
    if resolved_team is None:
        raise ValueError(
            "team_id is required for the default KinematicControlModel; "
            "pass it explicitly or ensure the event has a team_id"
        )

    if pitch is None:
        raise ValueError("pitch is required when no control_model is given")

    from pitch_aura.space.kinematic import KinematicControlModel

    model = KinematicControlModel(pitch=pitch)
    return model.control(frame, team_id=resolved_team)


def batch_event_control(
    events: list[EventRecord],
    *,
    team_id: str | None = None,
    control_model: object | None = None,
    pitch: PitchSpec | None = None,
    event_types: tuple[str, ...] | None = None,
) -> list[tuple[EventRecord, ProbabilityGrid | VoronoiResult]]:
    """Compute spatial control for all events with freeze frames.

    Parameters:
        events:        List of event records.
        team_id:       Attacking team identifier (see :func:`event_control`).
        control_model: Custom spatial model (see :func:`event_control`).
        pitch:         Pitch specification.
        event_types:   Filter to these event types (``None`` = all with
                       freeze frames).

    Returns:
        List of ``(event, result)`` tuples for events that have freeze
        frames.
    """
    allowed = (
        {et.lower() for et in event_types} if event_types is not None else None
    )

    results: list[tuple[EventRecord, ProbabilityGrid | VoronoiResult]] = []
    for ev in events:
        if ev.freeze_frame is None:
            continue
        if allowed is not None and ev.event_type.lower() not in allowed:
            continue

        result = event_control(
            ev, team_id=team_id, control_model=control_model, pitch=pitch,
        )
        results.append((ev, result))

    return results
