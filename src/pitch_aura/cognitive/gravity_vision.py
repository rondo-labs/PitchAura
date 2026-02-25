"""
Project: PitchAura
File Created: 2026-02-25
Author: Xingnan Zhu
File Name: gravity_vision.py
Description:
    Vision-aware control model for gravity/deformation analysis.
    Chains KinematicControlModel → VisionModel to produce pitch control
    grids that account for defender visual awareness.  Exposes the same
    control(frame, team_id=...) -> ProbabilityGrid interface as
    KinematicControlModel, making it a drop-in replacement for the
    control_model parameter in any gravity function.
"""

from __future__ import annotations

from pitch_aura.cognitive.blind_spots import VisionModel
from pitch_aura.constants import DEFAULT_GRID_RESOLUTION
from pitch_aura.space.kinematic import KinematicControlModel
from pitch_aura.types import FrameRecord, PitchSpec, ProbabilityGrid


class VisionAwareControlModel:
    """Pitch control model with defender visual awareness applied.

    Chains :class:`~pitch_aura.space.kinematic.KinematicControlModel`
    (continuous PPCF field) with
    :class:`~pitch_aura.cognitive.blind_spots.VisionModel` (blind-spot
    penalisation) to produce grids where defenders who cannot see the runner
    have reduced spatial control — amplifying the deformation effect of
    off-ball runs into blind spots.

    Exposes ``control(frame, team_id=...) -> ProbabilityGrid`` so it can
    be passed as ``control_model=VisionAwareControlModel(...)`` to any gravity
    function (:func:`~pitch_aura.tactics.gravity.spatial_drag_index`,
    :func:`~pitch_aura.tactics.gravity.net_space_generated`, etc.).

    Parameters:
        pitch:                Pitch specification; defaults to 105×68 m.
        resolution:           Grid resolution ``(nx, ny)``.
        cone_half_angle:      Defender vision cone half-angle in degrees
                              (default 100°).
        peripheral_penalty:   Control weight fraction retained outside the
                              vision cone.  0.0 = fully blind; 1.0 = no effect
                              (default 0.5).
        transition_sharpness: Sigmoid sharpness at the cone boundary
                              (default 10.0).
    """

    def __init__(
        self,
        *,
        pitch: PitchSpec | None = None,
        resolution: tuple[int, int] = DEFAULT_GRID_RESOLUTION,
        cone_half_angle: float = 100.0,
        peripheral_penalty: float = 0.5,
        transition_sharpness: float = 10.0,
    ) -> None:
        resolved_pitch = pitch if pitch is not None else PitchSpec()
        self._kinematic = KinematicControlModel(
            pitch=resolved_pitch, resolution=resolution
        )
        self._vision = VisionModel(
            cone_half_angle=cone_half_angle,
            peripheral_penalty=peripheral_penalty,
            transition_sharpness=transition_sharpness,
        )
        self._pitch = resolved_pitch

    def control(
        self,
        frame: FrameRecord,
        *,
        team_id: str,
    ) -> ProbabilityGrid:
        """Compute vision-aware pitch control for *team_id*.

        Steps:

        1. Compute base PPCF grid via :class:`KinematicControlModel`.
        2. Determine the defending team (any team that is not *team_id*).
        3. Apply :meth:`VisionModel.apply` to reduce control in defenders'
           blind spots, raising the attacker's effective probability there.

        Parameters:
            frame:   Tracking frame to evaluate.
            team_id: Attacking team ID (grid values = attacking probability).

        Returns:
            Vision-adjusted :class:`~pitch_aura.types.ProbabilityGrid`.
        """
        base_grid = self._kinematic.control(frame, team_id=team_id)

        # Defending team = any team_id in the frame that is not the attacker
        defending_team_id: str | None = None
        for tid in frame.team_ids:
            if tid != team_id:
                defending_team_id = tid
                break

        if defending_team_id is None:
            # No defenders in frame — return base grid unchanged
            return base_grid

        return self._vision.apply(
            base_grid,
            frame,
            defending_team_id=defending_team_id,
            pitch=self._pitch,
        )
