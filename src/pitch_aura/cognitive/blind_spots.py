"""
Project: PitchAura
File Created: 2026-02-23
Author: Xingnan Zhu
File Name: blind_spots.py
Description:
    Defender awareness model: vision cones and blind spot penalization.
    VisionModel takes a ProbabilityGrid produced by KinematicControlModel
    and reduces the defending team's control probability in grid cells that
    fall outside defenders' vision cones, reflecting the cognitive cost of
    monitoring threats in peripheral vision or completely out of sight.
"""

from __future__ import annotations

import numpy as np

from pitch_aura._grid import make_grid
from pitch_aura.cognitive.vision_cone import player_heading, vision_cone_mask
from pitch_aura.types import FrameRecord, PitchSpec, ProbabilityGrid


class VisionModel:
    """Adjust pitch control probabilities for defender visual awareness.

    For each frame, applies a vision-cone mask computed from each
    defender's velocity direction to their individual contribution to
    the control surface.  Cells in a defender's blind spot have their
    defensive control weight reduced by *peripheral_penalty*, making the
    attacking probability in those cells correspondingly higher.

    Stationary defenders (speed < 0.1 m/s) are assumed to face the ball.

    Parameters:
        cone_half_angle:      Half-angle of each defender's visual field in
                              degrees (default 100°).
        peripheral_penalty:   Fraction of control weight retained outside
                              the cone.  0.0 = fully blind; 1.0 = no effect.
        transition_sharpness: Sigmoid sharpness at the cone boundary.
    """

    def __init__(
        self,
        *,
        cone_half_angle: float = 100.0,
        peripheral_penalty: float = 0.5,
        transition_sharpness: float = 10.0,
    ) -> None:
        self.cone_half_angle = cone_half_angle
        self.peripheral_penalty = peripheral_penalty
        self.transition_sharpness = transition_sharpness

    def apply(
        self,
        grid: ProbabilityGrid,
        frame: FrameRecord,
        *,
        defending_team_id: str,
        pitch: PitchSpec | None = None,
    ) -> ProbabilityGrid:
        """Apply vision-cone penalty to defenders and return adjusted grid.

        The adjustment is performed on the *complement* of the attacker's
        control probability (i.e. the defender's contribution):

        1. Compute each defender's vision-cone mask over the grid.
        2. Combine masks by taking the maximum (most attentive defender wins).
        3. Reduce the defender side's control by multiplying by the mask.
        4. Renormalise so attacking + defending probabilities still sum to 1.

        Parameters:
            grid:              Pitch control grid (attacker probability).
            frame:             The same frame used to compute *grid*.
            defending_team_id: Team ID of the defending side.
            pitch:             Pitch spec for grid reconstruction. Falls back
                               to ``grid.pitch`` if omitted.

        Returns:
            A new :class:`ProbabilityGrid` with vision-adjusted probabilities,
            clipped to ``[0, 1]``.
        """
        resolved_pitch = pitch if pitch is not None else grid.pitch
        nx, ny = grid.resolution
        targets, _, _ = make_grid(resolved_pitch, (nx, ny))  # (G, 2)

        def_mask_arr = np.array([t == defending_team_id for t in frame.team_ids], dtype=bool)

        if not def_mask_arr.any():
            # No defenders: return original grid unchanged
            return ProbabilityGrid(
                values=grid.values.copy(),
                x_edges=grid.x_edges.copy(),
                y_edges=grid.y_edges.copy(),
                pitch=grid.pitch,
                timestamp=grid.timestamp,
            )

        def_positions = frame.positions[def_mask_arr]    # (D, 2)
        def_velocities = (
            frame.velocities[def_mask_arr]
            if frame.velocities is not None
            else np.zeros_like(def_positions)
        )

        # Combined vision mask: max over all defenders (G,)
        combined_mask = np.zeros(len(targets))
        for i in range(len(def_positions)):
            fallback = frame.ball_position - def_positions[i]
            heading = player_heading(def_velocities[i], fallback_direction=fallback)
            cone = vision_cone_mask(
                def_positions[i],
                heading,
                targets,
                cone_half_angle=self.cone_half_angle,
                peripheral_penalty=self.peripheral_penalty,
                transition_sharpness=self.transition_sharpness,
            )
            combined_mask = np.maximum(combined_mask, cone)

        # PPCF_att is grid.values; PPCF_def = 1 - PPCF_att
        ppcf_att = grid.values.ravel()       # (G,)
        ppcf_def = 1.0 - ppcf_att            # (G,)

        # Reduce defensive control in blind spots
        ppcf_def_adjusted = ppcf_def * combined_mask  # (G,)

        # Renormalise so att + def = 1
        total = ppcf_att + ppcf_def_adjusted
        safe_total = np.where(total < 1e-9, 1.0, total)
        ppcf_att_adjusted = ppcf_att / safe_total

        adjusted_values = np.clip(ppcf_att_adjusted.reshape(nx, ny), 0.0, 1.0)

        return ProbabilityGrid(
            values=adjusted_values,
            x_edges=grid.x_edges.copy(),
            y_edges=grid.y_edges.copy(),
            pitch=grid.pitch,
            timestamp=grid.timestamp,
        )
