"""
Project: PitchAura
File Created: 2026-02-23
Author: Xingnan Zhu
File Name: vision_cone.py
Description:
    Vision cone geometry computation based on player heading.
    Provides a vectorized function that builds a soft angular-attention mask
    over a spatial grid for a single player, given their position and
    heading direction. Cells within the cone half-angle receive weight 1.0;
    cells outside receive a configurable peripheral penalty multiplier.
    Transition between in-cone and out-of-cone regions uses a sigmoid
    to avoid hard discontinuities.
"""

from __future__ import annotations

import numpy as np


def player_heading(
    velocity: np.ndarray,
    fallback_direction: np.ndarray | None = None,
) -> np.ndarray:
    """Infer a player's heading direction from their velocity vector.

    Parameters:
        velocity:           Velocity vector, shape ``(2,)``.
        fallback_direction: Direction to use when the player is stationary
                            (speed < 0.1 m/s).  Shape ``(2,)``.  If ``None``
                            and the player is stationary, the heading is
                            considered undefined and the all-ones mask is
                            returned for any cone computation.

    Returns:
        Unit-length heading vector, shape ``(2,)``, or ``None`` if the
        velocity is negligible and no fallback is provided.
    """
    speed = float(np.linalg.norm(velocity))
    if speed < 0.1:
        if fallback_direction is None:
            return np.array([1.0, 0.0])  # arbitrary default
        fb = np.asarray(fallback_direction, dtype=float)
        fb_speed = float(np.linalg.norm(fb))
        if fb_speed < 1e-9:
            return np.array([1.0, 0.0])
        return fb / fb_speed
    return velocity / speed


def vision_cone_mask(
    position: np.ndarray,
    heading: np.ndarray,
    targets: np.ndarray,
    *,
    cone_half_angle: float = 100.0,
    peripheral_penalty: float = 0.5,
    transition_sharpness: float = 10.0,
) -> np.ndarray:
    """Compute a soft vision-cone attention mask for a grid.

    For each target point (grid cell centre), computes a weight in
    ``[peripheral_penalty, 1.0]`` representing how well the player can
    attend to that cell given their heading.  The transition from full
    attention to peripheral is modelled with a sigmoid.

    Parameters:
        position:             Player position, shape ``(2,)``.
        heading:              Unit heading vector, shape ``(2,)``.
        targets:              Grid cell centres, shape ``(G, 2)``.
        cone_half_angle:      Half-angle of the vision cone in degrees.
                              Default 100° (generous peripheral vision).
        peripheral_penalty:   Weight assigned to cells outside the cone.
                              0.0 = complete blindness; 1.0 = no penalty.
        transition_sharpness: Controls how sharply the sigmoid transitions
                              at the cone boundary.  Higher = sharper.

    Returns:
        Weight array, shape ``(G,)``, values in
        ``[peripheral_penalty, 1.0]``.
    """
    diff = targets - position[np.newaxis, :]          # (G, 2)
    dist = np.linalg.norm(diff, axis=1, keepdims=True)  # (G, 1)

    # Avoid division by zero for targets at the player's position
    safe_dist = np.where(dist < 1e-9, 1.0, dist)
    unit_diff = diff / safe_dist                        # (G, 2)

    cos_angle = np.clip(unit_diff @ heading, -1.0, 1.0)  # (G,)
    angle_deg = np.degrees(np.arccos(cos_angle))          # (G,)

    # Soft mask: sigmoid centred at cone_half_angle
    # angle < half_angle → in cone (weight → 1.0)
    # angle > half_angle → outside (weight → peripheral_penalty)
    logit = transition_sharpness * (cone_half_angle - angle_deg) / cone_half_angle
    sigmoid = 1.0 / (1.0 + np.exp(-logit))              # (G,)

    # Remap [0, 1] sigmoid to [peripheral_penalty, 1.0]
    mask = peripheral_penalty + (1.0 - peripheral_penalty) * sigmoid  # (G,)

    # Cells exactly at player position are fully visible
    at_player = (dist[:, 0] < 1e-9)
    mask[at_player] = 1.0

    return mask
