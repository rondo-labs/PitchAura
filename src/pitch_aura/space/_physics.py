"""
Project: PitchAura
File Created: 2026-02-23
Author: Xingnan Zhu
File Name: _physics.py
Description:
    Internal physics functions: time-to-intercept and influence calculations.
    All operations are fully vectorized over (N players) x (G grid points)
    to avoid Python-level loops in hot paths.
"""

from __future__ import annotations

import numpy as np

_SIGMOID_SCALE = np.pi / np.sqrt(3.0)


def time_to_intercept(
    positions: np.ndarray,
    targets: np.ndarray,
    reaction_time: float,
    v_max: float,
) -> np.ndarray:
    """Compute time for each player to intercept each target position.

    Uses the simplified kinematic model: the player first waits
    ``reaction_time`` seconds, then sprints at ``v_max`` toward the target.

    Parameters:
        positions: Player positions, shape ``(N, 2)``.
        targets:   Grid cell centres, shape ``(G, 2)``.
        reaction_time: Reaction delay in seconds.
        v_max:     Maximum sprint speed in m/s.

    Returns:
        TTI matrix, shape ``(N, G)``.
    """
    # (N, 1, 2) - (1, G, 2)  ->  (N, G, 2)
    diff = targets[np.newaxis, :, :] - positions[:, np.newaxis, :]
    dist = np.linalg.norm(diff, axis=2)          # (N, G)
    return reaction_time + dist / v_max           # (N, G)


def sigmoid_influence(
    tti: np.ndarray,
    t: float,
    sigma: float,
    lam: float = 1.0,
) -> np.ndarray:
    """Logistic influence of each player over each grid point at time ``t``.

    Models the probability that a player, whose time-to-intercept is
    ``tti``, is exerting meaningful control at time ``t``.

    Parameters:
        tti:   TTI matrix, shape ``(N, G)``.
        t:     Current integration time in seconds.
        sigma: Transition width (TTI uncertainty).
        lam:   Influence weight; scales the sigmoid slope.
               Higher = steeper transition to full control.

    Returns:
        Influence matrix, shape ``(N, G)``, values in ``(0, 1)``.
    """
    return 1.0 / (1.0 + np.exp(-_SIGMOID_SCALE * lam * (t - tti) / sigma))


def accumulate_control(
    tti_att: np.ndarray,
    tti_def: np.ndarray,
    *,
    sigma_att: float,
    sigma_def: float,
    lam_att: float,
    lam_def: float,
    dt: float,
    t_max: float,
    convergence_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Numerically integrate the Pitch Possession Control Function (PPCF).

    Implements the Spearman (2018) model via forward Euler integration.
    Each timestep computes the rate of control accumulation for both
    teams simultaneously across all ``G`` grid points.

    Parameters:
        tti_att: Attacker TTI matrix, shape ``(N_att, G)``.
        tti_def: Defender TTI matrix, shape ``(N_def, G)``.
        sigma_att, sigma_def: TTI uncertainty per team.
        lam_att, lam_def:     Influence weight per team.
        dt:                   Integration timestep in seconds.
        t_max:                Maximum integration time in seconds.
        convergence_threshold: Stop when remaining capacity < threshold.

    Returns:
        Tuple of ``(PPCF_att, PPCF_def)``, each shape ``(G,)``.
        ``PPCF_att[g]`` is the attacking team's control probability at
        grid point ``g``.
    """
    G = tti_att.shape[1]
    PPCF_att = np.zeros(G)
    PPCF_def = np.zeros(G)

    for t in np.arange(0.0, t_max + dt, dt):
        f_att = sigmoid_influence(tti_att, t, sigma_att, lam_att)  # (N_att, G)
        f_def = sigmoid_influence(tti_def, t, sigma_def, lam_def)  # (N_def, G)

        # Product of (1 - influence) for the opposing team
        prod_def = np.prod(1.0 - f_def, axis=0)   # (G,) -- defenders uncontested
        prod_att = np.prod(1.0 - f_att, axis=0)   # (G,) -- attackers uncontested

        # Remaining "unclaimed" probability mass at each grid point
        remaining = np.maximum(0.0, 1.0 - PPCF_att - PPCF_def)  # (G,)

        # Forward Euler accumulation
        PPCF_att += remaining * np.sum(f_att, axis=0) * prod_def * dt
        PPCF_def += remaining * np.sum(f_def, axis=0) * prod_att * dt

        # Early exit once all grid points have converged
        if np.all((PPCF_att + PPCF_def) >= 1.0 - convergence_threshold):
            break

    return PPCF_att, PPCF_def
