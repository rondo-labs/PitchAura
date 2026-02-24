"""
Project: PitchAura
File Created: 2026-02-23
Author: Xingnan Zhu
File Name: kinematic.py
Description:
    Kinematic pitch control model based on Spearman (2018).
    Computes the Pitch Possession Control Function (PPCF) for every cell
    of a spatial grid using fully vectorised NumPy operations.
"""

from __future__ import annotations

import numpy as np

from pitch_aura._grid import make_grid
from pitch_aura.constants import (
    DEFAULT_CONVERGENCE_THRESHOLD,
    DEFAULT_GRID_RESOLUTION,
    DEFAULT_INTEGRATION_DT,
    DEFAULT_INTEGRATION_T_MAX,
    DEFAULT_LAMBDA_ATT,
    DEFAULT_LAMBDA_DEF,
    DEFAULT_LAMBDA_GK,
    DEFAULT_MAX_PLAYER_SPEED,
    DEFAULT_REACTION_TIME,
    DEFAULT_TTI_SIGMA,
)
from pitch_aura.space._physics import accumulate_control, time_to_intercept
from pitch_aura.types import FrameRecord, PitchSpec, ProbabilityGrid


class KinematicControlModel:
    """Physics-based pitch control using player kinematics.

    Implements the Spearman (2018) model: for each cell of a spatial grid
    the model numerically integrates competing player influences over time,
    producing a probability that the *attacking* team controls that cell.

    All heavy computation is vectorised: the full ``(N_players, G_cells)``
    TTI matrix is built in one NumPy broadcast, and the integration loop
    operates on ``(G,)`` arrays — no per-player or per-cell Python loops.

    Parameters:
        resolution:            ``(nx, ny)`` grid cells along each axis.
        pitch:                 Pitch specification. Defaults to 105 x 68 m.
        max_player_speed:      Sprint speed cap in m/s.
        reaction_time:         Reaction delay before sprint begins (s).
        tti_sigma:             TTI transition width (uncertainty).
        lambda_att:            Sigmoid slope weight for outfield attackers.
        lambda_def:            Sigmoid slope weight for outfield defenders.
        lambda_gk:             Sigmoid slope weight for goalkeepers
                               (lower = wider transition = less decisive).
        integration_dt:        Forward-Euler timestep in seconds.
        integration_t_max:     Maximum integration horizon in seconds.
        convergence_threshold: Stop early when remaining capacity
                               falls below this value at every grid cell.
    """

    def __init__(
        self,
        *,
        resolution: tuple[int, int] = DEFAULT_GRID_RESOLUTION,
        pitch: PitchSpec | None = None,
        max_player_speed: float = DEFAULT_MAX_PLAYER_SPEED,
        reaction_time: float = DEFAULT_REACTION_TIME,
        tti_sigma: float = DEFAULT_TTI_SIGMA,
        lambda_att: float = DEFAULT_LAMBDA_ATT,
        lambda_def: float = DEFAULT_LAMBDA_DEF,
        lambda_gk: float = DEFAULT_LAMBDA_GK,
        integration_dt: float = DEFAULT_INTEGRATION_DT,
        integration_t_max: float = DEFAULT_INTEGRATION_T_MAX,
        convergence_threshold: float = DEFAULT_CONVERGENCE_THRESHOLD,
    ) -> None:
        self.resolution = resolution
        self._pitch = pitch
        self.max_player_speed = max_player_speed
        self.reaction_time = reaction_time
        self.tti_sigma = tti_sigma
        self.lambda_att = lambda_att
        self.lambda_def = lambda_def
        self.lambda_gk = lambda_gk
        self.integration_dt = integration_dt
        self.integration_t_max = integration_t_max
        self.convergence_threshold = convergence_threshold

    def _resolve_pitch(self) -> PitchSpec:
        return self._pitch if self._pitch is not None else PitchSpec()

    def _build_tti(
        self,
        positions: np.ndarray,
        is_goalkeeper: np.ndarray,
        targets: np.ndarray,
        is_attacker: bool,
    ) -> tuple[np.ndarray, float, float]:
        """Build TTI matrix and resolve sigma/lambda for a team.

        Parameters:
            positions:     Shape ``(N, 2)``.
            is_goalkeeper: Boolean mask, shape ``(N,)``.
            targets:       Grid centres, shape ``(G, 2)``.
            is_attacker:   True if this is the attacking team.

        Returns:
            Tuple of ``(tti, sigma, lam)`` where ``tti`` has shape ``(N, G)``.
        """
        tti = time_to_intercept(
            positions, targets, self.reaction_time, self.max_player_speed
        )

        # Goalkeepers get a flatter (wider) sigmoid — less decisive
        lam_base = self.lambda_att if is_attacker else self.lambda_def
        if is_goalkeeper.any():
            # Per-player lambda vector applied via broadcasting
            lam_vec = np.where(is_goalkeeper, self.lambda_gk, lam_base)
            # Scale TTI sigma inversely by lambda so the product lam*sigma
            # stays comparable; simplest: just use the mean effective sigma.
            # For this implementation we apply lam per-player via the tti
            # offset trick: effective_tti = tti / lam * lam_base.
            # Cleaner: scale TTI by (lam_base / lam_vec) so that
            # sigmoid_influence(tti_scaled, t, sigma, lam_base) reproduces
            # the per-player lam effect.
            tti = tti * (lam_base / lam_vec[:, np.newaxis])

        sigma = self.tti_sigma
        lam = lam_base
        return tti, sigma, lam

    def control(
        self,
        frame: FrameRecord,
        *,
        team_id: str,
        return_per_player: bool = False,
    ) -> ProbabilityGrid:
        """Compute pitch control for a single frame.

        Parameters:
            frame:   A :class:`FrameRecord` with player positions.
            team_id: The *attacking* team's identifier. Values close to 1
                     indicate this team controls that cell.
            return_per_player: Reserved for future use by the cognitive
                     module; currently has no effect.

        Returns:
            A :class:`ProbabilityGrid` where ``values[i, j]`` is the
            probability that *team_id* controls cell ``(i, j)``.
        """
        pitch = self._resolve_pitch()
        targets, x_edges, y_edges = make_grid(pitch, self.resolution)  # (G, 2)

        att_mask = frame.team_mask(team_id)    # (N,) bool
        def_mask = ~att_mask

        pos_att = frame.positions[att_mask]    # (N_att, 2)
        pos_def = frame.positions[def_mask]    # (N_def, 2)
        gk_att = frame.is_goalkeeper[att_mask] if len(frame.is_goalkeeper) else np.zeros(len(pos_att), bool)
        gk_def = frame.is_goalkeeper[def_mask] if len(frame.is_goalkeeper) else np.zeros(len(pos_def), bool)

        tti_att, sigma_att, lam_att = self._build_tti(
            pos_att, gk_att, targets, is_attacker=True
        )
        tti_def, sigma_def, lam_def = self._build_tti(
            pos_def, gk_def, targets, is_attacker=False
        )

        PPCF_att, _ = accumulate_control(
            tti_att,
            tti_def,
            sigma_att=sigma_att,
            sigma_def=sigma_def,
            lam_att=lam_att,
            lam_def=lam_def,
            dt=self.integration_dt,
            t_max=self.integration_t_max,
            convergence_threshold=self.convergence_threshold,
        )

        nx, ny = self.resolution
        values = np.clip(PPCF_att.reshape(nx, ny), 0.0, 1.0)

        return ProbabilityGrid(
            values=values,
            x_edges=x_edges,
            y_edges=y_edges,
            pitch=pitch,
            timestamp=frame.timestamp,
        )

    def control_batch(
        self,
        frames: list[FrameRecord],
        *,
        team_id: str,
    ) -> list[ProbabilityGrid]:
        """Compute pitch control for a sequence of frames.

        Parameters:
            frames:  List of :class:`FrameRecord` objects.
            team_id: The *attacking* team's identifier.

        Returns:
            List of :class:`ProbabilityGrid` in the same order.
        """
        return [self.control(f, team_id=team_id) for f in frames]
