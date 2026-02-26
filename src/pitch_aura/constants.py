"""
Project: PitchAura
File Created: 2026-02-23
Author: Xingnan Zhu
File Name: constants.py
Description:
    Default parameters for spatial models and pitch dimensions.
"""

# Shared constants (re-exported from pitch-core)
from pitch_core.constants import (  # noqa: F401
    DEFAULT_GRID_RESOLUTION,
    PITCH_LENGTH,
    PITCH_WIDTH,
)

# --- Kinematic Control Model (Spearman) ---
DEFAULT_MAX_PLAYER_SPEED: float = 5.0  # m/s
DEFAULT_REACTION_TIME: float = 0.7  # seconds
DEFAULT_TTI_SIGMA: float = 0.45  # time-to-intercept uncertainty (seconds)
DEFAULT_LAMBDA_ATT: float = 4.3  # attacker influence weight
DEFAULT_LAMBDA_DEF: float = 4.3  # defender influence weight
DEFAULT_LAMBDA_GK: float = 3.0  # goalkeeper influence weight (reduced)
DEFAULT_INTEGRATION_DT: float = 0.04  # integration timestep (seconds)
DEFAULT_INTEGRATION_T_MAX: float = 10.0  # max integration time (seconds)
DEFAULT_CONVERGENCE_THRESHOLD: float = 0.01  # early stopping threshold

# --- Vision Model ---
DEFAULT_CONE_HALF_ANGLE: float = 100.0  # degrees (200° total FOV)
