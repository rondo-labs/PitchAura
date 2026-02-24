"""
Project: PitchAura
File Created: 2026-02-23
Author: Xingnan Zhu
File Name: constants.py
Description:
    Default parameters for spatial models and pitch dimensions.
"""

# --- Pitch ---
PITCH_LENGTH: float = 105.0  # meters (FIFA standard)
PITCH_WIDTH: float = 68.0  # meters (FIFA standard)

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

# --- Grid ---
DEFAULT_GRID_RESOLUTION: tuple[int, int] = (105, 68)  # 1m x 1m cells

# --- Vision Model ---
DEFAULT_CONE_HALF_ANGLE: float = 100.0  # degrees (200° total FOV)
