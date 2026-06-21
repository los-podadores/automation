from __future__ import annotations

from typing import Any

# --- Robot geometry ---
ROBOT_SIDE: float = 1.0
ROBOT_RADIUS: float = ROBOT_SIDE / 2.0

# --- Sensor layout ---
RAY_COLORS: list[tuple[int, int, int]] = [
    (255, 165, 0),
    (0, 255, 200),
    (255, 165, 0),
    (100, 100, 255),
    (100, 100, 255),
    (200, 200, 200),
]
NUM_RAYS: int = 6
RAY_MAX_DIST: float = 1.0
SENSOR_DIM: int = 11

# --- Reward weights ---
REWARD_BASE_PENALTY: float = -0.05
REWARD_COLLISION: float = -3.0
REWARD_TV_SCALE: float = (
    2.0  # TODO: if things still dont improve this is the next value to reduce
)
REWARD_TV_MAX: float = 3.0
REWARD_AREA_SCALE: float = 1.5
REWARD_AREA_MAX: float = 2.0
# --- Dynamics ---
ROBOT_SPEED_V: float = 0.15
ROBOT_SPEED_W: float = 1.0
DT: float = 0.5

# --- Map / grid ---
METERS_PER_PIXEL: float = 0.1
NUM_MAPS: int = 4
MAP_SIZE: int = 32
SCALES: list[int] = [1, 3, 7, 20]

# --- Noise ---
POSITION_NOISE: float = 0.01
HEADING_NOISE: float = 0.05

# --- Collision / safety ---
OBSTACLE_DILATION: int = 1
ROBOT_RADIUS_PX: int = 5
VIRTUAL_MARGIN_PX: int = 5
SPAWN_SAFETY_RADIUS_PX: int = 1

# --- Episode / curriculum ---
MAX_FIELD_ATTEMPTS: int = 100
SUCCESS_WINDOW: int = 50
SUCCESS_THRESHOLD: float = 0.8
MAX_NON_NEW_STEPS: int = 750
CELLS_MISSED_THRESHOLD: int = 20
PHASE_WEIGHT_DECAY: float = 0.5

PHASES: dict[int, dict[str, Any]] = {
    1: {"radii": (2.5, 6.0), "obst": (1, 2), "obs_rad": (0.4, 0.8), "max_steps": 3500},
    2: {"radii": (6.0, 8.0), "obst": (2, 3), "obs_rad": (0.5, 1.0), "max_steps": 4429},
    3: {"radii": (8.0, 9.5), "obst": (2, 3), "obs_rad": (0.7, 1.5), "max_steps": 5357},
    4: {"radii": (9.5, 11.0), "obst": (3, 4), "obs_rad": (0.8, 2.0), "max_steps": 6286},
    5: {
        "radii": (11.0, 13.0),
        "obst": (4, 5),
        "obs_rad": (1.0, 2.5),
        "max_steps": 7214,
    },
    6: {
        "radii": (13.0, 14.5),
        "obst": (5, 6),
        "obs_rad": (1.0, 2.5),
        "max_steps": 8143,
    },
    7: {
        "radii": (14.5, 16.0),
        "obst": (6, 8),
        "obs_rad": (1.5, 3.0),
        "max_steps": 9071,
    },
    8: {
        "radii": (16.0, 18.0),
        "obst": (7, 9),
        "obs_rad": (1.5, 4.0),
        "max_steps": 10000,
    },
}
