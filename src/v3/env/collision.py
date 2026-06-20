from __future__ import annotations

import numpy as np

from .config import ROBOT_RADIUS_PX


def is_out_of_bounds(
    pos_m: np.ndarray,
    render_offset: np.ndarray,
    pixels_per_meter: float,
    grid_size_p: int,
    field_grid: np.ndarray,
) -> bool:
    """Check whether *pos_m* is outside the field or beyond the grid edges."""
    pos_p = (pos_m - render_offset) * pixels_per_meter
    r = ROBOT_RADIUS_PX
    x, y = int(pos_p[0]), int(pos_p[1])
    if x - r < 0 or x + r >= grid_size_p:
        return True
    if y - r < 0 or y + r >= grid_size_p:
        return True
    return field_grid[y, x] == 0


def is_obstacle_collision(
    pos_m: np.ndarray,
    render_offset: np.ndarray,
    pixels_per_meter: float,
    grid_size_p: int,
    collision_map: np.ndarray,
) -> bool:
    """Check whether *pos_m* overlaps a dilated obstacle pixel."""
    pos_p = (pos_m - render_offset) * pixels_per_meter
    ix, iy = int(round(pos_p[0])), int(round(pos_p[1]))
    if ix < 0 or ix >= grid_size_p or iy < 0 or iy >= grid_size_p:
        return True
    return collision_map[iy, ix] > 0


def check_collision(
    pos_m: np.ndarray,
    render_offset: np.ndarray,
    pixels_per_meter: float,
    grid_size_p: int,
    field_grid: np.ndarray,
    collision_map: np.ndarray,
) -> bool:
    """Return True if *pos_m* is out of bounds **or** collides with an obstacle."""
    return is_out_of_bounds(
        pos_m,
        render_offset,
        pixels_per_meter,
        grid_size_p,
        field_grid,
    ) or is_obstacle_collision(
        pos_m,
        render_offset,
        pixels_per_meter,
        grid_size_p,
        collision_map,
    )
