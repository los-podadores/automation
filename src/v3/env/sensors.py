from __future__ import annotations

import math

import numpy as np

from .config import MAP_SIZE, METERS_PER_PIXEL, RAY_MAX_DIST, ROBOT_SIDE
from .transforms import local_to_global, m_to_grid_px


def cast_ray_pixel(
    origin_m: np.ndarray,
    angle: float,
    render_offset: np.ndarray,
    pixels_per_meter: float,
    grid_size_p: int,
    true_obstacle_map: np.ndarray,
) -> tuple[float, tuple[int, int] | None]:
    """Bresenham ray-cast from *origin_m* at *angle* until an obstacle or edge."""
    origin_p = (np.asarray(origin_m) - render_offset) * pixels_per_meter
    sx, sy = int(origin_p[0]), int(origin_p[1])
    max_steps = int(RAY_MAX_DIST / METERS_PER_PIXEL)
    dx = math.cos(angle)
    dy = math.sin(angle)
    ex = sx + int(round(dx * max_steps))
    ey = sy + int(round(dy * max_steps))

    abs_dx = abs(ex - sx)
    abs_dy = abs(ey - sy)
    if abs_dx == 0 and abs_dy == 0:
        return RAY_MAX_DIST, None
    step_x = 1 if ex > sx else -1
    step_y = 1 if ey > sy else -1
    err = abs_dx - abs_dy

    cx, cy = sx, sy
    for _ in range(max_steps + 1):
        if 0 <= cx < grid_size_p and 0 <= cy < grid_size_p:
            if true_obstacle_map[cy, cx] > 0:
                hit_m = cx * METERS_PER_PIXEL + render_offset[0]
                hit_my = cy * METERS_PER_PIXEL + render_offset[1]
                dist = math.hypot(hit_m - origin_m[0], hit_my - origin_m[1])
                return min(dist, RAY_MAX_DIST), (cx, cy)
        else:
            return RAY_MAX_DIST, None
        e2 = 2 * err
        if e2 > -abs_dy:
            err -= abs_dy
            cx += step_x
        if e2 < abs_dx:
            err += abs_dx
            cy += step_y
    return RAY_MAX_DIST, None


def compute_sensors(
    agent_pos_m: np.ndarray,
    agent_heading: float,
    last_v: float,
    last_w: float,
    frontier_map: np.ndarray,
    render_offset: np.ndarray,
    pixels_per_meter: float,
    grid_size_p: int,
    true_obstacle_map: np.ndarray,
) -> tuple[np.ndarray, list[tuple[int, int] | None]]:
    """Return the sensor vector *(SENSOR_DIM,)* and hit-point pixel coords."""
    a = ROBOT_SIDE
    b = ROBOT_SIDE
    origins_local = [
        (a / 2, b / 2),
        (a / 2, 0),
        (a / 2, -b / 2),
        (0, b / 2),
        (0, -b / 2),
        (-a / 2, 0),
    ]
    angles_offset = [
        math.pi / 4,
        0.0,
        -math.pi / 4,
        math.pi / 2,
        -math.pi / 2,
        math.pi,
    ]

    dists: list[float] = []
    hit_points: list[tuple[int, int] | None] = []
    for (lx, ly), ang_off in zip(origins_local, angles_offset, strict=True):
        ox, oy = local_to_global(lx, ly, agent_pos_m, agent_heading)
        dist, hit = cast_ray_pixel(
            np.array([ox, oy]),
            agent_heading + ang_off,
            render_offset,
            pixels_per_meter,
            grid_size_p,
            true_obstacle_map,
        )
        dists.append(dist)
        hit_points.append(hit)

    normalized_dists = [d / RAY_MAX_DIST for d in dists]

    # # Homing beacon to closest frontier
    # frontier_y, frontier_x = np.where(frontier_map > 0)
    # if len(frontier_x) == 0:
    #     homing_dist, homing_cos, homing_sin = 0.0, 0.0, 0.0
    # else:
    #     agent_px, agent_py = m_to_grid_px(agent_pos_m, render_offset, pixels_per_meter)
    #     distances = np.sqrt((frontier_x - agent_px) ** 2 + (frontier_y - agent_py) ** 2)
    #     min_idx = np.argmin(distances)

    #     dy = frontier_y[min_idx] - agent_py
    #     dx = frontier_x[min_idx] - agent_px
    #     target_angle = math.atan2(dy, dx)
    #     rel_angle = target_angle - agent_heading

    #     max_map_dist = MAP_SIZE * pixels_per_meter
    #     homing_dist = min(distances[min_idx] / max_map_dist, 1.0)
    #     homing_cos = math.cos(rel_angle)
    #     homing_sin = math.sin(rel_angle)

    sensors = np.array(
        normalized_dists + [last_v, last_w],
        dtype=np.float32,
    )
    return sensors, hit_points


def get_distance_to_closest_frontier(
    frontier_map: np.ndarray,
    agent_pos_m: np.ndarray,
    render_offset: np.ndarray,
    pixels_per_meter: float,
    grid_size_p: int,
) -> float:
    """Return the Euclidean distance (in metres) to the nearest frontier pixel."""
    frontier_y, frontier_x = np.where(frontier_map > 0)
    if len(frontier_x) == 0:
        return 0.0
    agent_px, agent_py = m_to_grid_px(agent_pos_m, render_offset, pixels_per_meter)
    distances = np.sqrt((frontier_x - agent_px) ** 2 + (frontier_y - agent_py) ** 2)
    return float(np.min(distances)) * METERS_PER_PIXEL
