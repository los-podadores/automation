from __future__ import annotations

import math

import cv2
import numpy as np
import pygame

from .config import METERS_PER_PIXEL, RAY_COLORS, ROBOT_RADIUS, ROBOT_SIDE, RAY_MAX_DIST
from .transforms import m_to_grid_px, m_to_p


def get_square_corners(pos_m: np.ndarray, heading: float) -> np.ndarray:
    """Return the four world-space corners of the square robot footprint."""
    half = ROBOT_RADIUS
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)
    corners_local = [(-half, -half), (half, -half), (half, half), (-half, half)]
    corners = []
    for lx, ly in corners_local:
        gx = pos_m[0] + lx * cos_h - ly * sin_h
        gy = pos_m[1] + lx * sin_h + ly * cos_h
        corners.append([gx, gy])
    return np.array(corners, dtype=np.float64)


def draw_robot_footprint_local(
    pos_m: np.ndarray,
    heading: float,
    local_size: int,
    render_offset: np.ndarray,
    pixels_per_meter: float,
) -> np.ndarray:
    """Return a ``(local_size, local_size)`` binary mask of the robot footprint."""
    half = ROBOT_RADIUS
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)
    corners = []
    for lx, ly in [(-half, -half), (half, -half), (half, half), (-half, half)]:
        gx = pos_m[0] + lx * cos_h - ly * sin_h
        gy = pos_m[1] + lx * sin_h + ly * cos_h
        corners.append([gx, gy])
    corners = np.array(corners, dtype=np.float64)
    center_p = m_to_p(np.array(pos_m), render_offset, pixels_per_meter)
    corners_p = m_to_p(corners, render_offset, pixels_per_meter)
    local_offset = corners_p - center_p + local_size // 2
    footprint = np.zeros((local_size, local_size), dtype=np.uint8)
    cv2.fillConvexPoly(footprint, local_offset.astype(np.int32), 1)
    return footprint


def render_frame(
    env,
    toggles: dict[str, bool],
    window_size: int,
) -> np.ndarray:
    """Render the full debug frame and return an ``(H, W, 3)`` RGB array."""
    ws = window_size
    canvas = pygame.Surface((ws, ws))
    canvas.fill((30, 30, 30))

    if env.grid_size_p <= 0 or env.field is None:
        return np.transpose(pygame.surfarray.array3d(canvas), (1, 0, 2))

    pad = 5.0
    minx, miny, maxx, maxy = env.field.bounds
    width = (maxx - minx) + 2 * pad
    height = (maxy - miny) + 2 * pad
    scl = ws / max(width, height)
    off = np.array([minx - pad, miny - pad])

    def to_screen(wx: float, wy: float) -> tuple[int, int]:
        px = int((wx - off[0]) * scl)
        py = int(ws - (wy - off[1]) * scl)
        return px, py

    # --- base map ---
    img = np.full((env.grid_size_p, env.grid_size_p, 3), 30, dtype=np.uint8)
    field_mask = env.field_grid > 0
    img[field_mask] = [220, 220, 220]
    cov_mask = env.coverage_map > 0
    img[cov_mask] = [80, 160, 80]

    if toggles.get("dilated", False):
        dilated_mask = env.virtual_wall_map > 0
        img[dilated_mask] = [50, 140, 200]

    if (
        toggles.get("coverable", False)
        and hasattr(env, "coverable_area")
        and env.coverable_area is not None
    ):
        coverable_mask = env.coverable_area > 0
        img[coverable_mask] = [0, 180, 180]

    obs_mask = env.obstacle_map > 0
    img[obs_mask] = (0.5 * img[obs_mask] + 0.5 * np.array([200, 80, 80])).astype(
        np.uint8
    )

    img = cv2.resize(img, (ws, ws), interpolation=cv2.INTER_NEAREST)
    img = cv2.cvtColor(img[::-1], cv2.COLOR_BGR2RGB)
    screen_arr = np.transpose(img, (1, 0, 2))
    surf = pygame.surfarray.make_surface(screen_arr)
    canvas.blit(surf, (0, 0))

    # --- field boundary ---
    ext_points = [to_screen(x, y) for x, y in env.field.exterior.coords]
    pygame.draw.polygon(canvas, (0, 0, 0), ext_points, 2)
    for interior in env.field.interiors:
        in_points = [to_screen(x, y) for x, y in interior.coords]
        pygame.draw.polygon(canvas, (255, 255, 255), in_points)
        pygame.draw.polygon(canvas, (255, 0, 0), in_points, 1)

    # --- robot ---
    corners = get_square_corners(env.agent_pos_m, env.agent_heading)
    corner_pg = [to_screen(c[0], c[1]) for c in corners]
    pygame.draw.polygon(canvas, (50, 50, 200), corner_pg)

    hx = env.agent_pos_m[0] + ROBOT_RADIUS * math.cos(env.agent_heading)
    hy = env.agent_pos_m[1] + ROBOT_RADIUS * math.sin(env.agent_heading)
    pygame.draw.line(
        canvas, (0, 255, 0), to_screen(*env.agent_pos_m), to_screen(hx, hy), 2
    )

    # --- sensor rays ---
    a = ROBOT_SIDE
    theta = env.agent_heading
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    def local_pt(lx: float, ly: float) -> tuple[float, float]:
        return (
            env.agent_pos_m[0] + lx * cos_t - ly * sin_t,
            env.agent_pos_m[1] + lx * sin_t + ly * cos_t,
        )

    origins = [
        local_pt(a / 2, a / 2),
        local_pt(a / 2, 0),
        local_pt(a / 2, -a / 2),
        local_pt(0, a / 2),
        local_pt(0, -a / 2),
        local_pt(-a / 2, 0),
    ]
    ray_angles = [math.pi / 4, 0.0, -math.pi / 4, math.pi / 2, -math.pi / 2, math.pi]

    sensors = env._last_sensors  # already computed in step/reset
    for i, (origin, ang_off) in enumerate(zip(origins, ray_angles, strict=True)):
        ang = theta + ang_off
        dist = sensors[i] * RAY_MAX_DIST
        end = (origin[0] + math.cos(ang) * dist, origin[1] + math.sin(ang) * dist)
        pygame.draw.line(canvas, RAY_COLORS[i], to_screen(*origin), to_screen(*end), 2)
        pygame.draw.circle(canvas, (255, 255, 255), to_screen(*end), 3)
        pygame.draw.circle(canvas, RAY_COLORS[i], to_screen(*origin), 3)

        if toggles.get("rays", False):
            _draw_ray_pixels(canvas, env, origin, end, RAY_COLORS[i], to_screen)

    # --- stamped bbox ---
    if toggles.get("stamped", False) and env._last_stamp_bbox is not None:
        _draw_stamp_bbox(canvas, env, to_screen)

    return np.transpose(pygame.surfarray.array3d(canvas), (1, 0, 2))


def _draw_ray_pixels(
    canvas: pygame.Surface,
    env,
    origin: tuple[float, float],
    end: tuple[float, float],
    color: tuple[int, int, int],
    to_screen,
) -> None:
    """Bresenham pixel walk for debug ray visualisation."""
    ox_p, oy_p = m_to_grid_px(np.array(origin), env.render_offset, env.pixels_per_meter)
    ex_p, ey_p = m_to_grid_px(np.array(end), env.render_offset, env.pixels_per_meter)
    adx = abs(ex_p - ox_p)
    ady = abs(ey_p - oy_p)
    if adx == 0 and ady == 0:
        return
    sx_step = 1 if ex_p > ox_p else -1
    sy_step = 1 if ey_p > oy_p else -1
    err = adx - ady
    cx, cy = ox_p, oy_p
    max_it = adx + ady + 1
    for _ in range(max_it):
        if 0 <= cx < env.grid_size_p and 0 <= cy < env.grid_size_p:
            wx = cx * METERS_PER_PIXEL + env.render_offset[0]
            wy = cy * METERS_PER_PIXEL + env.render_offset[1]
            pygame.draw.circle(canvas, color, to_screen(wx, wy), 1)
        if cx == ex_p and cy == ey_p:
            break
        e2 = 2 * err
        if e2 > -ady:
            err -= ady
            cx += sx_step
        if e2 < adx:
            err += adx
            cy += sy_step


def _draw_stamp_bbox(canvas: pygame.Surface, env, to_screen) -> None:
    """Draw the yellow bounding box of the last coverage stamp."""
    bb = env._last_stamp_bbox
    min_x, max_x, min_y, max_y = bb
    corners_bb = [
        to_screen(
            min_x * METERS_PER_PIXEL + env.render_offset[0],
            max_y * METERS_PER_PIXEL + env.render_offset[1],
        ),
        to_screen(
            max_x * METERS_PER_PIXEL + env.render_offset[0],
            max_y * METERS_PER_PIXEL + env.render_offset[1],
        ),
        to_screen(
            max_x * METERS_PER_PIXEL + env.render_offset[0],
            min_y * METERS_PER_PIXEL + env.render_offset[1],
        ),
        to_screen(
            min_x * METERS_PER_PIXEL + env.render_offset[0],
            min_y * METERS_PER_PIXEL + env.render_offset[1],
        ),
    ]
    pygame.draw.lines(canvas, (255, 200, 0), True, corners_bb, 2)
