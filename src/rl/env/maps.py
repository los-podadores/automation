from __future__ import annotations

import cv2
import numpy as np

from .config import (
    ROBOT_RADIUS_PX,
    SPAWN_SAFETY_RADIUS_PX,
    VIRTUAL_MARGIN_PX,
)
from .transforms import m_to_grid_px


def rasterize_field(
    field_grid: np.ndarray,
    field,
    render_offset: np.ndarray,
    pixels_per_meter: float,
) -> tuple[np.ndarray, int, float]:
    """Fill *field_grid* from the Shapely field polygon.

    Returns ``(field_grid, grid_size_p, grid_size_m)``.
    """
    minx, miny, maxx, maxy = field.bounds
    pad = 5.0
    grid_size_m = max(maxx - minx, maxy - miny) + 2 * pad
    grid_size_p = max(1, int(grid_size_m * pixels_per_meter))
    render_offset[:] = np.array([minx - pad, miny - pad])

    field_grid[:] = 0
    exterior = np.array(field.exterior.coords, dtype=np.float32)
    exterior_px = ((exterior - render_offset) * pixels_per_meter).astype(np.int32)
    cv2.fillPoly(field_grid, [exterior_px], 1)

    for interior in field.interiors:
        hole = np.array(interior.coords, dtype=np.float32)
        hole_px = ((hole - render_offset) * pixels_per_meter).astype(np.int32)
        cv2.fillPoly(field_grid, [hole_px], 0)

    return field_grid, grid_size_p, grid_size_m


def compute_static_maps(true_obstacle_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build the binary collision map and the soft virtual-wall distance map.

    Returns ``(collision_map, virtual_wall_map)``.
    """
    obs_for_dilation = true_obstacle_map.copy()
    obs_for_dilation[0, :] = 1
    obs_for_dilation[-1, :] = 1
    obs_for_dilation[:, 0] = 1
    obs_for_dilation[:, -1] = 1

    # Physical collision map
    kernel_phys = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (ROBOT_RADIUS_PX * 2 + 1,) * 2,
    )
    collision_map = cv2.dilate(obs_for_dilation, kernel_phys, iterations=1)

    # Virtual wall map (soft glow around obstacles)
    free_space = (obs_for_dilation == 0).astype(np.uint8)
    dist_transform = cv2.distanceTransform(free_space, cv2.DIST_L2, 5)
    safe_distance_px = float(VIRTUAL_MARGIN_PX + ROBOT_RADIUS_PX)
    dist_clipped = np.clip(dist_transform, 0, safe_distance_px)
    virtual_wall_map = 1.0 - (dist_clipped / safe_distance_px)

    return collision_map, virtual_wall_map


def compute_spawn_safety_map(true_obstacle_map: np.ndarray) -> np.ndarray:
    """Return a binary mask of unsafe spawn regions."""
    obs_for_dilation = true_obstacle_map.copy()
    obs_for_dilation[0, :] = 1
    obs_for_dilation[-1, :] = 1
    obs_for_dilation[:, 0] = 1
    obs_for_dilation[:, -1] = 1

    safety_r_px = SPAWN_SAFETY_RADIUS_PX
    if safety_r_px > 0:
        kernel = np.ones((safety_r_px * 2 + 1,) * 2, dtype=np.float32)
        return cv2.dilate(obs_for_dilation, kernel, iterations=1)
    return obs_for_dilation


def compute_coverable_area(
    field_grid: np.ndarray,
    collision_map: np.ndarray,
    agent_pos_m: np.ndarray,
    render_offset: np.ndarray,
    pixels_per_meter: float,
) -> np.ndarray:
    """Return a binary mask of cells the agent can physically reach."""
    valid_positions = ((field_grid > 0) & (collision_map == 0)).astype(np.uint8)
    free_space = (collision_map == 0).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(free_space, connectivity=4)

    px = m_to_grid_px(agent_pos_m, render_offset, pixels_per_meter)
    spawn_label = labels[px[1], px[0]]
    reachable_mask = (labels == spawn_label).astype(np.uint8)
    return valid_positions & reachable_mask


def init_maps(
    coverage_map: np.ndarray,
    overlap_map: np.ndarray,
    obstacle_map: np.ndarray,
    agent_pos_m: np.ndarray,
    render_offset: np.ndarray,
    pixels_per_meter: float,
    grid_size_p: int,
    collision_map: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Zero-initialise maps, stamp initial coverage, compute frontier.

    Returns ``(coverage_map, overlap_map, obstacle_map, frontier_map, metrics)``.
    """
    obstacle_map[:] = 0
    coverage_map[:] = 0
    overlap_map[:] = 0

    stamp_initial_coverage(
        coverage_map,
        overlap_map,
        collision_map,
        agent_pos_m,
        render_offset,
        pixels_per_meter,
        grid_size_p,
    )
    frontier_map = compute_frontier_map(coverage_map, collision_map)
    return coverage_map, overlap_map, obstacle_map, frontier_map


def stamp_initial_coverage(
    coverage_map: np.ndarray,
    overlap_map: np.ndarray,
    collision_map: np.ndarray,
    pos_m: np.ndarray,
    render_offset: np.ndarray,
    pixels_per_meter: float,
    grid_size_p: int,
) -> None:
    """Stamp a circular footprint at *pos_m* onto the coverage map."""
    px = m_to_grid_px(pos_m, render_offset, pixels_per_meter)
    radius = ROBOT_RADIUS_PX

    min_x = max(0, px[0] - radius - 1)
    max_x = min(grid_size_p, px[0] + radius + 2)
    min_y = max(0, px[1] - radius - 1)
    max_y = min(grid_size_p, px[1] + radius + 2)
    if min_x >= max_x or min_y >= max_y:
        return

    local_cov = coverage_map[min_y:max_y, min_x:max_x]
    local_overlap = overlap_map[min_y:max_y, min_x:max_x]
    local_obs = collision_map[min_y:max_y, min_x:max_x]
    local_h = max_y - min_y
    local_w = max_x - min_x
    local_mask = np.zeros((local_h, local_w), dtype=np.uint8)
    cv2.circle(local_mask, (px[0] - min_x, px[1] - min_y), radius, 1, thickness=-1)
    local_mask[local_obs > 0] = 0
    local_cov[:] = np.maximum(local_cov, local_mask.astype(np.float32))
    local_overlap[:] = local_overlap + local_mask.astype(np.float32)


def stamp_coverage(
    coverage_map: np.ndarray,
    overlap_map: np.ndarray,
    collision_map: np.ndarray,
    old_pos_m: np.ndarray,
    new_pos_m: np.ndarray,
    render_offset: np.ndarray,
    pixels_per_meter: float,
    grid_size_p: int,
) -> tuple[int, tuple[int, int, int, int] | None]:
    """Stamp a swept rectangle between two positions.

    Returns ``(new_pixels, last_stamp_bbox)``.
    """
    old_px = m_to_grid_px(old_pos_m, render_offset, pixels_per_meter)
    new_px = m_to_grid_px(new_pos_m, render_offset, pixels_per_meter)
    radius = ROBOT_RADIUS_PX

    min_x = max(0, min(old_px[0], new_px[0]) - radius - 1)
    max_x = min(grid_size_p, max(old_px[0], new_px[0]) + radius + 2)
    min_y = max(0, min(old_px[1], new_px[1]) - radius - 1)
    max_y = min(grid_size_p, max(old_px[1], new_px[1]) + radius + 2)
    last_stamp_bbox = (min_x, max_x, min_y, max_y)
    if min_x >= max_x or min_y >= max_y:
        return 0, last_stamp_bbox

    local_cov = coverage_map[min_y:max_y, min_x:max_x]
    local_overlap = overlap_map[min_y:max_y, min_x:max_x]
    local_obs = collision_map[min_y:max_y, min_x:max_x]

    local_h = max_y - min_y
    local_w = max_x - min_x
    local_mask = np.zeros((local_h, local_w), dtype=np.uint8)

    ox_local = old_px[0] - min_x
    oy_local = old_px[1] - min_y
    nx_local = new_px[0] - min_x
    ny_local = new_px[1] - min_y

    cv2.line(
        local_mask,
        (ox_local, oy_local),
        (nx_local, ny_local),
        1,
        thickness=2 * radius + 2,
    )
    local_mask[local_obs > 0] = 0

    new_pixels = int(np.logical_and(local_mask, (local_cov == 0)).sum())
    local_cov[:] = np.maximum(local_cov, local_mask.astype(np.float32))
    local_overlap[:] = local_overlap + local_mask.astype(np.float32)

    return new_pixels, last_stamp_bbox


def compute_frontier_map(
    coverage_map: np.ndarray, collision_map: np.ndarray
) -> np.ndarray:
    """Return an exaggerated frontier mask (boundary between covered and free).

    Uses *collision_map* (dilated by robot radius) so the frontier boundary
    aligns with the same navigable-space mask used during coverage stamping.
    """
    cov = coverage_map.copy()
    obs = collision_map.copy()
    obs[0, :] = 1
    obs[-1, :] = 1
    obs[:, 0] = 1
    obs[:, -1] = 1
    cov[obs > 0] = 0
    free = (cov + obs) == 0

    k3 = np.ones((3, 3), dtype=np.float32)
    cov_dilated = cv2.dilate(cov, k3, iterations=1)
    frontier = np.logical_and(cov_dilated, free).astype(np.float32)
    return cv2.dilate(frontier, k3, iterations=1)


def get_local_crop(
    world_map: np.ndarray,
    pos_m: np.ndarray,
    radius_m: float,
    render_offset: np.ndarray,
    pixels_per_meter: float,
    grid_size_p: int,
) -> np.ndarray:
    """Return a local square crop around *pos_m*."""
    pos_p = (pos_m - render_offset) * pixels_per_meter
    r = int(radius_m * pixels_per_meter) + 10
    y1 = max(0, int(pos_p[1]) - r)
    y2 = min(grid_size_p, int(pos_p[1]) + r + 1)
    x1 = max(0, int(pos_p[0]) - r)
    x2 = min(grid_size_p, int(pos_p[0]) + r + 1)
    return world_map[y1:y2, x1:x2].copy()


def update_obstacle_map_from_sensors(
    obstacle_map: np.ndarray,
    virtual_wall_map: np.ndarray,
    hit_points: list[tuple[int, int] | None],
    grid_size_p: int,
) -> None:
    """Inflate obstacle map from ray-cast hit points."""
    inflation_radius = VIRTUAL_MARGIN_PX + ROBOT_RADIUS_PX
    for hp in hit_points:
        if hp is not None:
            ix, iy = hp
            if 0 <= ix < grid_size_p and 0 <= iy < grid_size_p:
                y1 = max(0, iy - inflation_radius)
                y2 = min(grid_size_p, iy + inflation_radius + 1)
                x1 = max(0, ix - inflation_radius)
                x2 = min(grid_size_p, ix + inflation_radius + 1)
                perfect_wall_patch = virtual_wall_map[y1:y2, x1:x2]
                obstacle_map[y1:y2, x1:x2] = np.maximum(
                    obstacle_map[y1:y2, x1:x2],
                    perfect_wall_patch,
                )
