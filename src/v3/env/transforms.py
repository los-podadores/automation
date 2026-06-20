from __future__ import annotations

import math

import cv2
import numpy as np

from .config import MAP_SIZE


def m_to_p(
    pos_m: np.ndarray, render_offset: np.ndarray, pixels_per_meter: float
) -> np.ndarray:
    """Convert metres to pixel coordinates."""
    return (pos_m - render_offset) * pixels_per_meter


def p_to_m(
    pos_p: np.ndarray, render_offset: np.ndarray, pixels_per_meter: float
) -> np.ndarray:
    """Convert pixel coordinates to metres."""
    return pos_p / pixels_per_meter + render_offset


def m_to_grid_px(
    pos_m: np.ndarray,
    render_offset: np.ndarray,
    pixels_per_meter: float,
) -> tuple[int, int]:
    """Return ``(col, row)`` integer pixel position on the world grid."""
    pos_p = m_to_p(np.asarray(pos_m), render_offset, pixels_per_meter)
    return int(round(pos_p[0])), int(round(pos_p[1]))


def local_to_global(
    lx: float,
    ly: float,
    agent_pos_m: np.ndarray,
    agent_heading: float,
) -> tuple[float, float]:
    """Rotate *(lx, ly)* by *agent_heading* and translate to *agent_pos_m*."""
    x, y = agent_pos_m
    cos_t = math.cos(agent_heading)
    sin_t = math.sin(agent_heading)
    gx = x + lx * cos_t - ly * sin_t
    gy = y + lx * sin_t + ly * cos_t
    return gx, gy


def get_transform_matrix(
    scale: int,
    noisy_heading: float,
    noisy_pos_m: np.ndarray,
    render_offset: np.ndarray,
    pixels_per_meter: float,
) -> np.ndarray:
    """Build a 3×3 affine matrix that centres the view on the agent."""
    heading_deg = noisy_heading * 180 / math.pi
    noisy_p = m_to_p(noisy_pos_m, render_offset, pixels_per_meter)

    t1 = np.eye(3)
    t1[0, 2] = -noisy_p[0] / scale
    t1[1, 2] = -noisy_p[1] / scale

    rot = np.eye(3)
    rot[:2] = cv2.getRotationMatrix2D(center=(0, 0), angle=heading_deg, scale=1)

    t2 = np.eye(3)
    t2[0, 2] = MAP_SIZE / 2
    t2[1, 2] = MAP_SIZE / 2

    return t2 @ rot @ t1


def get_relative_map(
    world_map: np.ndarray,
    pad_value: float,
    scale: int,
    is_frontier: bool,
    grid_size_p: int,
    transform_matrix: np.ndarray,
) -> np.ndarray:
    """Down-sample *world_map* by *scale*, then warp to agent-centric view."""
    sc = min(scale, grid_size_p)

    if is_frontier:
        kernel_size = int(math.ceil(sc))
        if kernel_size > 1:
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            world_map = cv2.dilate(world_map, kernel)
        interp_method = cv2.INTER_NEAREST
    else:
        interp_method = cv2.INTER_AREA

    downsampled = cv2.resize(
        world_map,
        (int(0.5 + grid_size_p / sc),) * 2,
        interpolation=interp_method,
    )
    return cv2.warpAffine(
        downsampled,
        M=transform_matrix[:2],
        dsize=(MAP_SIZE,) * 2,
        borderValue=pad_value,
        flags=cv2.INTER_NEAREST if is_frontier else cv2.INTER_AREA,
    )


def get_multi_scale_map(
    world_map: np.ndarray,
    pad_value: float,
    is_frontier: bool,
    grid_size_p: int,
    noisy_heading: float,
    noisy_pos_m: np.ndarray,
    render_offset: np.ndarray,
    pixels_per_meter: float,
) -> np.ndarray:
    """Return a ``(NUM_MAPS, MAP_SIZE, MAP_SIZE)`` array of multi-scale views."""
    from .config import NUM_MAPS, SCALES

    ms = np.zeros((NUM_MAPS, MAP_SIZE, MAP_SIZE), dtype=np.float32)
    for i, s in enumerate(SCALES):
        matrix = get_transform_matrix(
            s,
            noisy_heading,
            noisy_pos_m,
            render_offset,
            pixels_per_meter,
        )
        ms[i] = get_relative_map(
            world_map,
            pad_value,
            s,
            is_frontier,
            grid_size_p,
            matrix,
        )
    return ms
