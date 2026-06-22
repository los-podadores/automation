from __future__ import annotations

import math

import numpy as np
from shapely.geometry import MultiPolygon, Point, Polygon

from .config import (
    METERS_PER_PIXEL,
    ROBOT_RADIUS,
    ROBOT_RADIUS_PX,
    ROBOT_SIDE,
    VIRTUAL_MARGIN_PX,
)


def generate_random_field(np_random: np.random.Generator, phase_cfg: dict) -> Polygon:
    """Generate a random convex-ish polygon field with circular obstacles.

    Keeps retrying until the result is a single ``Polygon`` (not a
    ``MultiPolygon`` after subtracting obstacles).
    """
    radii_low, radii_high = phase_cfg["radii"]
    obst_min, obst_max = phase_cfg["obst"]
    obs_rad_min, obs_rad_max = phase_cfg["obs_rad"]

    safe_margin_m = (ROBOT_RADIUS_PX + VIRTUAL_MARGIN_PX) * METERS_PER_PIXEL + 0.5

    while True:
        angles = np.sort(np_random.uniform(0, 2 * np.pi, 12))
        radii = np_random.uniform(radii_low, radii_high, 12)
        points = [
            (r * math.cos(a), r * math.sin(a))
            for r, a in zip(radii, angles, strict=True)
        ]
        outer = Polygon(points).buffer(0.5).simplify(0.3)

        num_obstacles = (
            np_random.integers(obst_min, obst_max + 1) if obst_max > 0 else 0
        )
        obstacles: list[Polygon] = []

        for _ in range(num_obstacles):
            lo, la, hi, ha = outer.bounds
            if hi - lo < 2 * safe_margin_m or ha - la < 2 * safe_margin_m:
                break
            ox = np_random.uniform(lo + safe_margin_m, hi - safe_margin_m)
            oy = np_random.uniform(la + safe_margin_m, ha - safe_margin_m)

            obs_poly = (
                Point(ox, oy)
                .buffer(np_random.uniform(obs_rad_min, obs_rad_max))
                .simplify(0.2)
            )

            if (
                outer.contains(obs_poly)
                and outer.boundary.distance(obs_poly) > safe_margin_m
            ):
                too_close = any(obs_poly.distance(e) < safe_margin_m for e in obstacles)
                if not too_close:
                    obstacles.append(obs_poly)

        field: Polygon = outer
        for obs in obstacles:
            field = field.difference(obs)

        if not isinstance(field, MultiPolygon):
            return field


def validate_field(field: Polygon) -> bool:
    """Return True if the field is navigable after erosion."""
    erosion = ROBOT_RADIUS
    nav = field.buffer(-erosion)
    return not (nav.is_empty or isinstance(nav, MultiPolygon))


def get_safe_spawn(
    field: Polygon,
    np_random: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Find a valid spawn position along the field boundary.

    Returns ``(agent_pos_m, agent_heading)``.
    Raises ``RuntimeError`` if no valid position is found within 100 attempts.
    """
    for _ in range(100):
        boundary = list(field.exterior.coords)
        num_edges = len(boundary) - 1
        edge_idx = np_random.integers(0, num_edges)
        x1, y1 = boundary[edge_idx]
        x2, y2 = boundary[(edge_idx + 1) % num_edges]
        t = np_random.uniform(0.25, 0.75)
        px = x1 + t * (x2 - x1)
        py = y1 + t * (y2 - y1)

        dx, dy = x2 - x1, y2 - y1
        edge_len = math.hypot(dx, dy)
        nx, ny = -dy / edge_len, dx / edge_len
        inward = field.representative_point()
        if nx * (inward.x - px) + ny * (inward.y - py) < 0:
            nx, ny = -nx, -ny

        spawn_dist = ROBOT_RADIUS + 0.5 * ROBOT_SIDE
        x = px + nx * spawn_dist
        y = py + ny * spawn_dist
        theta = math.atan2(dy, dx)

        agent_pos_m = np.array([x, y], dtype=np.float64)
        agent_heading = theta

        return agent_pos_m, agent_heading

    raise RuntimeError("Failed to find valid spawn position")
