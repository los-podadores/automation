"""Spatial map builder for converting 3D obstacle tracks and walls into RL policy observations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np

from src.translate.config import LIDAR_FOV_RAD, LIDAR_MAX_RANGE, LIDAR_NUM_RAYS
from src.translate.fusion_types import (
    CLASS_FREE,
    CLASS_STATIC_HAZARD,
    GoalState,
    TrackedObstacle,
)

NUM_RAYS: Final[int] = LIDAR_NUM_RAYS
MAX_RANGE_METERS: Final[float] = LIDAR_MAX_RANGE
RAY_FOV_RAD: Final[float] = LIDAR_FOV_RAD  # 360 degrees surround (-pi to +pi)
ANGULAR_RESOLUTION: Final[float] = RAY_FOV_RAD / NUM_RAYS


@dataclass(frozen=True)
class EgoKinematics:
    """Vehicle dynamic state in SI units."""

    vx: float  # Longitudinal velocity (m/s)
    vy: float  # Lateral velocity (m/s)
    steering_angle: float  # Current wheel steering angle (rad)
    yaw_rate: float  # Heading change rate (rad/s)
    ax: float  # Longitudinal acceleration (m/s^2)
    ay: float  # Lateral acceleration (m/s^2)

    def to_array(self) -> np.ndarray:
        return np.array(
            [self.vx, self.vy, self.steering_angle, self.yaw_rate, self.ax, self.ay],
            dtype=np.float32,
        )


class SpatialMapBuilder:
    """Encodes 360-degree tracked obstacles, walls, and ego state into an observation vector."""

    def __init__(
        self,
        num_rays: int = NUM_RAYS,
        max_range: float = MAX_RANGE_METERS,
    ) -> None:
        self.num_rays = num_rays
        self.max_range = max_range
        # Angles for each ray center covering full 360 degrees: [-pi, +pi)
        self.ray_angles = np.linspace(-math.pi, math.pi, self.num_rays, endpoint=False)

    def compute_ray_scan(
        self,
        obstacles: list[TrackedObstacle],
        wall_segments_ego: list[tuple[tuple[float, float], tuple[float, float]]] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Cast 72 radial rays across the full 360-degree surround perimeter.

        Args:
            obstacles: List of tracked obstacles in vehicle ego frame.
            wall_segments_ego: Optional 2D line segments ((x1, y1), (x2, y2)) of walls in ego frame.

        Returns:
            distances: (72,) normalized float array in [0.0, 1.0].
            classes: (72,) float array of spatial class codes (0=Free, 1=Vehicle, 2=Ped, 3=Static).
        """
        distances = np.full(self.num_rays, self.max_range, dtype=np.float32)
        classes = np.full(self.num_rays, CLASS_FREE, dtype=np.float32)

        # 1. Cast against point/bounding obstacles
        for obs in obstacles:
            dist = obs.distance
            if dist > self.max_range + obs.radius:
                continue

            azimuth = obs.azimuth  # rad: -pi to +pi
            ang_radius = math.asin(min(1.0, obs.radius / max(dist, 0.1)))

            # Angular difference wrapped to [-pi, pi]
            ang_diff = np.arctan2(
                np.sin(self.ray_angles - azimuth),
                np.cos(self.ray_angles - azimuth),
            )
            hit_mask = np.abs(ang_diff) <= ang_radius
            ray_indices = np.where(hit_mask)[0]

            if len(ray_indices) == 0:
                closest_idx = int(np.argmin(np.abs(ang_diff)))
                ray_indices = np.array([closest_idx])

            obs_code = float(obs.to_spatial_code())
            boundary_dist = max(0.0, dist - obs.radius)

            for idx in ray_indices:
                if boundary_dist < distances[idx]:
                    distances[idx] = boundary_dist
                    classes[idx] = obs_code

        # 2. Cast against wall segments in ego coordinates
        if wall_segments_ego:
            for (wx1, wy1), (wx2, wy2) in wall_segments_ego:
                # Segment vector
                dx = wx2 - wx1
                dy = wy2 - wy1
                seg_len_sq = dx * dx + dy * dy
                if seg_len_sq < 1e-4:
                    continue

                for i, r_ang in enumerate(self.ray_angles):
                    rx = math.cos(r_ang)
                    ry = math.sin(r_ang)

                    # Ray from (0, 0) in direction (rx, ry) vs line segment (wx1, wy1) to (wx2, wy2)
                    det = dx * ry - dy * rx
                    if abs(det) < 1e-6:
                        continue

                    # t is ray distance, u is segment parameter [0, 1]
                    t = (dx * wy1 - dy * wx1) / det
                    u = (rx * wy1 - ry * wx1) / det

                    if t > 0.0 and 0.0 <= u <= 1.0 and t < distances[i]:
                        distances[i] = t
                        classes[i] = float(CLASS_STATIC_HAZARD)

        # Normalize distance to [0.0, 1.0]
        norm_distances = np.clip(distances / self.max_range, 0.0, 1.0)
        return norm_distances, classes

    def build_observation(
        self,
        kinematics: EgoKinematics,
        goal_state: GoalState,
        obstacles: list[TrackedObstacle],
        wall_segments_ego: list[tuple[tuple[float, float], tuple[float, float]]] | None = None,
    ) -> np.ndarray:
        """Assemble the complete 156-float observation vector for RLlib.

        Layout:
          [0:6]    - EgoKinematics (vx, vy, steering, yaw_rate, ax, ay)
          [6:12]   - GoalState (dx_ego, dy_ego, distance, heading_error, cross_track_error, path_heading_error)
          [12:84]  - Normalized radial distances (72 floats in [0, 1])
          [84:156] - Categorical obstacle classes (72 floats in {0, 1, 2, 3})
        """
        k_arr = kinematics.to_array()
        g_arr = goal_state.to_array()
        dist_arr, cls_arr = self.compute_ray_scan(obstacles, wall_segments_ego=wall_segments_ego)

        obs = np.concatenate([k_arr, g_arr, dist_arr, cls_arr]).astype(np.float32)
        assert len(obs) == 156, f"Observation size mismatch: expected 156, got {len(obs)}"
        return obs

