"""Spatial map builder for converting segmented 3D LiDAR clouds into RL policy observations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np

from src.translate.config import (
    LIDAR_FOV_RAD,
    LIDAR_MAX_RANGE,
    LIDAR_NUM_RAYS,
)
from src.translate.fusion_types import (
    CLASS_FREE,
    CLASS_STATIC_HAZARD,
    GoalState,
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

    def fuse_point_cloud_obstacles(
        self,
        obstacle_points: np.ndarray | None,
        negative_hazards: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project segmented 3D non-ground obstacle points into 360-degree polar range and classification grid.

        Args:
            obstacle_points: (N, 3) or (N, 2) Cartesian hit coordinates in vehicle ego frame.
            negative_hazards: Optional (M, 3) or (M, 2) drop-off / ditch coordinates in ego frame.

        Returns:
            distances: (72,) normalized float array in [0.0, 1.0].
            classes: (72,) float array of spatial class codes (FREE or STATIC_HAZARD).
        """
        distances = np.full(self.num_rays, self.max_range, dtype=np.float32)
        classes = np.full(self.num_rays, CLASS_FREE, dtype=np.float32)

        all_obs_pts: list[np.ndarray] = []
        if obstacle_points is not None and obstacle_points.size > 0:
            all_obs_pts.append(obstacle_points[:, :2])
        if negative_hazards is not None and negative_hazards.size > 0:
            all_obs_pts.append(negative_hazards[:, :2])

        if len(all_obs_pts) > 0:
            pts = np.vstack(all_obs_pts)
            x_ego = pts[:, 0]
            y_ego = pts[:, 1]
            d_ego = np.hypot(x_ego, y_ego)
            valid_mask = (d_ego >= 0.15) & (d_ego <= self.max_range)

            if np.any(valid_mask):
                d_val = d_ego[valid_mask]
                ye_val = y_ego[valid_mask]
                phi_ego = np.arctan2(ye_val, x_ego[valid_mask])
                bin_indices = (
                    np.floor((phi_ego + math.pi) / ANGULAR_RESOLUTION).astype(int)
                    % self.num_rays
                )

                for b_idx, d in zip(bin_indices, d_val):
                    if d < distances[b_idx]:
                        distances[b_idx] = d
                        classes[b_idx] = float(CLASS_STATIC_HAZARD)

        norm_distances = np.clip(distances / self.max_range, 0.0, 1.0)
        return norm_distances, classes

    def build_observation(
        self,
        kinematics: EgoKinematics,
        goal_state: GoalState,
        obstacle_points: np.ndarray | None = None,
        negative_hazards: np.ndarray | None = None,
    ) -> np.ndarray:
        """Assemble the complete 156-float observation vector for RLlib.

        Layout:
          [0:6]    - EgoKinematics (vx, vy, steering, yaw_rate, ax, ay)
          [6:12]   - GoalState (dx_ego, dy_ego, distance, heading_error, cross_track_error, path_heading_error)
          [12:84]  - Normalized radial distances (72 floats in [0, 1])
          [84:156] - Categorical obstacle classes (72 floats in {0, 3})
        """
        k_arr = kinematics.to_array()
        g_arr = goal_state.to_array()

        dist_arr, cls_arr = self.fuse_point_cloud_obstacles(
            obstacle_points=obstacle_points,
            negative_hazards=negative_hazards,
        )

        obs = np.concatenate([k_arr, g_arr, dist_arr, cls_arr]).astype(np.float32)
        assert len(obs) == 156, f"Observation size mismatch: expected 156, got {len(obs)}"
        return obs
