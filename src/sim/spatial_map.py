"""Spatial map builder for converting 3D obstacle tracks into RL policy observations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np

from src.translate.fusion_types import CLASS_FREE, TrackedObstacle

NUM_RAYS: Final[int] = 72
MAX_RANGE_METERS: Final[float] = 25.0
RAY_FOV_RAD: Final[float] = math.pi  # 180 degrees (-pi/2 to +pi/2)
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


@dataclass(frozen=True)
class PathDeviation:
    """Cross-track error, heading error, and road geometry."""

    lateral_error: float  # Distance from road/lane center in meters (+left / -right)
    heading_error: float  # Angular deviation from lane direction (rad)
    curvature_now: float  # Road curvature 1/R at current position (1/m)
    curvature_lookahead: float  # Road curvature 1/R 10 meters ahead (1/m)

    def to_array(self) -> np.ndarray:
        return np.array(
            [
                self.lateral_error,
                self.heading_error,
                self.curvature_now,
                self.curvature_lookahead,
            ],
            dtype=np.float32,
        )


class SpatialMapBuilder:
    """Encodes tracked obstacles and ego state into a normalized observation vector."""

    def __init__(
        self,
        num_rays: int = NUM_RAYS,
        max_range: float = MAX_RANGE_METERS,
    ) -> None:
        self.num_rays = num_rays
        self.max_range = max_range
        # Angles for each ray center, from -pi/2 (-90 deg, right) to +pi/2 (+90 deg, left)
        self.ray_angles = np.linspace(-RAY_FOV_RAD / 2.0, RAY_FOV_RAD / 2.0, self.num_rays)

    def compute_ray_scan(
        self,
        obstacles: list[TrackedObstacle],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Cast 72 radial rays across the forward 180-degree field of view.

        Returns:
            distances: (72,) normalized float array in [0.0, 1.0], where 1.0 is unobstructed.
            classes: (72,) int/float array of spatial class codes (0=Free, 1=Vehicle, 2=Ped, 3=Static).
        """
        distances = np.full(self.num_rays, self.max_range, dtype=np.float32)
        classes = np.full(self.num_rays, CLASS_FREE, dtype=np.float32)

        for obs in obstacles:
            # Only consider obstacles in front of the vehicle
            if obs.rel_x <= 0.05:
                continue

            dist = obs.distance
            if dist > self.max_range:
                continue

            azimuth = obs.azimuth  # rad: -pi/2 (right) to +pi/2 (left)
            if abs(azimuth) > (RAY_FOV_RAD / 2.0 + 0.2):
                continue

            # Angular width of obstacle given its radius
            ang_radius = math.asin(min(1.0, obs.radius / max(dist, 0.1)))
            min_ang = azimuth - ang_radius
            max_ang = azimuth + ang_radius

            # Find rays intersected by this obstacle
            ray_indices = np.where((self.ray_angles >= min_ang) & (self.ray_angles <= max_ang))[0]
            if len(ray_indices) == 0:
                # Fallback to single closest ray if obstacle is very small or distant
                closest_idx = int(np.argmin(np.abs(self.ray_angles - azimuth)))
                ray_indices = np.array([closest_idx])

            obs_code = float(obs.to_spatial_code())
            boundary_dist = max(0.0, dist - obs.radius)

            for idx in ray_indices:
                if boundary_dist < distances[idx]:
                    distances[idx] = boundary_dist
                    classes[idx] = obs_code

        # Normalize distance to [0.0, 1.0]
        norm_distances = distances / self.max_range
        return norm_distances, classes

    def build_observation(
        self,
        kinematics: EgoKinematics,
        deviation: PathDeviation,
        obstacles: list[TrackedObstacle],
    ) -> np.ndarray:
        """Assemble the complete 154-float observation vector for RLlib.

        Layout:
          [0:6]   - EgoKinematics (vx, vy, steering, yaw_rate, ax, ay)
          [6:10]  - PathDeviation (lateral_error, heading_error, curvature_now, curvature_lookahead)
          [10:82] - Normalized radial distances (72 floats in [0, 1])
          [82:154]- Categorical obstacle classes (72 floats in {0, 1, 2, 3})
        """
        k_arr = kinematics.to_array()
        d_arr = deviation.to_array()
        dist_arr, cls_arr = self.compute_ray_scan(obstacles)

        obs = np.concatenate([k_arr, d_arr, dist_arr, cls_arr]).astype(np.float32)
        assert len(obs) == 154, f"Observation size mismatch: expected 154, got {len(obs)}"
        return obs
