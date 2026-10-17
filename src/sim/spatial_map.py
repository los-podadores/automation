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


POD_CONFIGS: Final[dict[str, tuple[float, float, float]]] = {
    "front": (1.05, 0.0, 0.0),
    "rear": (-1.05, 0.0, math.pi),
    "right": (0.50, -0.70, -math.pi / 2.0),
    "left": (0.50, 0.70, math.pi / 2.0),
}


@dataclass(frozen=True)
class LaserScanData:
    """Normalized representation of a 2D planar LiDAR scan."""

    ranges: np.ndarray
    angle_min: float
    angle_max: float
    angle_increment: float
    range_min: float = 0.15
    range_max: float = 25.0


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

    def fuse_pod_scans(
        self,
        pod_scans: dict[str, LaserScanData],
        obstacles: list[TrackedObstacle] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fuse 4 distributed LaserScans into a 360-degree polar range and classification grid.

        Args:
            pod_scans: Mapping of pod name ('front', 'rear', 'right', 'left') to LaserScanData.
            obstacles: Optional list of tracked obstacles in vehicle ego frame for semantic tagging.

        Returns:
            distances: (72,) normalized float array in [0.0, 1.0].
            classes: (72,) float array of spatial class codes.
        """
        distances = np.full(self.num_rays, self.max_range, dtype=np.float32)
        classes = np.full(self.num_rays, CLASS_FREE, dtype=np.float32)

        for pod_name, scan in pod_scans.items():
            if pod_name not in POD_CONFIGS or len(scan.ranges) == 0:
                continue

            x_pod, y_pod, yaw_pod = POD_CONFIGS[pod_name]
            raw_ranges = np.asarray(scan.ranges, dtype=np.float32)
            n_samples = len(raw_ranges)

            angles = scan.angle_min + np.arange(n_samples, dtype=np.float32) * scan.angle_increment
            valid_mask = (
                np.isfinite(raw_ranges)
                & (raw_ranges >= scan.range_min)
                & (raw_ranges <= scan.range_max)
                & (raw_ranges <= self.max_range)
            )

            if not np.any(valid_mask):
                continue

            valid_r = raw_ranges[valid_mask]
            valid_ang = angles[valid_mask] + yaw_pod

            # Transform sensor-local ray hit into vehicle ego frame
            x_ego = x_pod + valid_r * np.cos(valid_ang)
            y_ego = y_pod + valid_r * np.sin(valid_ang)

            d_ego = np.hypot(x_ego, y_ego)
            phi_ego = np.arctan2(y_ego, x_ego)

            # Map azimuth to [0, 71] uniform angular bin
            bin_indices = np.floor((phi_ego + math.pi) / ANGULAR_RESOLUTION).astype(int) % self.num_rays

            for b_idx, d, xe, ye in zip(bin_indices, d_ego, x_ego, y_ego):
                if d < distances[b_idx]:
                    distances[b_idx] = d
                    hit_cls = float(CLASS_STATIC_HAZARD)
                    if obstacles:
                        for obs in obstacles:
                            if math.hypot(xe - obs.rel_x, ye - obs.rel_y) <= (obs.radius + 0.35):
                                hit_cls = float(obs.to_spatial_code())
                                break
                    classes[b_idx] = hit_cls

        norm_distances = np.clip(distances / self.max_range, 0.0, 1.0)
        return norm_distances, classes

    def extract_point_cloud_ego(
        self, pod_scans: dict[str, LaserScanData]
    ) -> np.ndarray:
        """Extract all valid 2D Cartesian hit points (x_ego, y_ego) from 4-pod LiDARs."""
        all_pts: list[np.ndarray] = []
        for pod_name, scan in pod_scans.items():
            if pod_name not in POD_CONFIGS or len(scan.ranges) == 0:
                continue
            x_pod, y_pod, yaw_pod = POD_CONFIGS[pod_name]
            raw_ranges = np.asarray(scan.ranges, dtype=np.float32)
            valid_mask = (
                np.isfinite(raw_ranges)
                & (raw_ranges >= scan.range_min)
                & (raw_ranges <= scan.range_max)
                & (raw_ranges <= self.max_range)
            )
            if not np.any(valid_mask):
                continue
            valid_r = raw_ranges[valid_mask]
            n_samples = len(raw_ranges)
            angles = scan.angle_min + np.arange(n_samples, dtype=np.float32) * scan.angle_increment
            valid_ang = angles[valid_mask] + yaw_pod
            x_ego = x_pod + valid_r * np.cos(valid_ang)
            y_ego = y_pod + valid_r * np.sin(valid_ang)
            all_pts.append(np.column_stack([x_ego, y_ego]))
        if all_pts:
            return np.vstack(all_pts).astype(np.float32)
        return np.empty((0, 2), dtype=np.float32)

    def compute_ray_scan(
        self,
        obstacles: list[TrackedObstacle],
        wall_segments_ego: list[tuple[tuple[float, float], tuple[float, float]]] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Cast 72 radial rays across the full 360-degree surround perimeter (analytical fallback).

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
        obstacles: list[TrackedObstacle] | None = None,
        wall_segments_ego: list[tuple[tuple[float, float], tuple[float, float]]] | None = None,
        pod_scans: dict[str, LaserScanData] | None = None,
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

        if pod_scans is not None and len(pod_scans) > 0:
            dist_arr, cls_arr = self.fuse_pod_scans(pod_scans, obstacles=obstacles)
        else:
            dist_arr, cls_arr = self.compute_ray_scan(
                obstacles if obstacles is not None else [],
                wall_segments_ego=wall_segments_ego,
            )

        obs = np.concatenate([k_arr, g_arr, dist_arr, cls_arr]).astype(np.float32)
        assert len(obs) == 156, f"Observation size mismatch: expected 156, got {len(obs)}"
        return obs


