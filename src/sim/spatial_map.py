"""Spatial map builder for converting 3D obstacle tracks and walls into RL policy observations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from src.translate.config import (
    CAR_WIDTH,
    FRONT_LIDAR_X,
    LIDAR_FOV_RAD,
    LIDAR_MAX_RANGE,
    LIDAR_NUM_RAYS,
    REAR_LIDAR_X,
    VEHICLE_SIZE_SCALE,
    WHEELBASE,
)
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


# Side pods sit just outside the scaled body: x ahead of center, y clear of half-width.
SIDE_POD_X: Final[float] = WHEELBASE / 4.0
SIDE_POD_Y: Final[float] = CAR_WIDTH / 2.0 + 0.10 * VEHICLE_SIZE_SCALE

POD_CONFIGS: Final[dict[str, tuple[float, float, float]]] = {
    "front": (FRONT_LIDAR_X, 0.0, 0.0),
    "rear": (REAR_LIDAR_X, 0.0, math.pi),
    "right": (SIDE_POD_X, -SIDE_POD_Y, -math.pi / 2.0),
    "left": (SIDE_POD_X, SIDE_POD_Y, math.pi / 2.0),
}

POD_SPECS: Final[dict[str, dict[str, float | int]]] = {
    "front": {
        "x": FRONT_LIDAR_X,
        "y": 0.0,
        "yaw": 0.0,
        "samples": 180,
        "angle_min": -1.047198,
        "angle_max": 1.047198,
    },
    "rear": {
        "x": REAR_LIDAR_X,
        "y": 0.0,
        "yaw": math.pi,
        "samples": 180,
        "angle_min": -1.047198,
        "angle_max": 1.047198,
    },
    "right": {
        "x": SIDE_POD_X,
        "y": -SIDE_POD_Y,
        "yaw": -math.pi / 2.0,
        "samples": 150,
        "angle_min": -0.872665,
        "angle_max": 0.872665,
    },
    "left": {
        "x": SIDE_POD_X,
        "y": SIDE_POD_Y,
        "yaw": math.pi / 2.0,
        "samples": 150,
        "angle_min": -0.872665,
        "angle_max": 0.872665,
    },
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
        self.latest_mock_pod_scans: dict[str, LaserScanData] = {}

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

    def fuse_point_cloud_obstacles(
        self,
        obstacle_points: np.ndarray | None,
        negative_hazards: np.ndarray | None = None,
        obstacles: list[TrackedObstacle] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project segmented 3D non-ground obstacle points into 360-degree polar range and classification grid.

        Args:
            obstacle_points: (N, 3) or (N, 2) Cartesian hit coordinates in vehicle ego frame.
            negative_hazards: Optional (M, 3) or (M, 2) drop-off / ditch coordinates in ego frame.
            obstacles: Optional list of tracked obstacles for semantic classification.

        Returns:
            distances: (72,) normalized float array in [0.0, 1.0].
            classes: (72,) float array of spatial class codes.
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
                xe_val = x_ego[valid_mask]
                ye_val = y_ego[valid_mask]
                phi_ego = np.arctan2(ye_val, xe_val)
                bin_indices = (
                    np.floor((phi_ego + math.pi) / ANGULAR_RESOLUTION).astype(int)
                    % self.num_rays
                )

                for b_idx, d, xe, ye in zip(bin_indices, d_val, xe_val, ye_val):
                    if d < distances[b_idx]:
                        distances[b_idx] = d
                        hit_cls = float(CLASS_STATIC_HAZARD)
                        if obstacles:
                            for obs in obstacles:
                                if math.hypot(xe - obs.rel_x, ye - obs.rel_y) <= (
                                    obs.radius + 0.35
                                ):
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

    def extract_pod_beam_segments(
        self,
        pod_scans: dict[str, LaserScanData],
        obstacles: list[TrackedObstacle] | None = None,
        stride: int = 1,
    ) -> list[dict[str, Any]]:
        """Extract physical LiDAR rays originating from each physical pod's mounting position.

        Returns a list of dictionaries per pod:
            {
                "pod_name": str,
                "origin_ego": (float, float),
                "endpoints_ego": np.ndarray (N, 2),
                "ranges": np.ndarray (N,),
                "classes": np.ndarray (N,),
                "is_hit": np.ndarray (N,),
            }
        """
        pod_results: list[dict[str, Any]] = []
        for pod_name, scan in pod_scans.items():
            if pod_name not in POD_CONFIGS or len(scan.ranges) == 0:
                continue
            x_pod, y_pod, yaw_pod = POD_CONFIGS[pod_name]
            raw_ranges = np.asarray(scan.ranges, dtype=np.float32)
            n_samples = len(raw_ranges)
            angles = scan.angle_min + np.arange(n_samples, dtype=np.float32) * scan.angle_increment

            if stride > 1:
                raw_ranges = raw_ranges[::stride]
                angles = angles[::stride]

            beam_endpoints: list[tuple[float, float]] = []
            beam_classes: list[int] = []
            beam_hits: list[bool] = []
            beam_eff_r: list[float] = []

            for r, ang in zip(raw_ranges, angles):
                total_angle = yaw_pod + float(ang)
                if np.isfinite(r) and scan.range_min <= r <= self.max_range:
                    eff_r = float(r)
                    is_hit = True
                    xe = x_pod + eff_r * math.cos(total_angle)
                    ye = y_pod + eff_r * math.sin(total_angle)
                    hit_cls = CLASS_STATIC_HAZARD
                    if obstacles:
                        for obs in obstacles:
                            if math.hypot(xe - obs.rel_x, ye - obs.rel_y) <= (obs.radius + 0.35):
                                hit_cls = obs.to_spatial_code()
                                break
                else:
                    eff_r = float(self.max_range)
                    is_hit = False
                    xe = x_pod + eff_r * math.cos(total_angle)
                    ye = y_pod + eff_r * math.sin(total_angle)
                    hit_cls = CLASS_FREE

                beam_endpoints.append((xe, ye))
                beam_classes.append(hit_cls)
                beam_hits.append(is_hit)
                beam_eff_r.append(eff_r)

            pod_results.append({
                "pod_name": pod_name,
                "origin_ego": (x_pod, y_pod),
                "endpoints_ego": np.array(beam_endpoints, dtype=np.float32),
                "ranges": np.array(beam_eff_r, dtype=np.float32),
                "classes": np.array(beam_classes, dtype=np.int32),
                "is_hit": np.array(beam_hits, dtype=bool),
            })
        return pod_results

    def generate_mock_pod_scans(
        self,
        wall_segments_ego: list[tuple[tuple[float, float], tuple[float, float]]] | None = None,
        obstacles: list[TrackedObstacle] | None = None,
    ) -> dict[str, LaserScanData]:
        """Synthesize 4-pod LaserScanData analytically from physical pod locations."""
        scans: dict[str, LaserScanData] = {}
        for pod_name, spec in POD_SPECS.items():
            n_samples = int(spec["samples"])
            ang_min = float(spec["angle_min"])
            ang_max = float(spec["angle_max"])
            ang_inc = (ang_max - ang_min) / max(1, n_samples - 1)
            x_pod = float(spec["x"])
            y_pod = float(spec["y"])
            yaw_pod = float(spec["yaw"])

            ranges = np.full(n_samples, self.max_range, dtype=np.float32)

            for i in range(n_samples):
                beam_ang = yaw_pod + ang_min + i * ang_inc
                dx = math.cos(beam_ang)
                dy = math.sin(beam_ang)

                # Cast against walls
                if wall_segments_ego:
                    for (wx1, wy1), (wx2, wy2) in wall_segments_ego:
                        sx = wx2 - wx1
                        sy = wy2 - wy1
                        det = sx * dy - sy * dx
                        if abs(det) < 1e-6:
                            continue
                        del_x = wx1 - x_pod
                        del_y = wy1 - y_pod
                        t = (sx * del_y - sy * del_x) / det
                        u = (dx * del_y - dy * del_x) / det
                        if t > 0.0 and 0.0 <= u <= 1.0 and t < ranges[i]:
                            ranges[i] = t

                # Cast against obstacles
                if obstacles:
                    for obs in obstacles:
                        del_x = obs.rel_x - x_pod
                        del_y = obs.rel_y - y_pod
                        proj = del_x * dx + del_y * dy
                        if proj <= 0.0:
                            continue
                        h_sq = (del_x * del_x + del_y * del_y) - proj * proj
                        r_sq = obs.radius * obs.radius
                        if h_sq <= r_sq:
                            t = proj - math.sqrt(max(0.0, r_sq - h_sq))
                            if 0.0 < t < ranges[i]:
                                ranges[i] = t

            scans[pod_name] = LaserScanData(
                ranges=ranges,
                angle_min=ang_min,
                angle_max=ang_max,
                angle_increment=ang_inc,
                range_min=0.15,
                range_max=self.max_range,
            )
        return scans

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
        obstacle_points: np.ndarray | None = None,
        negative_hazards: np.ndarray | None = None,
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

        if obstacle_points is not None and obstacle_points.size > 0:
            dist_arr, cls_arr = self.fuse_point_cloud_obstacles(
                obstacle_points=obstacle_points,
                negative_hazards=negative_hazards,
                obstacles=obstacles,
            )
        elif pod_scans is not None and len(pod_scans) > 0:
            dist_arr, cls_arr = self.fuse_pod_scans(pod_scans, obstacles=obstacles)
        elif wall_segments_ego is not None:
            mock_scans = self.generate_mock_pod_scans(
                wall_segments_ego=wall_segments_ego,
                obstacles=obstacles,
            )
            self.latest_mock_pod_scans = mock_scans
            dist_arr, cls_arr = self.fuse_pod_scans(mock_scans, obstacles=obstacles)
        else:
            dist_arr, cls_arr = self.compute_ray_scan(
                obstacles if obstacles is not None else [],
                wall_segments_ego=wall_segments_ego,
            )

        obs = np.concatenate([k_arr, g_arr, dist_arr, cls_arr]).astype(np.float32)
        assert len(obs) == 156, f"Observation size mismatch: expected 156, got {len(obs)}"
        return obs


