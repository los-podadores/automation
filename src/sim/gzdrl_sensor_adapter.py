"""Adapter for zero-copy GzDRL GPU LiDAR buffers to 3D Cartesian point clouds.

Projects raw spherical range buffers (V, H, 3) from Gazebo Harmonic render memory
into calibrated 3D Cartesian points (N, 3) in the vehicle chassis coordinate frame,
and feeds them to GroundSegmentationEngine and SpatialMapBuilder.
"""

from __future__ import annotations

import math
from typing import Any, Final

import numpy as np

from src.translate.config import LIDAR_MAX_RANGE, VEHICLE_SIZE_SCALE
from .ground_segmentation import GroundSegmentationEngine, SegmentationResult
from .spatial_map import EgoKinematics, GoalState, SpatialMapBuilder

# Standard mounting parameters (scaled with vehicle size)
PITCH_DOWN_RAD: Final[float] = 0.349066  # ~20 degrees down
FRONT_X: Final[float] = 1.45 * VEHICLE_SIZE_SCALE  # 2.175 m
FRONT_Z: Final[float] = 0.60 * VEHICLE_SIZE_SCALE  # 0.900 m
REAR_X: Final[float] = -1.45 * VEHICLE_SIZE_SCALE  # -2.175 m
REAR_Z: Final[float] = 0.60 * VEHICLE_SIZE_SCALE   # 0.900 m

HORIZ_MIN_RAD: Final[float] = -1.047198  # -60 deg
HORIZ_MAX_RAD: Final[float] = 1.047198   # +60 deg
VERT_MIN_RAD: Final[float] = -0.436332   # -25 deg
VERT_MAX_RAD: Final[float] = 0.261799    # +15 deg


class GzDrlSensorAdapter:
    """Vectorized projection adapter for GPU LiDAR range buffers in GzDRL."""

    def __init__(
        self,
        front_sensor_name: str = "tractor::chassis::lidar_front",
        rear_sensor_name: str = "tractor::chassis::lidar_rear",
        min_range: float = 0.15,
        max_range: float = LIDAR_MAX_RANGE,
        pitch_down_rad: float = PITCH_DOWN_RAD,
        front_x: float = FRONT_X,
        front_z: float = FRONT_Z,
        rear_x: float = REAR_X,
        rear_z: float = REAR_Z,
    ) -> None:
        self.front_name = front_sensor_name
        self.rear_name = rear_sensor_name
        self.min_range = min_range
        self.max_range = max_range
        self.pitch_down = pitch_down_rad
        self.front_x = front_x
        self.front_z = front_z
        self.rear_x = rear_x
        self.rear_z = rear_z

        # Cached ray directions for (V, H) grid
        self._cached_shape: tuple[int, int] | None = None
        self._front_dir_x: np.ndarray | None = None
        self._front_dir_y: np.ndarray | None = None
        self._front_dir_z: np.ndarray | None = None
        self._rear_dir_x: np.ndarray | None = None
        self._rear_dir_y: np.ndarray | None = None
        self._rear_dir_z: np.ndarray | None = None

    def _ensure_grid(self, v_samples: int, h_samples: int) -> None:
        """Precompute unit ray direction vectors for given angular grid resolution."""
        if self._cached_shape == (v_samples, h_samples):
            return

        theta = np.linspace(HORIZ_MIN_RAD, HORIZ_MAX_RAD, h_samples, endpoint=True, dtype=np.float32)
        phi = np.linspace(VERT_MIN_RAD, VERT_MAX_RAD, v_samples, endpoint=True, dtype=np.float32)
        th_grid, ph_grid = np.meshgrid(theta, phi)

        cos_ph = np.cos(ph_grid)
        sin_ph = np.sin(ph_grid)
        cos_th = np.cos(th_grid)
        sin_th = np.sin(th_grid)

        # In sensor frame: +x forward, +y left, +z up
        xs = cos_ph * cos_th
        ys = cos_ph * sin_th
        zs = sin_ph

        # Front mount: rotation Ry(pitch_down)
        # x_c =  xs * cos(p) + zs * sin(p)
        # y_c =  ys
        # z_c = -xs * sin(p) + zs * cos(p)
        p = self.pitch_down
        cos_p = math.cos(p)
        sin_p = math.sin(p)

        self._front_dir_x = (xs * cos_p + zs * sin_p).astype(np.float32)
        self._front_dir_y = ys.astype(np.float32)
        self._front_dir_z = (-xs * sin_p + zs * cos_p).astype(np.float32)

        # Rear mount: Rz(pi) * Ry(pitch_down)
        # x_c = -xs * cos(p) - zs * sin(p)
        # y_c = -ys
        # z_c = -xs * sin(p) + zs * cos(p)
        self._rear_dir_x = (-xs * cos_p - zs * sin_p).astype(np.float32)
        self._rear_dir_y = (-ys).astype(np.float32)
        self._rear_dir_z = (-xs * sin_p + zs * cos_p).astype(np.float32)

        self._cached_shape = (v_samples, h_samples)

    def extract_point_cloud(self, server: Any) -> np.ndarray:
        """Extract front and rear GPU LiDAR scans and project to ego chassis (N, 3) point cloud."""
        front_scan = server.get_sensor_gpu_lidar(self.front_name)
        rear_scan = server.get_sensor_gpu_lidar(self.rear_name)

        if front_scan.size == 0 or rear_scan.size == 0:
            return np.empty((0, 3), dtype=np.float32)

        v_samples, h_samples = front_scan.shape[:2]
        self._ensure_grid(v_samples, h_samples)

        rf = front_scan[:, :, 0]
        rr = rear_scan[:, :, 0]

        pts_list: list[np.ndarray] = []

        # Process front scan
        mask_f = np.isfinite(rf) & (rf >= self.min_range) & (rf <= self.max_range)
        if np.any(mask_f):
            rf_val = rf[mask_f]
            xf = rf_val * self._front_dir_x[mask_f] + self.front_x
            yf = rf_val * self._front_dir_y[mask_f]
            zf = rf_val * self._front_dir_z[mask_f] + self.front_z
            pts_list.append(np.column_stack([xf, yf, zf]))

        # Process rear scan
        mask_r = np.isfinite(rr) & (rr >= self.min_range) & (rr <= self.max_range)
        if np.any(mask_r):
            rr_val = rr[mask_r]
            xr = rr_val * self._rear_dir_x[mask_r] + self.rear_x
            yr = rr_val * self._rear_dir_y[mask_r]
            zr = rr_val * self._rear_dir_z[mask_r] + self.rear_z
            pts_list.append(np.column_stack([xr, yr, zr]))

        if not pts_list:
            return np.empty((0, 3), dtype=np.float32)

        return np.vstack(pts_list).astype(np.float32)

    def process_and_observe(
        self,
        server: Any,
        seg_engine: GroundSegmentationEngine,
        map_builder: SpatialMapBuilder,
        kinematics: EgoKinematics,
        goal_state: GoalState,
    ) -> tuple[np.ndarray, SegmentationResult]:
        """End-to-end pipeline: GPU LiDAR buffers -> (N,3) points -> ground segmentation -> 156-obs vector."""
        points_ego = self.extract_point_cloud(server)
        seg_res = seg_engine.segment(points_ego)
        obs = map_builder.build_observation(
            kinematics=kinematics,
            goal_state=goal_state,
            obstacle_points=seg_res.obstacle_points,
            negative_hazards=seg_res.negative_hazards,
        )
        return obs, seg_res
