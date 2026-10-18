"""Ground segmentation and obstacle extraction engine for 3D directional LiDARs.

Differentiates traversable terrain (valid ground) from positive obstacles (trees,
curbs, small animals) and negative obstacles (ditches, holes, drop-offs) in the
vehicle's ego frame.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final

import numpy as np

from src.translate.config import (
    DITCH_NEGATIVE_THRESHOLD,
    GROUND_EPSILON_METERS,
    LIDAR_MAX_RANGE,
    MAX_OBSTACLE_HEIGHT,
    MAX_TRAVERSABLE_SLOPE_RAD,
    SMALL_OBSTACLE_MIN_HEIGHT,
)

DEFAULT_NUM_ZONES: Final[int] = 5
DEFAULT_NUM_SECTORS: Final[int] = 16


@dataclass(frozen=True)
class SegmentationResult:
    """Segmented point cloud components and classified terrain features in ego coordinates."""

    ground_points: np.ndarray  # (N_g, 3) traversable ground surface points
    obstacle_points: np.ndarray  # (N_o, 3) positive obstacles (curbs, walls, small animals)
    negative_hazards: np.ndarray  # (N_h, 3) ditches, holes, or steep drop-offs
    terrain_slope_rad: float  # Dominant surface slope in driving corridor
    roughness: float  # Elevation variance of ground surface


class GroundSegmentationEngine:
    """Concentric-zone sector plane fitter for outdoor 3D LiDAR point clouds."""

    def __init__(
        self,
        num_zones: int = DEFAULT_NUM_ZONES,
        num_sectors: int = DEFAULT_NUM_SECTORS,
        max_range: float = LIDAR_MAX_RANGE,
        ground_epsilon: float = GROUND_EPSILON_METERS,
        small_obstacle_min_h: float = SMALL_OBSTACLE_MIN_HEIGHT,
        max_obstacle_h: float = MAX_OBSTACLE_HEIGHT,
        ditch_threshold: float = DITCH_NEGATIVE_THRESHOLD,
        max_slope_rad: float = MAX_TRAVERSABLE_SLOPE_RAD,
    ) -> None:
        if num_zones < 1:
            raise ValueError(f"num_zones must be >= 1, got {num_zones}")
        if num_sectors < 1:
            raise ValueError(f"num_sectors must be >= 1, got {num_sectors}")
        if max_range <= 0.0:
            raise ValueError(f"max_range must be > 0.0, got {max_range}")

        self.num_zones = num_zones
        self.num_sectors = num_sectors
        self.max_range = max_range
        self.ground_epsilon = ground_epsilon
        self.small_obstacle_min_h = small_obstacle_min_h
        self.max_obstacle_h = max_obstacle_h
        self.ditch_threshold = ditch_threshold
        self.max_slope_rad = max_slope_rad

        # Define concentric radial zone boundaries (logarithmically or linearly spaced)
        self.zone_boundaries = np.linspace(0.2, self.max_range, self.num_zones + 1)
        # Azimuth sector boundaries covering [-pi, pi]
        self.sector_boundaries = np.linspace(-math.pi, math.pi, self.num_sectors + 1)

    def segment(self, points_ego: np.ndarray) -> SegmentationResult:
        """Partition ego-frame 3D points into ground, obstacles, and negative drop-offs.

        Args:
            points_ego: (N, 3) array of [x, y, z] coordinates in vehicle base_link frame.

        Returns:
            SegmentationResult containing separated point arrays and terrain metrics.
        """
        if points_ego.size == 0 or points_ego.shape[0] == 0:
            empty = np.empty((0, 3), dtype=np.float32)
            return SegmentationResult(
                ground_points=empty,
                obstacle_points=empty,
                negative_hazards=empty,
                terrain_slope_rad=0.0,
                roughness=0.0,
            )

        if points_ego.ndim != 2 or points_ego.shape[1] < 3:
            raise ValueError(
                f"points_ego must have shape (N, 3), got {points_ego.shape}"
            )

        xyz = points_ego[:, :3].astype(np.float32)
        r = np.hypot(xyz[:, 0], xyz[:, 1])

        # Filter points within operational range
        range_mask = (r >= self.zone_boundaries[0]) & (r <= self.max_range)
        valid_xyz = xyz[range_mask]
        valid_r = r[range_mask]

        if valid_xyz.shape[0] == 0:
            empty = np.empty((0, 3), dtype=np.float32)
            return SegmentationResult(
                ground_points=empty,
                obstacle_points=empty,
                negative_hazards=empty,
                terrain_slope_rad=0.0,
                roughness=0.0,
            )

        azimuth = np.arctan2(valid_xyz[:, 1], valid_xyz[:, 0])

        zone_idx = np.clip(
            np.digitize(valid_r, self.zone_boundaries) - 1, 0, self.num_zones - 1
        )
        sector_idx = np.clip(
            np.digitize(azimuth, self.sector_boundaries) - 1, 0, self.num_sectors - 1
        )

        ground_mask = np.zeros(valid_xyz.shape[0], dtype=bool)
        obstacle_mask = np.zeros(valid_xyz.shape[0], dtype=bool)
        negative_mask = np.zeros(valid_xyz.shape[0], dtype=bool)

        slopes: list[float] = []

        # Process radial rings per sector from inner zone outwards to track ground continuity
        for s_i in range(self.num_sectors):
            prev_ground_z: float | None = None

            for z_i in range(self.num_zones):
                patch_indices = np.where((zone_idx == z_i) & (sector_idx == s_i))[0]
                if patch_indices.size == 0:
                    continue

                patch_pts = valid_xyz[patch_indices]

                # Identify seed ground elevation as the lower 15th percentile to reject obstacles
                z_seed = float(np.percentile(patch_pts[:, 2], 15))

                # Check continuity with inner zone in this sector
                if prev_ground_z is not None:
                    step_z = z_seed - prev_ground_z
                    if step_z < self.ditch_threshold:
                        # Abrupt downward drop-off / ditch
                        negative_mask[patch_indices] = True
                        continue
                    if step_z > self.small_obstacle_min_h:
                        # Abrupt upward step / retaining wall / curb
                        obstacle_mask[patch_indices] = True
                        continue

                near_seed_mask = np.abs(patch_pts[:, 2] - z_seed) <= (self.ground_epsilon * 1.5)
                seed_pts = patch_pts[near_seed_mask]

                if seed_pts.shape[0] >= 3:
                    # Fit local surface plane: z = ax + by + c
                    centroid = np.mean(seed_pts, axis=0)
                    centered = seed_pts - centroid
                    try:
                        _, _, vh = np.linalg.svd(centered)
                        normal = vh[2]
                        if normal[2] < 0.0:
                            normal = -normal

                        normal_norm = float(np.linalg.norm(normal))
                        if normal_norm > 1e-6:
                            normal = normal / normal_norm
                        else:
                            normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)

                        # Slope angle relative to horizontal ground
                        slope = float(math.acos(np.clip(normal[2], -1.0, 1.0)))
                        slopes.append(slope)

                        if slope > self.max_slope_rad:
                            # Slope is too steep for vehicle traversal (e.g. embankment/ditch wall)
                            obstacle_mask[patch_indices] = True
                            continue

                        # Perpendicular signed distance to the plane for all points in this patch
                        dists_to_plane = np.dot(patch_pts - centroid, normal)

                        is_g = np.abs(dists_to_plane) <= self.ground_epsilon
                        is_obs = (dists_to_plane > self.small_obstacle_min_h) & (
                            dists_to_plane <= self.max_obstacle_h
                        )
                        is_neg = dists_to_plane < self.ditch_threshold

                        ground_mask[patch_indices[is_g]] = True
                        obstacle_mask[patch_indices[is_obs]] = True
                        negative_mask[patch_indices[is_neg]] = True
                        prev_ground_z = float(centroid[2])

                    except np.linalg.LinAlgError:
                        delta_z = patch_pts[:, 2] - z_seed
                        is_g = np.abs(delta_z) <= self.ground_epsilon
                        is_obs = (delta_z > self.small_obstacle_min_h) & (
                            delta_z <= self.max_obstacle_h
                        )
                        is_neg = delta_z < self.ditch_threshold
                        ground_mask[patch_indices[is_g]] = True
                        obstacle_mask[patch_indices[is_obs]] = True
                        negative_mask[patch_indices[is_neg]] = True
                        prev_ground_z = z_seed
                else:
                    delta_z = patch_pts[:, 2] - z_seed
                    is_g = np.abs(delta_z) <= self.ground_epsilon
                    is_obs = (delta_z > self.small_obstacle_min_h) & (
                        delta_z <= self.max_obstacle_h
                    )
                    is_neg = delta_z < self.ditch_threshold

                    ground_mask[patch_indices[is_g]] = True
                    obstacle_mask[patch_indices[is_obs]] = True
                    negative_mask[patch_indices[is_neg]] = True
                    prev_ground_z = z_seed

        ground_pts = valid_xyz[ground_mask]
        obstacle_pts = valid_xyz[obstacle_mask]
        negative_pts = valid_xyz[negative_mask]

        avg_slope = float(np.mean(slopes)) if len(slopes) > 0 else 0.0
        roughness = (
            float(np.var(ground_pts[:, 2])) if ground_pts.shape[0] > 1 else 0.0
        )

        return SegmentationResult(
            ground_points=ground_pts,
            obstacle_points=obstacle_pts,
            negative_hazards=negative_pts,
            terrain_slope_rad=avg_slope,
            roughness=roughness,
        )
