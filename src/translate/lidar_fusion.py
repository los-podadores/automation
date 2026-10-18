"""Camera-LiDAR sensor fusion engine for Option B (Forward-biased 3-Pod Array).

Associates 2D YOLO detections with 3D LiDAR point clouds to produce
calibrated metric 3D obstacle tracks in the vehicle's ego-coordinate frame.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final

import numpy as np

from .config import (
    CAR_WIDTH,
    FRONT_LIDAR_X,
    FRONT_LIDAR_Z,
    REAR_LIDAR_X,
    REAR_LIDAR_Z,
    VEHICLE_SIZE_SCALE,
    WHEELBASE,
)
from .fusion_types import Detection, PodExtrinsics, TrackedObstacle

_SIDE_POD_X: Final[float] = WHEELBASE / 4.0
_SIDE_POD_Y: Final[float] = CAR_WIDTH / 2.0 + 0.10 * VEHICLE_SIZE_SCALE
_SIDE_POD_Z: Final[float] = 0.70 * VEHICLE_SIZE_SCALE

DEFAULT_PODS: Final[tuple[PodExtrinsics, ...]] = (
    PodExtrinsics(
        pod_id=0,
        name="front_center",
        yaw_deg=0.0,
        x_offset=FRONT_LIDAR_X,
        y_offset=0.0,
        z_offset=FRONT_LIDAR_Z,
        fx=600.0,
        fy=600.0,
        cx=320.0,
        cy=240.0,
        lidar_to_cam_tx=0.0,
        lidar_to_cam_ty=-0.05,
        lidar_to_cam_tz=-0.05,
    ),
    PodExtrinsics(
        pod_id=1,
        name="rear_center",
        yaw_deg=180.0,
        x_offset=REAR_LIDAR_X,
        y_offset=0.0,
        z_offset=REAR_LIDAR_Z,
        fx=600.0,
        fy=600.0,
        cx=320.0,
        cy=240.0,
        lidar_to_cam_tx=0.0,
        lidar_to_cam_ty=-0.05,
        lidar_to_cam_tz=-0.05,
    ),
    PodExtrinsics(
        pod_id=2,
        name="front_right",
        yaw_deg=-90.0,
        x_offset=_SIDE_POD_X,
        y_offset=-_SIDE_POD_Y,
        z_offset=_SIDE_POD_Z,
        fx=600.0,
        fy=600.0,
        cx=320.0,
        cy=240.0,
        lidar_to_cam_tx=0.0,
        lidar_to_cam_ty=-0.05,
        lidar_to_cam_tz=-0.05,
    ),
    PodExtrinsics(
        pod_id=3,
        name="front_left",
        yaw_deg=90.0,
        x_offset=_SIDE_POD_X,
        y_offset=_SIDE_POD_Y,
        z_offset=_SIDE_POD_Z,
        fx=600.0,
        fy=600.0,
        cx=320.0,
        cy=240.0,
        lidar_to_cam_tx=0.0,
        lidar_to_cam_ty=-0.05,
        lidar_to_cam_tz=-0.05,
    ),
)



class LidarCameraFusionEngine:
    """Projects 3D LiDAR point clouds into 2D camera images and associates depths."""

    def __init__(
        self,
        detector: ObstacleDetector | None = None,
        pods: tuple[PodExtrinsics, ...] = DEFAULT_PODS,
    ) -> None:
        self.detector = detector
        self.pods = {pod.pod_id: pod for pod in pods}

    def project_points_to_pixels(
        self,
        points: np.ndarray,
        pod: PodExtrinsics,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project (N, 3) 3D points in the pod's local LiDAR frame into 2D camera pixels.

        Args:
            points: (N, 3) array of [X_fwd, Y_left, Z_up] points relative to LiDAR origin.
            pod: Calibration extrinsics for the specific pod.

        Returns:
            uv: (M, 2) array of pixel coordinates [u, v] for points in front of camera.
            valid_mask: (N,) boolean mask indicating which input points projected in front.
        """
        if points.shape[0] == 0:
            return np.empty((0, 2), dtype=np.float32), np.zeros(0, dtype=bool)

        # Transform from LiDAR frame to Camera frame:
        # LiDAR frame: X=forward, Y=left, Z=up
        # Camera optical frame: X_c=right (-Y), Y_c=down (-Z), Z_c=forward (+X)
        pts_cam_x = -(points[:, 1] + pod.lidar_to_cam_ty)
        pts_cam_y = -(points[:, 2] + pod.lidar_to_cam_tz)
        pts_cam_z = points[:, 0] + pod.lidar_to_cam_tx

        # Only consider points in front of the lens (positive optical depth)
        valid_mask = pts_cam_z > 0.1
        if not np.any(valid_mask):
            return np.empty((0, 2), dtype=np.float32), valid_mask

        zc = pts_cam_z[valid_mask]
        u = (pts_cam_x[valid_mask] * pod.fx / zc) + pod.cx
        v = (pts_cam_y[valid_mask] * pod.fy / zc) + pod.cy

        uv = np.stack([u, v], axis=1)
        return uv, valid_mask

    def associate_detections_with_points(
        self,
        detections: list[Detection],
        points: np.ndarray,
        pod: PodExtrinsics,
    ) -> list[TrackedObstacle]:
        """Associate 2D detections with 3D points and transform into vehicle base_link frame.

        Args:
            detections: List of 2D bounding boxes from YOLOv8s.
            points: (N, 3) raw point cloud points in local pod coordinates.
            pod: Extrinsics of the camera-lidar pod.

        Returns:
            List of TrackedObstacle instances in vehicle ego coordinates.
        """
        if not detections or points.shape[0] == 0:
            return []

        uv, valid_mask = self.project_points_to_pixels(points, pod)
        if uv.shape[0] == 0:
            return []

        valid_points = points[valid_mask]
        tracked: list[TrackedObstacle] = []

        yaw_rad = math.radians(pod.yaw_deg)
        cos_yaw = math.cos(yaw_rad)
        sin_yaw = math.sin(yaw_rad)

        for det in detections:
            x1, y1, x2, y2 = det.bbox

            # Filter points landing inside the bounding box
            in_box = (
                (uv[:, 0] >= x1)
                & (uv[:, 0] <= x2)
                & (uv[:, 1] >= y1)
                & (uv[:, 1] <= y2)
            )

            box_points = valid_points[in_box]
            if len(box_points) < 3:
                # Insufficient points for robust depth estimation
                continue

            # Compute median 3D position in pod frame to suppress background/ground outliers
            med_x = float(np.median(box_points[:, 0]))
            med_y = float(np.median(box_points[:, 1]))

            # Estimated spatial obstacle radius based on point cloud spread
            spread_y = float(np.percentile(box_points[:, 1], 80) - np.percentile(box_points[:, 1], 20))
            radius = max(0.2, spread_y / 2.0)

            # Transform from pod frame to vehicle base_link frame:
            # 1. Rotate by pod mounting yaw
            rel_x = pod.x_offset + (med_x * cos_yaw - med_y * sin_yaw)
            rel_y = pod.y_offset + (med_x * sin_yaw + med_y * cos_yaw)

            tracked.append(
                TrackedObstacle(
                    class_id=det.class_id,
                    class_name=det.class_name,
                    rel_x=rel_x,
                    rel_y=rel_y,
                    rel_vx=0.0,
                    rel_vy=0.0,
                    radius=radius,
                    confidence=det.confidence,
                )
            )

        return tracked

    def merge_and_deduplicate(
        self,
        obstacles: list[TrackedObstacle],
        distance_threshold: float = 0.8,
    ) -> list[TrackedObstacle]:
        """Merge overlapping obstacle detections from adjacent pods."""
        if len(obstacles) <= 1:
            return obstacles

        # Sort by confidence descending
        sorted_obs = sorted(obstacles, key=lambda o: o.confidence, reverse=True)
        merged: list[TrackedObstacle] = []

        for obs in sorted_obs:
            is_duplicate = False
            for existing in merged:
                dist = math.hypot(obs.rel_x - existing.rel_x, obs.rel_y - existing.rel_y)
                if dist < distance_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                merged.append(obs)

        return merged
