"""Shared types and data structures for sensor fusion and spatial tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from .config import NAVIGABLE_CLASSES, TRANSIT_OBSTACLE_CLASSES

# Standardized class IDs for RL spatial map encoding
CLASS_FREE: Final[int] = 0
CLASS_VEHICLE: Final[int] = 1
CLASS_PEDESTRIAN: Final[int] = 2
CLASS_STATIC_HAZARD: Final[int] = 3


@dataclass
class Detection:
    """A single 2D obstacle detection from camera imagery."""

    class_name: str
    class_id: int
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 in pixels
    center: tuple[int, int]  # cx, cy in pixels
    area: int  # bounding box area in pixels

    @property
    def is_transit_obstacle(self) -> bool:
        target = self.class_name.lower()
        return any(target == c.lower() for c in TRANSIT_OBSTACLE_CLASSES)

    @property
    def is_navigable(self) -> bool:
        target = self.class_name.lower()
        return any(target == c.lower() for c in NAVIGABLE_CLASSES)


@dataclass(frozen=True)
class TrackedObstacle:
    """A metric 3D obstacle represented in the vehicle's ego-coordinate frame.

    Coordinate system (REP-103):
      +X: Forward (meters)
      +Y: Left (meters)
      +Z: Up (meters)
    """

    class_id: int
    class_name: str
    rel_x: float  # Longitudinal distance (meters ahead, +X)
    rel_y: float  # Lateral distance (meters left +Y / right -Y)
    rel_vx: float  # Relative longitudinal velocity (m/s)
    rel_vy: float  # Relative lateral velocity (m/s)
    radius: float  # Estimated radial bounding size (m)
    confidence: float  # Detection confidence in [0.0, 1.0]

    @property
    def distance(self) -> float:
        """Euclidean distance from ego origin in the ground plane."""
        return (self.rel_x**2 + self.rel_y**2) ** 0.5

    @property
    def azimuth(self) -> float:
        """Angle in radians relative to forward heading (-pi to +pi)."""
        import math

        return math.atan2(self.rel_y, self.rel_x)

    def to_spatial_code(self) -> int:
        """Map class name/id to categorical RL spatial code."""
        name = self.class_name.lower()
        if name in ("car", "bus", "truck", "motorcycle"):
            return CLASS_VEHICLE
        if name in ("person", "dog"):
            return CLASS_PEDESTRIAN
        return CLASS_STATIC_HAZARD


@dataclass(frozen=True)
class PodExtrinsics:
    """Extrinsic and intrinsic calibration parameters for a Camera-LiDAR pod."""

    pod_id: int
    name: str
    yaw_deg: float  # Heading offset in degrees (+45 for left, -45 for right, 0 for center)
    x_offset: float  # Offset from base_link in meters (+X forward)
    y_offset: float  # Offset from base_link in meters (+Y left)
    z_offset: float  # Offset from base_link in meters (+Z up)
    fx: float  # Focal length X in pixels
    fy: float  # Focal length Y in pixels
    cx: float  # Optical center X in pixels
    cy: float  # Optical center Y in pixels
    lidar_to_cam_tx: float  # Translation X from LiDAR to camera (meters)
    lidar_to_cam_ty: float  # Translation Y from LiDAR to camera (meters)
    lidar_to_cam_tz: float  # Translation Z from LiDAR to camera (meters)
