"""Obstacle detection, sensor fusion, and spatial tracking module."""

from .fusion_types import (
    CLASS_FREE,
    CLASS_PEDESTRIAN,
    CLASS_STATIC_HAZARD,
    CLASS_VEHICLE,
    PodExtrinsics,
    TrackedObstacle,
)

__all__ = [
    "CLASS_FREE",
    "CLASS_PEDESTRIAN",
    "CLASS_STATIC_HAZARD",
    "CLASS_VEHICLE",
    "PodExtrinsics",
    "TrackedObstacle",
    "ObstacleDetector",
    "MultiLawnPlanner",
    "LidarCameraFusionEngine",
]


def __getattr__(name: str):
    if name == "ObstacleDetector":
        from .detect import ObstacleDetector

        return ObstacleDetector
    if name == "MultiLawnPlanner":
        from .planner import MultiLawnPlanner

        return MultiLawnPlanner
    if name == "LidarCameraFusionEngine":
        from .lidar_fusion import LidarCameraFusionEngine

        return LidarCameraFusionEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
