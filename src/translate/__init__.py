"""Obstacle detection and multi-lawn path planning module.

Uses YOLOv8s trained on the ROD-Dataset to detect real-world obstacles
during inter-lawn navigation.
"""

from .detect import ObstacleDetector
from .planner import MultiLawnPlanner

__all__ = ["ObstacleDetector", "MultiLawnPlanner"]
