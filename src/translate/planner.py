"""Multi-lawn path planner using YOLO obstacle detections.

Plans paths between separate lawn areas, using YOLO detections
as dynamic obstacles to avoid during inter-lawn transit.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

from .detect import ObstacleDetector

logger = logging.getLogger(__name__)


@dataclass
class LawnArea:
    """Represents a mowable lawn zone."""

    name: str
    center: tuple[float, float]  # (x, y) in world coordinates (meters)
    radius: float  # approximate radius in meters
    mowed: bool = False


@dataclass
class Waypoint:
    """A navigation waypoint between lawns."""

    position: tuple[float, float]
    label: str = ""


@dataclass
class TransitPath:
    """A planned path from one lawn to another."""

    source: LawnArea
    target: LawnArea
    waypoints: list[Waypoint]
    obstacle_free: bool = True
    blocked_by: list[str] = field(default_factory=list)


class MultiLawnPlanner:
    """Plans navigation between multiple lawn areas.

    Maintains a list of lawn areas and uses YOLO detections
    to find obstacle-free paths for inter-lawn transit.

    Usage::

        planner = MultiLawnPlanner(detector)
        planner.add_lawn("front-yard", center=(0, 0), radius=10.0)
        planner.add_lawn("back-yard", center=(0, 25), radius=8.0)
        path = planner.plan_transit("front-yard", "back-yard")
    """

    def __init__(self, detector: ObstacleDetector) -> None:
        self.detector = detector
        self.lawns: dict[str, LawnArea] = {}
        self.transition_graph: dict[str, list[str]] = {}

    def add_lawn(
        self,
        name: str,
        center: tuple[float, float],
        radius: float,
    ) -> None:
        self.lawns[name] = LawnArea(name=name, center=center, radius=radius)
        if name not in self.transition_graph:
            self.transition_graph[name] = []

    def add_transition(self, from_lawn: str, to_lawn: str) -> None:
        """Register that a transition is possible between two lawns."""
        self.transition_graph.setdefault(from_lawn, [])
        if to_lawn not in self.transition_graph[from_lawn]:
            self.transition_graph[from_lawn].append(to_lawn)
        self.transition_graph.setdefault(to_lawn, [])
        if from_lawn not in self.transition_graph[to_lawn]:
            self.transition_graph[to_lawn].append(from_lawn)

    def find_path_bfs(
        self,
        start: str,
        goal: str,
    ) -> list[str] | None:
        """Find the shortest lawn sequence from start to goal using BFS."""
        if start == goal:
            return [start]
        if start not in self.transition_graph or goal not in self.transition_graph:
            return None

        visited = {start}
        queue = [[start]]
        while queue:
            path = queue.pop(0)
            current = path[-1]
            for neighbor in self.transition_graph.get(current, []):
                if neighbor == goal:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(path + [neighbor])
        return None

    def plan_transit(
        self,
        source_name: str,
        target_name: str,
        *,
        frame: np.ndarray | None = None,
    ) -> TransitPath | None:
        """Plan a path from one lawn to another.

        If a camera frame is provided, YOLO detections are used to check
        for obstacles along the straight-line path. Otherwise, the path
        is assumed obstacle-free.
        """
        if source_name not in self.lawns or target_name not in self.lawns:
            logger.error("Unknown lawn: %s or %s", source_name, target_name)
            return None

        source = self.lawns[source_name]
        target = self.lawns[target_name]

        path_sequence = self.find_path_bfs(source_name, target_name)
        if path_sequence is None:
            logger.warning("No route found from %s to %s", source_name, target_name)
            return None

        waypoints = self._compute_waypoints(path_sequence)

        blocked_by: list[str] = []
        obstacle_free = True
        if frame is not None:
            obstacle_free, blocked_by = self._check_path_clear(frame, waypoints)

        return TransitPath(
            source=source,
            target=target,
            waypoints=waypoints,
            obstacle_free=obstacle_free,
            blocked_by=blocked_by,
        )

    def _compute_waypoints(self, path_sequence: list[str]) -> list[Waypoint]:
        """Generate intermediate waypoints for a sequence of lawns."""
        if len(path_sequence) < 2:
            return []

        waypoints: list[Waypoint] = []
        for i in range(len(path_sequence) - 1):
            src = self.lawns[path_sequence[i]]
            dst = self.lawns[path_sequence[i + 1]]

            exit_angle = math.atan2(
                dst.center[1] - src.center[1],
                dst.center[0] - src.center[0],
            )
            exit_point = (
                src.center[0] + src.radius * math.cos(exit_angle),
                src.center[1] + src.radius * math.sin(exit_angle),
            )
            entry_point = (
                dst.center[0] - dst.radius * math.cos(exit_angle),
                dst.center[1] - dst.radius * math.sin(exit_angle),
            )

            waypoints.append(Waypoint(position=exit_point, label=f"exit-{src.name}"))
            waypoints.append(Waypoint(position=entry_point, label=f"enter-{dst.name}"))

        return waypoints

    def _check_path_clear(
        self,
        frame: np.ndarray,
        waypoints: list[Waypoint],
    ) -> tuple[bool, list[str]]:
        """Use YOLO to check if the path is clear of obstacles."""
        detections = self.detector.detect(frame)
        transit_obstacles = self.detector.filter_transit_obstacles(detections)

        if not transit_obstacles:
            return True, []

        blocked_classes = list({d.class_name for d in transit_obstacles})
        logger.warning(
            "Path blocked by %d obstacles: %s",
            len(transit_obstacles),
            ", ".join(blocked_classes),
        )
        return False, blocked_classes
