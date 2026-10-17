"""Mock perception engine for Gazebo simulation.

Extracts ground-truth obstacle positions from Gazebo physics, applies pod
frustum culling, and injects realistic sensor noise and detection dropout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np

from src.translate.config import LIDAR_MAX_RANGE
from src.translate.fusion_types import TrackedObstacle

MAX_PERCEPTION_RANGE: Final[float] = LIDAR_MAX_RANGE


@dataclass
class SimulatedEntity:
    """Ground truth state of an actor/obstacle in Gazebo."""

    name: str
    class_id: int
    class_name: str
    world_x: float
    world_y: float
    world_vx: float
    world_vy: float
    radius: float
    shape: str = "cylinder"  # "cylinder" or "box"
    length: float = 0.0  # longitudinal footprint (if box)
    width: float = 0.0  # lateral footprint (if box)
    height: float = 0.8
    yaw: float = 0.0  # orientation (radians)
    is_dynamic: bool = False
    patrol_distance: float = 0.0  # max excursion before reversing
    origin_x: float = 0.0
    origin_y: float = 0.0

    def step(self, dt: float) -> None:
        """Advance dynamic entity position along velocity vector with boundary bouncing."""
        if not self.is_dynamic:
            return
        speed = math.hypot(self.world_vx, self.world_vy)
        if speed < 1e-4:
            return

        self.world_x += self.world_vx * dt
        self.world_y += self.world_vy * dt

        if self.patrol_distance > 0.0:
            dist_from_origin = math.hypot(
                self.world_x - self.origin_x, self.world_y - self.origin_y
            )
            if dist_from_origin >= self.patrol_distance:
                # Reverse velocity vector
                self.world_vx = -self.world_vx
                self.world_vy = -self.world_vy


class MockPerceptionEngine:
    """Simulates realistic Camera-LiDAR fusion output from simulation ground truth."""

    def __init__(
        self,
        noise_std_base: float = 0.05,
        noise_std_rate: float = 0.01,
        dropout_probability: float = 0.03,
        random_seed: int | None = None,
    ) -> None:
        self.noise_std_base = noise_std_base
        self.noise_std_rate = noise_std_rate
        self.dropout_probability = dropout_probability
        self.rng = np.random.default_rng(random_seed)

    def filter_and_noise(
        self,
        car_x: float,
        car_y: float,
        car_yaw: float,
        entities: list[SimulatedEntity],
    ) -> list[TrackedObstacle]:
        """Transform world entities to ego-frame obstacles with sensor perturbations.

        Args:
            car_x: Vehicle X coordinate in world frame (meters).
            car_y: Vehicle Y coordinate in world frame (meters).
            car_yaw: Vehicle heading angle in world frame (radians).
            entities: Ground-truth world entities.

        Returns:
            List of TrackedObstacle instances in vehicle ego coordinates.
        """
        tracked: list[TrackedObstacle] = []
        cos_yaw = math.cos(-car_yaw)
        sin_yaw = math.sin(-car_yaw)

        for ent in entities:
            # 1. Transform to ego vehicle frame
            dx_w = ent.world_x - car_x
            dy_w = ent.world_y - car_y

            ego_x = dx_w * cos_yaw - dy_w * sin_yaw
            ego_y = dx_w * sin_yaw + dy_w * cos_yaw

            # 2. Check 360-degree surround range (all directions covered by 4 pods)
            dist = math.hypot(ego_x, ego_y)
            if dist > MAX_PERCEPTION_RANGE:
                continue

            # 3. Apply realistic dropout (occasional occlusion or missed detection)
            if self.rng.random() < self.dropout_probability:
                continue

            # 4. Inject distance-dependent Gaussian measurement noise
            sigma = self.noise_std_base + self.noise_std_rate * dist
            noise_x = self.rng.normal(0.0, sigma)
            noise_y = self.rng.normal(0.0, sigma)

            meas_x = float(ego_x + noise_x)
            meas_y = float(ego_y + noise_y)

            # Compute relative velocity in ego frame
            dvx_w = ent.world_vx
            dvy_w = ent.world_vy
            ego_vx = dvx_w * cos_yaw - dvy_w * sin_yaw
            ego_vy = dvx_w * sin_yaw + dvy_w * cos_yaw

            tracked.append(
                TrackedObstacle(
                    class_id=ent.class_id,
                    class_name=ent.class_name,
                    rel_x=meas_x,
                    rel_y=meas_y,
                    rel_vx=float(ego_vx),
                    rel_vy=float(ego_vy),
                    radius=ent.radius,
                    confidence=float(
                        max(0.45, 1.0 - (dist / MAX_PERCEPTION_RANGE) * 0.3)
                    ),
                )
            )

        return tracked
