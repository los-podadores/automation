"""Gymnasium environment for autonomous self-driving car navigation.

Provides standardized 154-float observations from spatial perception and
continuous Ackermann control [steering, throttle/brake].
"""

from __future__ import annotations

import math
from typing import Any, Final

import gymnasium as gym
import numpy as np

from .mock_perception import MockPerceptionEngine, SimulatedEntity
from .spatial_map import EgoKinematics, PathDeviation, SpatialMapBuilder

# Dynamic limits
MAX_STEER_RAD: Final[float] = 0.523599  # ~30 degrees
MAX_SPEED_MPS: Final[float] = 6.0  # 6 m/s (~21.6 km/h)
MAX_ACCEL_MPSS: Final[float] = 3.0
MAX_LATERAL_ERR_METERS: Final[float] = 4.0
COLLISION_RADIUS_METERS: Final[float] = 0.65
DT_SECONDS: Final[float] = 0.05  # 20 Hz control loop


class CarGazeboEnv(gym.Env):
    """Reinforcement learning environment for Ackermann-steering autonomous vehicle."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        sim_mode: str = "mock",
        num_obstacles: int = 5,
        random_seed: int | None = None,
    ) -> None:
        super().__init__()
        self.sim_mode = sim_mode
        self.num_obstacles = num_obstacles
        self.rng = np.random.default_rng(random_seed)

        # 154 floats: 6 kinematics + 4 path deviations + 72 ray distances + 72 ray classes
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(154,),
            dtype=np.float32,
        )

        # 2 continuous actions: [steering_ratio in [-1, 1], throttle_brake_ratio in [-1, 1]]
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.map_builder = SpatialMapBuilder()
        self.mock_perception = MockPerceptionEngine(random_seed=random_seed)

        # Vehicle internal state
        self.car_x: float = 0.0
        self.car_y: float = 0.0
        self.car_yaw: float = 0.0
        self.car_vx: float = 0.0
        self.car_vy: float = 0.0
        self.car_steer: float = 0.0
        self.car_ax: float = 0.0
        self.car_yaw_rate: float = 0.0

        self.last_steer_action: float = 0.0
        self.last_throttle_action: float = 0.0
        self.step_count: int = 0
        self.entities: list[SimulatedEntity] = []

    def _spawn_random_obstacles(self) -> None:
        """Spawn dynamic and static obstacles along the forward path."""
        self.entities.clear()
        types = [
            (2, "Car", 1.2),
            (3, "Person", 0.4),
            (18, "Traffic Cone", 0.25),
            (16, "Bus", 1.8),
        ]
        for i in range(self.num_obstacles):
            cls_id, cls_name, radius = self.rng.choice(types)
            obs_x = float(self.rng.uniform(6.0 + i * 5.0, 10.0 + i * 6.0))
            obs_y = float(self.rng.uniform(-2.5, 2.5))
            # Some obstacles are static, some dynamic
            vx = float(self.rng.uniform(-0.5, 0.5)) if cls_name in ("Car", "Bus") else 0.0
            self.entities.append(
                SimulatedEntity(
                    name=f"obs_{i}_{cls_name}",
                    class_id=int(cls_id),
                    class_name=str(cls_name),
                    world_x=obs_x,
                    world_y=obs_y,
                    world_vx=vx,
                    world_vy=0.0,
                    radius=float(radius),
                )
            )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.step_count = 0
        self.car_x = 0.0
        self.car_y = float(self.rng.uniform(-0.5, 0.5))
        self.car_yaw = float(self.rng.uniform(-0.05, 0.05))
        self.car_vx = float(self.rng.uniform(1.0, 2.5))
        self.car_vy = 0.0
        self.car_steer = 0.0
        self.car_ax = 0.0
        self.car_yaw_rate = 0.0
        self.last_steer_action = 0.0
        self.last_throttle_action = 0.0

        self._spawn_random_obstacles()
        obs = self._get_observation()
        return obs, {}

    def _get_observation(self) -> np.ndarray:
        kinematics = EgoKinematics(
            vx=self.car_vx,
            vy=self.car_vy,
            steering_angle=self.car_steer,
            yaw_rate=self.car_yaw_rate,
            ax=self.car_ax,
            ay=self.car_vx * self.car_yaw_rate,
        )
        deviation = PathDeviation(
            lateral_error=self.car_y,
            heading_error=self.car_yaw,
            curvature_now=0.0,
            curvature_lookahead=0.0,
        )
        tracked_obstacles = self.mock_perception.filter_and_noise(
            self.car_x, self.car_y, self.car_yaw, self.entities
        )
        return self.map_builder.build_observation(kinematics, deviation, tracked_obstacles)

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.step_count += 1
        steer_cmd = float(np.clip(action[0], -1.0, 1.0))
        throttle_cmd = float(np.clip(action[1], -1.0, 1.0))

        # Target steering angle and acceleration
        target_steer = steer_cmd * MAX_STEER_RAD
        # Simple steering actuator first-order lag
        self.car_steer += (target_steer - self.car_steer) * 0.4

        # Throttle/Brake control
        target_accel = throttle_cmd * MAX_ACCEL_MPSS
        self.car_ax = target_accel
        self.car_vx = max(0.0, min(MAX_SPEED_MPS, self.car_vx + self.car_ax * DT_SECONDS))

        # Bicycle kinematic model update
        wheelbase = 0.80  # from tractor geometry (80 cm)
        self.car_yaw_rate = (self.car_vx / wheelbase) * math.tan(self.car_steer)
        self.car_yaw += self.car_yaw_rate * DT_SECONDS
        self.car_x += self.car_vx * math.cos(self.car_yaw) * DT_SECONDS
        self.car_y += self.car_vx * math.sin(self.car_yaw) * DT_SECONDS

        # Update dynamic obstacle positions
        for ent in self.entities:
            ent.world_x += ent.world_vx * DT_SECONDS
            ent.world_y += ent.world_vy * DT_SECONDS

        # Check collision with any obstacle
        collision = False
        for ent in self.entities:
            dist = math.hypot(ent.world_x - self.car_x, ent.world_y - self.car_y)
            if dist < (COLLISION_RADIUS_METERS + ent.radius):
                collision = True
                break

        # Check lane / road boundary exit
        out_of_bounds = abs(self.car_y) > MAX_LATERAL_ERR_METERS

        terminated = collision or out_of_bounds
        truncated = self.step_count >= 500

        # Reward computation
        r_progress = self.car_vx * math.cos(self.car_yaw) * DT_SECONDS
        r_centering = -0.4 * (self.car_y**2)
        r_heading = -0.2 * (self.car_yaw**2)
        r_smooth = -0.05 * ((steer_cmd - self.last_steer_action) ** 2)

        reward = r_progress + r_centering + r_heading + r_smooth

        if collision:
            reward -= 200.0
        elif out_of_bounds:
            reward -= 100.0

        self.last_steer_action = steer_cmd
        self.last_throttle_action = throttle_cmd

        obs = self._get_observation()
        info: dict[str, Any] = {
            "collision": collision,
            "out_of_bounds": out_of_bounds,
            "car_x": self.car_x,
            "car_y": self.car_y,
            "car_vx": self.car_vx,
        }
        return obs, float(reward), terminated, truncated, info

    def close(self) -> None:
        pass
