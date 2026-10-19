"""Gymnasium reinforcement learning environment wrapping GzDRL for autonomous tractor navigation.

Features:
- Zero-copy C-extension DRLServer running Gazebo Harmonic in-process.
- 100% ground-truth physics & GPU LiDAR perception (zero mock or synthetic fallback).
- Deterministic stepping: server.run_N(N) without time.sleep().
- Complete 156-float observation space (kinematics, goal state, 360-deg ray distances & classes).
- Multi-phase mission (Point A -> Point B -> Remote Hold -> Point A).
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import sys
from typing import Any, Final

import gymnasium as gym
import numpy as np

# Ensure gzdrl _lib is importable
GZDRL_LIB = Path(__file__).resolve().parent / "gzdrl" / "_lib"
if str(GZDRL_LIB) not in sys.path:
    sys.path.insert(0, str(GZDRL_LIB))

import gzdrl  # noqa: E402

from src.translate.config import (  # noqa: E402
    CAR_LENGTH,
    CAR_WIDTH,
    COLLISION_PENALTY,
    CORRIDOR_DEADBAND_RATIO,
    CRAWL_PENALTY_WEIGHT,
    MAX_SPEED_MPS,
    MAX_STEER_RAD,
    MIN_CRUISE_SPEED_MPS,
    MIN_SPEED_MPS,
    OBSTACLE_PROXIMITY_WEIGHT,
    OBSTACLE_SAFETY_MARGIN_M,
)
from src.translate.fusion_types import GoalState  # noqa: E402
from .ground_segmentation import GroundSegmentationEngine, SegmentationResult  # noqa: E402
from .gzdrl_sensor_adapter import GzDrlSensorAdapter  # noqa: E402
from .spatial_map import EgoKinematics, SpatialMapBuilder  # noqa: E402
from .track_generator import ProceduralTrack  # noqa: E402

# Mission phase constants
PHASE_A_TO_B: Final[int] = 0
PHASE_REMOTE_HOLD: Final[int] = 1
PHASE_B_TO_A: Final[int] = 2

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class GzDrlCarEnv(gym.Env):
    """High-performance Gymnasium environment for autonomous vehicle navigation backed by GzDRL."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        world_path: str | Path | None = None,
        track_path: str | Path | None = None,
        model_name: str = "tractor",
        physics_hz: float = 200.0,
        control_hz: float = 20.0,
        render_mode: str | None = None,
        fast_lidar: bool = False,
        server_id: str | None = None,
        random_seed: int | None = None,
        pose_noise_std: float = 0.0,
        kinematics_noise_std: float = 0.0,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.physics_hz = float(physics_hz)
        self.control_hz = float(control_hz)
        self.steps_per_action = max(1, int(round(self.physics_hz / self.control_hz)))
        self.render_mode = render_mode
        self.fast_lidar = fast_lidar
        self.pose_noise_std = float(pose_noise_std)
        self.kinematics_noise_std = float(kinematics_noise_std)
        self.rng = np.random.default_rng(random_seed)

        # Paths
        if world_path is not None:
            self.world_path = Path(world_path)
        else:
            self.world_path = PROJECT_ROOT / "src/v2/model/procedural_track.sdf"

        if track_path is not None:
            self.track_path = Path(track_path)
        else:
            self.track_path = PROJECT_ROOT / "src/v2/model/procedural_track.json"

        if not self.world_path.exists():
            raise FileNotFoundError(f"Gazebo world SDF not found: {self.world_path}")
        if not self.track_path.exists():
            raise FileNotFoundError(f"Track JSON not found: {self.track_path}")

        # Ensure Gazebo resource and plugin paths are configured
        model_path = str(PROJECT_ROOT / "src/v2/model")
        plugin_path = str(GZDRL_LIB)
        cur_res = os.environ.get("GZ_SIM_RESOURCE_PATH", "")
        cur_plug = os.environ.get("GZ_SIM_SYSTEM_PLUGIN_PATH", "")
        os.environ["GZ_SIM_RESOURCE_PATH"] = f"{model_path}:{cur_res}" if cur_res else model_path
        os.environ["GZ_SIM_SYSTEM_PLUGIN_PATH"] = f"{plugin_path}:{cur_plug}" if cur_plug else plugin_path

        # Unique server identifier for worker isolation
        if server_id is None:
            server_id = f"gzdrl_car_{os.getpid()}_{id(self)}"
        self.server_id = server_id

        # Action and Observation spaces (156 floats)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(156,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Perception modules
        self.sensor_adapter = GzDrlSensorAdapter(
            front_sensor_name=f"{self.model_name}::chassis::lidar_front",
            rear_sensor_name=f"{self.model_name}::chassis::lidar_rear",
        )
        self.seg_engine = GroundSegmentationEngine()
        self.map_builder = SpatialMapBuilder()
        self._last_seg: SegmentationResult | None = None
        self._renderer: Any | None = None

        # Track and mission state
        self.track: ProceduralTrack = ProceduralTrack.load_json(self.track_path)
        self.max_episode_steps = self.track.max_episode_steps
        self.mission_phase: int = PHASE_A_TO_B
        self.hold_steps: int = 0
        self.hold_duration_steps: int = int(2.0 * self.control_hz)  # 2.0s hold
        self.step_count: int = 0

        # Kinematic states in world frame
        self.car_x: float = 0.0
        self.car_y: float = 0.0
        self.car_yaw: float = 0.0
        self.car_vx: float = 0.0
        self.car_vy: float = 0.0
        self.car_ax: float = 0.0
        self.car_yaw_rate: float = 0.0
        self.car_steer: float = 0.0
        self.last_steer_action: float = 0.0
        self.last_throttle_action: float = 0.0
        self.prev_dist_to_goal: float = 0.0

        # Instantiate GzDRL Server in-process
        self.server = gzdrl.DRLServer(
            self.server_id,
            str(self.world_path),
            [self.model_name],
            True,  # enable_sensors
        )

        # Step 20 iterations to initialize world and sensors
        self.server.run_N(20)

    @property
    def pose(self) -> tuple[float, float, float]:
        """Vehicle 2D pose: (x, y, yaw)."""
        return self.car_x, self.car_y, self.car_yaw

    def _sync_state_from_server(self) -> None:
        """Extract link state from Gazebo ECM via DRLServer."""
        state_map = self.server.state_info(self.model_name)
        if "chassis" not in state_map:
            raise RuntimeError(f"Link 'chassis' missing from state_info of {self.model_name}")

        state = state_map["chassis"]
        # state layout from GzDRL DRLHelperSystem::PostUpdate:
        # [0..2]: pos (x, y, z)
        # [3..6]: rot quaternion (w, x, y, z)
        # [7..9]: world linear velocity (vx, vy, vz)
        # [10..12]: body angular velocity (wx, wy, wz)
        # [13..15]: world linear acceleration (ax, ay, az)
        self.car_x = float(state[0])
        self.car_y = float(state[1])

        # Extract true planar yaw from quaternion (w, x, y, z)
        qw = float(state[3])
        qx = float(state[4])
        qy = float(state[5])
        qz = float(state[6])
        self.car_yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

        # Convert world velocities to body (chassis) frame
        vx_world = float(state[7])
        vy_world = float(state[8])
        cos_y = math.cos(self.car_yaw)
        sin_y = math.sin(self.car_yaw)

        self.car_vx = vx_world * cos_y + vy_world * sin_y
        self.car_vy = -vx_world * sin_y + vy_world * cos_y
        self.car_yaw_rate = float(state[12])

        ax_world = float(state[13])
        ay_world = float(state[14])
        self.car_ax = ax_world * cos_y + ay_world * sin_y

    def _check_contacts(self) -> bool:
        """Check for physical collisions via Gazebo Harmonic contact sensor on chassis."""
        contacts = self.server.get_contacts(self.model_name)
        if "chassis" not in contacts:
            return False
        for batch in contacts["chassis"]:
            if batch.contact_size() > 0:
                return True
        return False

    def _get_active_waypoints(self) -> np.ndarray:
        assert self.track is not None, "Track not loaded"
        if self.mission_phase == PHASE_A_TO_B:
            return self.track.waypoints_forward
        return self.track.waypoints_reverse

    def _get_target_goal(self) -> tuple[float, float, float]:
        assert self.track is not None, "Track not loaded"
        if self.mission_phase == PHASE_A_TO_B:
            return self.track.goal_pose
        return self.track.start_pose

    def _nearest_corridor_half_width(self) -> float:
        """Drivable half-width at the waypoint nearest the car."""
        assert self.track is not None, "Track not loaded"
        waypoints = self._get_active_waypoints()
        nearest_idx = int(
            np.argmin(
                np.hypot(
                    waypoints[:, 0] - self.car_x, waypoints[:, 1] - self.car_y
                )
            )
        )
        if self.mission_phase == PHASE_A_TO_B:
            return float(self.track.corridor_half_widths[nearest_idx])
        return float(self.track.corridor_half_widths[len(waypoints) - 1 - nearest_idx])

    def _compute_goal_state(self) -> GoalState:
        """Compute relative goal error vector and trajectory deviation metrics."""
        px, py, pyaw = self.car_x, self.car_y, self.car_yaw
        if self.pose_noise_std > 0.0:
            px = px + float(self.rng.normal(0.0, self.pose_noise_std))
            py = py + float(self.rng.normal(0.0, self.pose_noise_std))
            pyaw = pyaw + float(self.rng.normal(0.0, self.pose_noise_std * 0.4))

        waypoints = self._get_active_waypoints()
        gx, gy, _ = self._get_target_goal()

        dx_world = gx - px
        dy_world = gy - py
        dist_to_goal = math.hypot(dx_world, dy_world)

        cos_yaw = math.cos(pyaw)
        sin_yaw = math.sin(pyaw)
        dx_ego = dx_world * cos_yaw + dy_world * sin_yaw
        dy_ego = -dx_world * sin_yaw + dy_world * cos_yaw

        target_heading = math.atan2(dy_world, dx_world)
        heading_error = math.atan2(
            math.sin(target_heading - pyaw),
            math.cos(target_heading - pyaw),
        )

        dists_to_wps = np.hypot(waypoints[:, 0] - px, waypoints[:, 1] - py)
        nearest_idx = int(np.argmin(dists_to_wps))
        near_wp = waypoints[nearest_idx]

        lookahead_idx = min(len(waypoints) - 1, nearest_idx + 8)
        path_dx = waypoints[lookahead_idx, 0] - near_wp[0]
        path_dy = waypoints[lookahead_idx, 1] - near_wp[1]
        path_heading = (
            math.atan2(path_dy, path_dx)
            if (path_dx**2 + path_dy**2) > 0.01
            else near_wp[2]
        )

        path_heading_error = math.atan2(
            math.sin(path_heading - pyaw),
            math.cos(path_heading - pyaw),
        )

        rx = px - near_wp[0]
        ry = py - near_wp[1]
        cross_track_error = -rx * math.sin(path_heading) + ry * math.cos(path_heading)

        return GoalState(
            dx_ego=dx_ego,
            dy_ego=dy_ego,
            distance=dist_to_goal,
            heading_error=heading_error,
            cross_track_error=cross_track_error,
            path_heading_error=path_heading_error,
            phase=self.mission_phase,
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
        self.mission_phase = PHASE_A_TO_B
        self.hold_steps = 0
        self.last_steer_action = 0.0
        self.last_throttle_action = 0.0
        if self._renderer is not None:
            self._renderer.reset_trail()

        # Teleport tractor to Track Start Pose A
        sx, sy, syaw = self.track.start_pose
        pos = np.array([sx, sy, 0.0], dtype=np.float64)
        ori = np.array([0.0, 0.0, syaw], dtype=np.float64)
        self.server.reset_pos(self.model_name, pos, ori)

        # Clear velocity command and run 10 stabilization steps
        zero_cmd = np.array([0.0, 0.0], dtype=np.float64)
        self.server.set_ackermann_velocity_cmd(self.model_name, "chassis", zero_cmd)
        self.server.run_N(10)

        self._sync_state_from_server()
        goal_state = self._compute_goal_state()
        self.prev_dist_to_goal = goal_state.distance

        kinematics = EgoKinematics(
            vx=self.car_vx,
            vy=self.car_vy,
            steering_angle=self.car_steer,
            yaw_rate=self.car_yaw_rate,
            ax=self.car_ax,
            ay=self.car_vx * self.car_yaw_rate,
        )

        obs, self._last_seg = self.sensor_adapter.process_and_observe(
            server=self.server,
            seg_engine=self.seg_engine,
            map_builder=self.map_builder,
            kinematics=kinematics,
            goal_state=goal_state,
        )

        info: dict[str, Any] = {
            "phase": self.mission_phase,
            "pose": self.pose,
            "dist_to_goal": goal_state.distance,
            "speed": self.car_vx,
            "lidar_source": "gzdrl_inprocess_3d",
            "collision": False,
        }
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.step_count += 1
        steer_cmd = float(np.clip(action[0], -1.0, 1.0))
        throttle_cmd = float(np.clip(action[1], -1.0, 1.0))

        # Speed mapping: forward [0, 1] -> [0, MAX_SPEED_MPS], reverse [-1, 0) -> [MIN_SPEED_MPS, 0]
        if throttle_cmd >= 0.0:
            target_speed = throttle_cmd * MAX_SPEED_MPS
        else:
            target_speed = throttle_cmd * abs(MIN_SPEED_MPS)

        target_steer = steer_cmd * MAX_STEER_RAD

        # Set Ackermann command: [speed, steer]
        cmd = np.array([target_speed, target_steer], dtype=np.float64)
        self.server.set_ackermann_velocity_cmd(self.model_name, "chassis", cmd)

        # Step Gazebo simulation synchronously (no time.sleep!)
        self.server.run_N(self.steps_per_action)

        # Sync physical state and check contact collisions
        self._sync_state_from_server()
        collision = self._check_contacts()

        # Mission phase transitions
        goal_state = self._compute_goal_state()
        curr_dist = goal_state.distance
        reached_target = curr_dist < 2.0 and abs(self.car_vx) < 1.0

        phase_bonus = 0.0
        mission_completed = False

        if reached_target:
            if self.mission_phase == PHASE_A_TO_B:
                self.mission_phase = PHASE_REMOTE_HOLD
                self.hold_steps = 0
                phase_bonus += 50.0
            elif self.mission_phase == PHASE_REMOTE_HOLD:
                self.hold_steps += 1
                if self.hold_steps >= self.hold_duration_steps:
                    self.mission_phase = PHASE_B_TO_A
                    phase_bonus += 20.0
            elif self.mission_phase == PHASE_B_TO_A:
                mission_completed = True
                phase_bonus += 100.0

        # Perception and observation building
        kinematics = EgoKinematics(
            vx=self.car_vx,
            vy=self.car_vy,
            steering_angle=target_steer,
            yaw_rate=self.car_yaw_rate,
            ax=self.car_ax,
            ay=self.car_vx * self.car_yaw_rate,
        )

        obs, self._last_seg = self.sensor_adapter.process_and_observe(
            server=self.server,
            seg_engine=self.seg_engine,
            map_builder=self.map_builder,
            kinematics=kinematics,
            goal_state=goal_state,
        )

        # Obstacle proximity repulsive cost
        hazard_pts: list[np.ndarray] = []
        if self._last_seg.obstacle_points is not None and self._last_seg.obstacle_points.size > 0:
            hazard_pts.append(self._last_seg.obstacle_points[:, :2])
        if self._last_seg.negative_hazards is not None and self._last_seg.negative_hazards.size > 0:
            hazard_pts.append(self._last_seg.negative_hazards[:, :2])

        if len(hazard_pts) > 0:
            all_pts = np.vstack(hazard_pts)
            half_len = CAR_LENGTH / 2.0
            half_wid = CAR_WIDTH / 2.0
            dx = np.maximum(0.0, np.abs(all_pts[:, 0]) - half_len)
            dy = np.maximum(0.0, np.abs(all_pts[:, 1]) - half_wid)
            clearance = float(np.min(np.hypot(dx, dy)))
            if clearance < OBSTACLE_SAFETY_MARGIN_M:
                violation = 1.0 - (clearance / OBSTACLE_SAFETY_MARGIN_M)
                r_obstacle = -OBSTACLE_PROXIMITY_WEIGHT * (violation**2)
            else:
                r_obstacle = 0.0
        else:
            clearance = OBSTACLE_SAFETY_MARGIN_M
            r_obstacle = 0.0

        # Reward formulation
        half_road = self._nearest_corridor_half_width()
        track_len = self.track.nominal_length
        off_track = (
            abs(goal_state.cross_track_error) > (half_road + 2.0)
            or curr_dist > (track_len * 2.5)
        )

        dist_progress = (self.prev_dist_to_goal - curr_dist) * 5.0
        self.prev_dist_to_goal = curr_dist

        path_alignment = (self.car_vx / MAX_SPEED_MPS) * math.cos(
            goal_state.path_heading_error
        )
        r_velocity = 0.5 * max(0.0, path_alignment)

        # Centering penalty: penalty-free deadband within CORRIDOR_DEADBAND_RATIO of corridor
        # to permit obstacle avoidance maneuvers without continuous penalty.
        deadband_m = CORRIDOR_DEADBAND_RATIO * half_road
        excess_cte = max(0.0, abs(goal_state.cross_track_error) - deadband_m)
        buffer_zone = max(0.1, half_road - deadband_m)
        norm_cte = excess_cte / buffer_zone
        r_centering = -0.4 * (norm_cte**2)

        # Anti-crawl / anti-stalling penalty during transit phases
        if (
            self.mission_phase in (PHASE_A_TO_B, PHASE_B_TO_A)
            and self.car_vx < MIN_CRUISE_SPEED_MPS
        ):
            r_crawl = -CRAWL_PENALTY_WEIGHT * (
                1.0 - max(0.0, self.car_vx) / MIN_CRUISE_SPEED_MPS
            )
        else:
            r_crawl = 0.0

        r_heading = 0.2 * (math.cos(goal_state.path_heading_error) - 1.0)
        r_reverse = (
            -0.5 * abs(self.car_vx)
            if (self.mission_phase == PHASE_A_TO_B and self.car_vx < -0.05)
            else 0.0
        )
        r_smooth = (
            -0.02 * ((steer_cmd - self.last_steer_action) ** 2)
            - 0.01 * ((throttle_cmd - self.last_throttle_action) ** 2)
        )
        r_step = -0.05

        reward = (
            dist_progress
            + r_velocity
            + r_centering
            + r_heading
            + r_reverse
            + r_smooth
            + r_step
            + r_obstacle
            + r_crawl
            + phase_bonus
        )

        if collision:
            reward -= COLLISION_PENALTY
        elif off_track:
            reward -= COLLISION_PENALTY

        terminated = collision or off_track or mission_completed
        truncated = self.step_count >= self.max_episode_steps

        self.last_steer_action = steer_cmd
        self.last_throttle_action = throttle_cmd

        info: dict[str, Any] = {
            "collision": collision,
            "off_track": off_track,
            "reached_target": reached_target,
            "mission_completed": mission_completed,
            "phase": self.mission_phase,
            "pose": self.pose,
            "speed": self.car_vx,
            "lidar_source": "gzdrl_inprocess_3d",
            "obstacle_clearance": clearance,
            "r_obstacle": r_obstacle,
            "r_crawl": r_crawl,
        }
        return obs, float(reward), terminated, truncated, info

    @property
    def latest_raw_lidar_points_ego(self) -> np.ndarray:
        """Segmented non-ground 3D hits in ego coordinates."""
        if self._last_seg is None:
            return np.empty((0, 2), dtype=np.float32)
        parts = [p[:, :2] for p in (self._last_seg.obstacle_points, self._last_seg.negative_hazards) if p.size > 0]
        if not parts:
            return np.empty((0, 2), dtype=np.float32)
        return np.vstack(parts).astype(np.float32)

    def render(self) -> np.ndarray:
        """Render 2D top-down corridor visualization frame."""
        if self.render_mode is None:
            raise ValueError("render_mode was not set in __init__.")
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                from .env_renderer import CarEnvRenderer
                self._renderer = CarEnvRenderer()
            return self._renderer.render_frame(self)
        raise ValueError(f"Unsupported render_mode: {self.render_mode}")

    def close(self) -> None:
        """Clean up DRLServer instance and release memory."""
        if hasattr(self, "server"):
            del self.server
        self._renderer = None
        self._last_seg = None
