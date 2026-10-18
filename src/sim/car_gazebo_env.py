"""Gymnasium environment for autonomous self-driving car navigation.

Provides standardized 156-float observations from 360-degree surround perception,
procedural track corridors with arbitrary walls/obstacles, and continuous non-holonomic
Ackermann control [steering, throttle/brake].
"""

from __future__ import annotations

import math
from pathlib import Path
import time
from typing import Any, Final

import numpy as np
import gymnasium as gym

from src.translate.config import (
    DT_SECONDS,
    MAX_SPEED_MPS,
    MAX_STEER_RAD,
    MIN_SPEED_MPS,
)
from src.translate.fusion_types import GoalState
from .ground_segmentation import GroundSegmentationEngine
from .spatial_map import EgoKinematics, SpatialMapBuilder
from .track_generator import ProceduralTrack

# Mission phase constants
PHASE_A_TO_B: Final[int] = 0
PHASE_REMOTE_HOLD: Final[int] = 1
PHASE_B_TO_A: Final[int] = 2


class CarGazeboEnv(gym.Env):
    """Reinforcement learning environment for non-holonomic Ackermann autonomous vehicle."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        random_seed: int | None = None,
        track_path: Path | None = None,
        render_mode: str | None = None,
        allow_reverse: bool = False,
        pose_noise_std: float = 0.0,
        kinematics_noise_std: float = 0.0,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.allow_reverse = allow_reverse
        self.pose_noise_std = float(pose_noise_std)
        self.kinematics_noise_std = float(kinematics_noise_std)
        self._renderer: Any | None = None
        self.rng = np.random.default_rng(random_seed)
        self.track_path = (
            track_path
            if track_path is not None
            else Path("src/v2/model/procedural_track.json")
        )

        # 156 floats: 6 kinematics + 6 GoalState + 72 ray distances + 72 ray classes
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(156,),
            dtype=np.float32,
        )

        # 2 continuous actions: [steering in [-1, 1], throttle/brake/reverse in [-1, 1]]
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.map_builder = SpatialMapBuilder()

        # Active procedural track
        self.track: ProceduralTrack | None = None
        self.mission_phase: int = PHASE_A_TO_B
        self.hold_steps: int = 0
        self.hold_duration_steps: int = 40  # 2 seconds at 20 Hz for remote teleop hold
        self.max_episode_steps: int = 2400

        # Live ROS 2 / Gazebo Bridge
        from .ros_bridge import Ros2GazeboBridge

        self.ros_bridge: Any | None = Ros2GazeboBridge()
        self._current_clouds: dict[str, np.ndarray] = {}
        self.seg_engine = GroundSegmentationEngine()

        # Vehicle internal state
        self.car_x: float = 0.0
        self.car_y: float = 0.0
        self.car_yaw: float = 0.0
        self.car_vx: float = 0.0  # Signed forward velocity (negative for reverse)
        self.car_vy: float = (
            0.0  # Lateral slip velocity (near zero in no-slip Ackermann)
        )
        self.car_steer: float = 0.0
        self.car_ax: float = 0.0
        self.car_yaw_rate: float = 0.0

        self.last_steer_action: float = 0.0
        self.last_throttle_action: float = 0.0
        self.step_count: int = 0
        self.prev_dist_to_goal: float = 0.0

    @property
    def pose(self) -> tuple[float, float, float]:
        return (self.car_x, self.car_y, self.car_yaw)

    def _get_active_waypoints(self) -> np.ndarray:
        assert self.track is not None, "Track not generated"
        if self.mission_phase == PHASE_A_TO_B:
            return self.track.waypoints_forward
        return self.track.waypoints_reverse

    def _get_target_goal(self) -> tuple[float, float, float]:
        assert self.track is not None, "Track not generated"
        if self.mission_phase == PHASE_A_TO_B:
            return self.track.goal_pose
        return self.track.start_pose

    def _nearest_corridor_half_width(self) -> float:
        """Drivable half-width at the waypoint nearest the car (road or yard)."""
        assert self.track is not None, "Track not generated"
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
        px, py, pyaw = self.car_x, self.car_y, self.car_yaw
        if self.pose_noise_std > 0.0:
            px = px + float(self.rng.normal(0.0, self.pose_noise_std))
            py = py + float(self.rng.normal(0.0, self.pose_noise_std))
            pyaw = pyaw + float(self.rng.normal(0.0, self.pose_noise_std * 0.4))
        waypoints = self._get_active_waypoints()
        gx, gy, _ = self._get_target_goal()
        # Euclidean distance and relative offset to goal
        dx_world = gx - px
        dy_world = gy - py
        dist_to_goal = math.hypot(dx_world, dy_world)

        # Transform target offset to vehicle ego frame
        cos_yaw = math.cos(pyaw)
        sin_yaw = math.sin(pyaw)
        dx_ego = dx_world * cos_yaw + dy_world * sin_yaw
        dy_ego = -dx_world * sin_yaw + dy_world * cos_yaw

        # Angle to target in ego frame
        target_heading = math.atan2(dy_world, dx_world)
        heading_error = math.atan2(
            math.sin(target_heading - pyaw),
            math.cos(target_heading - pyaw),
        )

        # Cross-track error to nearest waypoint on nominal path
        dists_to_wps = np.hypot(waypoints[:, 0] - px, waypoints[:, 1] - py)
        nearest_idx = int(np.argmin(dists_to_wps))
        near_wp = waypoints[nearest_idx]

        # Lookahead waypoint for path tangent
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

        # Lateral signed deviation from path tangent
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

        # Synchronize track geometry with the live Gazebo world JSON.
        if options and "track" in options and isinstance(options["track"], ProceduralTrack):
            self.track = options["track"]
        else:
            if not self.track_path.exists():
                raise FileNotFoundError(f"Track JSON not found: {self.track_path}")
            self.track = ProceduralTrack.load_json(self.track_path)

        self.step_count = 0
        self.mission_phase = PHASE_A_TO_B
        self.hold_steps = 0
        if self._renderer is not None:
            self._renderer.reset_trail()
        if self.track is not None:
            self.max_episode_steps = self.track.max_episode_steps

        # Spawn vehicle at Point A aligned with track
        sx, sy, syaw = self.track.start_pose
        self.car_x = sx
        self.car_y = sy
        self.car_yaw = syaw
        self.car_vx = 0.0  # start stationary
        self.car_vy = 0.0
        self.car_steer = 0.0
        self.car_ax = 0.0
        self.car_yaw_rate = 0.0
        self.last_steer_action = 0.0
        self.last_throttle_action = 0.0

        if self.ros_bridge is None:
            raise RuntimeError("ros_bridge is not initialized.")
        self.ros_bridge.reset_vehicle(sx, sy, syaw)
        # Wait up to 3.0s for background executor to receive initial state from Gazebo
        gate_timeout_sec = 3.0
        t_start = time.monotonic()
        while time.monotonic() - t_start < gate_timeout_sec:
            odom, clouds = self.ros_bridge.get_state()
            if odom is not None:
                if self.ros_bridge.has_world_odom:
                    self.car_x = odom["x"]
                    self.car_y = odom["y"]
                    self.car_yaw = odom["yaw"]
                else:
                    self.car_yaw = syaw + odom["yaw"]
                    self.car_x = (
                        sx + odom["x"] * math.cos(syaw) - odom["y"] * math.sin(syaw)
                    )
                    self.car_y = (
                        sy + odom["x"] * math.sin(syaw) + odom["y"] * math.cos(syaw)
                    )
                self.car_vx = odom["vx"]
                self.car_vy = odom.get("vy", 0.0)
                self.car_yaw_rate = odom["yaw_rate"]
                self.car_steer = odom.get("steering_angle", 0.0)
                self.car_ax = odom.get("ax", 0.0)
            if clouds:
                self._current_clouds = clouds
            if odom is not None and {"front", "rear"} <= set(clouds):
                break
            time.sleep(0.02)

        if not ({"front", "rear"} <= set(self._current_clouds)):
            raise RuntimeError(
                f"Gazebo 3D LiDAR clouds unavailable after {gate_timeout_sec:.1f}s gate (received: {set(self._current_clouds)})."
            )

        goal_state = self._compute_goal_state()
        self.prev_dist_to_goal = goal_state.distance

        obs = self._get_observation()
        info: dict[str, Any] = {
            "phase": self.mission_phase,
            "pose": self.pose,
            "dist_to_goal": goal_state.distance,
            "speed": self.car_vx,
            "lidar_source": "ros_gz_3d",
            "active_clouds": len(self._current_clouds),
            "collision": False,
        }
        return obs, info

    def _get_observation(self) -> np.ndarray:
        vx, vy = self.car_vx, self.car_vy
        steer, yaw_rate = self.car_steer, self.car_yaw_rate
        ax = self.car_ax
        ay = self.car_vx * self.car_yaw_rate
        if self.kinematics_noise_std > 0.0:
            vx = vx + float(self.rng.normal(0.0, self.kinematics_noise_std))
            vy = vy + float(self.rng.normal(0.0, self.kinematics_noise_std))
            steer = steer + float(self.rng.normal(0.0, self.kinematics_noise_std))
            yaw_rate = yaw_rate + float(self.rng.normal(0.0, self.kinematics_noise_std))
            ax = ax + float(self.rng.normal(0.0, self.kinematics_noise_std))
            ay = ay + float(self.rng.normal(0.0, self.kinematics_noise_std))
        kinematics = EgoKinematics(
            vx=vx,
            vy=vy,
            steering_angle=steer,
            yaw_rate=yaw_rate,
            ax=ax,
            ay=ay,
        )
        goal_state = self._compute_goal_state()

        if not self._current_clouds:
            raise RuntimeError("Gazebo 3D LiDAR clouds unavailable.")
        combined = np.vstack([self._current_clouds[k] for k in sorted(self._current_clouds)])
        seg = self.seg_engine.segment(combined)
        self._last_seg = seg
        return self.map_builder.build_observation(
            kinematics=kinematics,
            goal_state=goal_state,
            obstacle_points=seg.obstacle_points,
            negative_hazards=seg.negative_hazards,
        )

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.step_count += 1
        steer_cmd = float(np.clip(action[0], -1.0, 1.0))
        throttle_cmd = float(np.clip(action[1], -1.0, 1.0))

        # Target speed mapping: forward [0, 1] -> [0, MAX_SPEED_MPS]
        # Reverse [-1, 0) -> [MIN_SPEED_MPS, 0] where |MIN_SPEED_MPS| < MAX_SPEED_MPS
        if throttle_cmd >= 0.0:
            target_speed = throttle_cmd * MAX_SPEED_MPS
        else:
            target_speed = throttle_cmd * abs(MIN_SPEED_MPS)

        if self.ros_bridge is None:
            raise RuntimeError("ros_bridge is not initialized.")
        target_steer = steer_cmd * MAX_STEER_RAD
        self.ros_bridge.publish_action(target_steer, target_speed)
        # Enforce simulation step pacing (50ms at 20 Hz) so Gazebo physics simulates the step
        time.sleep(DT_SECONDS)
        odom, clouds = self.ros_bridge.get_state()
        if odom is not None:
            if self.ros_bridge.has_world_odom:
                self.car_x = odom["x"]
                self.car_y = odom["y"]
                self.car_yaw = odom["yaw"]
            else:
                sx, sy, syaw = (
                    self.track.start_pose if self.track else (0.0, 0.0, 0.0)
                )
                self.car_yaw = syaw + odom["yaw"]
                self.car_x = (
                    sx + odom["x"] * math.cos(syaw) - odom["y"] * math.sin(syaw)
                )
                self.car_y = (
                    sy + odom["x"] * math.sin(syaw) + odom["y"] * math.cos(syaw)
                )
            self.car_vx = odom["vx"]
            self.car_vy = odom.get("vy", 0.0)
            self.car_yaw_rate = odom["yaw_rate"]
            self.car_steer = odom.get("steering_angle", target_steer)
            self.car_ax = odom.get("ax", 0.0)
        if clouds:
            self._current_clouds = clouds

        # Gazebo Harmonic Contact System is the sole ground truth
        collision = self.ros_bridge.get_collision()

        # 6. Mission Phase Transitions
        goal_state = self._compute_goal_state()
        curr_dist = goal_state.distance
        reached_target = curr_dist < 2.0 and abs(self.car_vx) < 1.0

        phase_bonus = 0.0
        mission_completed = False

        if reached_target:
            if self.mission_phase == PHASE_A_TO_B:
                # Arrived at Point B! Switch to Remote Hold
                self.mission_phase = PHASE_REMOTE_HOLD
                self.hold_steps = 0
                phase_bonus += 50.0
            elif self.mission_phase == PHASE_REMOTE_HOLD:
                self.hold_steps += 1
                if self.hold_steps >= self.hold_duration_steps:
                    # Switch to return navigation: B -> A
                    self.mission_phase = PHASE_B_TO_A
                    phase_bonus += 20.0
            elif self.mission_phase == PHASE_B_TO_A:
                # Successfully returned to Point A!
                mission_completed = True
                phase_bonus += 100.0

        # 7. Reward Formulation & Bounds Checking
        half_road = self._nearest_corridor_half_width()
        assert self.track is not None, "Track not generated"
        track_len = self.track.nominal_length
        off_track = (
            abs(goal_state.cross_track_error) > (half_road + 2.0)
            or curr_dist > (track_len * 2.5)
        )

        # Progress reward: Euclidean delta towards goal + forward velocity alignment
        dist_progress = (self.prev_dist_to_goal - curr_dist) * 5.0
        self.prev_dist_to_goal = curr_dist

        path_alignment = (self.car_vx / MAX_SPEED_MPS) * math.cos(
            goal_state.path_heading_error
        )
        r_velocity = 0.5 * max(0.0, path_alignment)

        # Centering penalty: quadratically scaled by corridor half-width
        norm_cte = abs(goal_state.cross_track_error) / max(1.0, half_road)
        r_centering = -0.3 * (norm_cte**2)

        # Heading alignment penalty
        r_heading = 0.2 * (math.cos(goal_state.path_heading_error) - 1.0)

        # Reverse driving penalty in forward mission phase
        r_reverse = (
            -0.5 * abs(self.car_vx)
            if (self.mission_phase == PHASE_A_TO_B and self.car_vx < -0.05)
            else 0.0
        )

        # Action smoothness penalties
        r_smooth = (
            -0.05 * ((steer_cmd - self.last_steer_action) ** 2)
            - 0.02 * ((throttle_cmd - self.last_throttle_action) ** 2)
        )

        # Step time penalty to discourage standing still / idling
        r_step = -0.05

        reward = (
            dist_progress
            + r_velocity
            + r_centering
            + r_heading
            + r_reverse
            + r_smooth
            + r_step
            + phase_bonus
        )

        if collision:
            reward -= 100.0
        elif off_track:
            reward -= 100.0

        terminated = collision or off_track or mission_completed
        truncated = self.step_count >= self.max_episode_steps

        self.last_steer_action = steer_cmd
        self.last_throttle_action = throttle_cmd

        obs = self._get_observation()
        info: dict[str, Any] = {
            "collision": collision,
            "off_track": off_track,
            "reached_target": reached_target,
            "mission_completed": mission_completed,
            "phase": self.mission_phase,
            "pose": self.pose,
            "speed": self.car_vx,
            "lidar_source": "ros_gz_3d",
            "active_clouds": len(self._current_clouds),
        }
        return obs, float(reward), terminated, truncated, info

    @property
    def latest_raw_lidar_points_ego(self) -> np.ndarray:
        """Segmented non-ground 3D hits in ego coordinates (obstacles + negative hazards)."""
        seg = getattr(self, "_last_seg", None)
        if seg is None:
            return np.empty((0, 2), dtype=np.float32)
        parts = [p[:, :2] for p in (seg.obstacle_points, seg.negative_hazards) if p.size > 0]
        if not parts:
            return np.empty((0, 2), dtype=np.float32)
        return np.vstack(parts).astype(np.float32)

    def render(self) -> np.ndarray:
        """Render high-fidelity top-down RGB frame of the environment."""
        if self.render_mode is None:
            raise ValueError(
                "Cannot call render() when render_mode was not specified in __init__."
            )
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                from .env_renderer import CarEnvRenderer

                self._renderer = CarEnvRenderer()
            return self._renderer.render_frame(self)
        raise ValueError(
            f"Unsupported render_mode: {self.render_mode!r}. Supported modes: ['rgb_array']"
        )

    def close(self) -> None:
        if self.ros_bridge is not None:
            self.ros_bridge.close()
            self.ros_bridge = None
