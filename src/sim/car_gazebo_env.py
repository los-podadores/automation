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
    CAR_LENGTH,
    CAR_WIDTH,
    DT_SECONDS,
    MAX_ACCEL_MPSS,
    MAX_BRAKE_MPSS,
    MAX_SPEED_MPS,
    MAX_STEER_RAD,
    MIN_SPEED_MPS,
    WHEELBASE,
)
from src.translate.fusion_types import GoalState
from .mock_perception import MockPerceptionEngine
from .spatial_map import EgoKinematics, LaserScanData, SpatialMapBuilder
from .track_generator import (
    DEFAULT_WALL_THICKNESS,
    ProceduralTrack,
    ProceduralTrackGenerator,
)

# Mission phase constants
PHASE_A_TO_B: Final[int] = 0
PHASE_REMOTE_HOLD: Final[int] = 1
PHASE_B_TO_A: Final[int] = 2


class CarGazeboEnv(gym.Env):
    """Reinforcement learning environment for non-holonomic Ackermann autonomous vehicle."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        sim_mode: str = "mock",
        num_obstacles: int = 4,
        track_length: float = 95.0,
        road_width: float = 7.0,
        random_seed: int | None = None,
        track_path: Path | None = None,
    ) -> None:
        super().__init__()
        self.sim_mode = sim_mode
        self.num_obstacles = num_obstacles
        self.track_length = track_length
        self.road_width = road_width
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
        self.mock_perception = MockPerceptionEngine(random_seed=random_seed)
        self.track_gen = ProceduralTrackGenerator(default_road_width=road_width)

        # Active procedural track
        self.track: ProceduralTrack | None = None
        self.mission_phase: int = PHASE_A_TO_B
        self.hold_steps: int = 0
        self.hold_duration_steps: int = 40  # 2 seconds at 20 Hz for remote teleop hold
        self.max_episode_steps: int = 2400

        # Live ROS 2 / Gazebo Bridge
        self.ros_bridge: Any | None = None
        self._current_scans: dict[str, LaserScanData] = {}
        if self.sim_mode == "gazebo":
            from .ros_bridge import Ros2GazeboBridge

            self.ros_bridge = Ros2GazeboBridge()

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

    def _compute_goal_state(self) -> GoalState:
        waypoints = self._get_active_waypoints()
        gx, gy, _ = self._get_target_goal()

        # Euclidean distance and relative offset to goal
        dx_world = gx - self.car_x
        dy_world = gy - self.car_y
        dist_to_goal = math.hypot(dx_world, dy_world)

        # Transform target offset to vehicle ego frame
        cos_yaw = math.cos(self.car_yaw)
        sin_yaw = math.sin(self.car_yaw)
        dx_ego = dx_world * cos_yaw + dy_world * sin_yaw
        dy_ego = -dx_world * sin_yaw + dy_world * cos_yaw

        # Angle to target in ego frame
        target_heading = math.atan2(dy_world, dx_world)
        heading_error = math.atan2(
            math.sin(target_heading - self.car_yaw),
            math.cos(target_heading - self.car_yaw),
        )

        # Cross-track error to nearest waypoint on nominal path
        dists_to_wps = np.hypot(
            waypoints[:, 0] - self.car_x, waypoints[:, 1] - self.car_y
        )
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
            math.sin(path_heading - self.car_yaw),
            math.cos(path_heading - self.car_yaw),
        )

        # Lateral signed deviation from path tangent
        rx = self.car_x - near_wp[0]
        ry = self.car_y - near_wp[1]
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

    def _transform_walls_to_ego(
        self,
    ) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        if not self.track:
            return []

        cos_yaw = math.cos(self.car_yaw)
        sin_yaw = math.sin(self.car_yaw)
        ego_walls: list[tuple[tuple[float, float], tuple[float, float]]] = []

        for (wx1, wy1), (wx2, wy2) in self.track.wall_segments:
            # Shift to ego origin
            dx1 = wx1 - self.car_x
            dy1 = wy1 - self.car_y
            dx2 = wx2 - self.car_x
            dy2 = wy2 - self.car_y

            # Rotate by -car_yaw
            ex1 = dx1 * cos_yaw + dy1 * sin_yaw
            ey1 = -dx1 * sin_yaw + dy1 * cos_yaw
            ex2 = dx2 * cos_yaw + dy2 * sin_yaw
            ey2 = -dx2 * sin_yaw + dy2 * cos_yaw

            ego_walls.append(((ex1, ey1), (ex2, ey2)))

        return ego_walls

    @staticmethod
    def _point_to_segment_dist_sq(
        px: float, py: float, x1: float, y1: float, x2: float, y2: float
    ) -> float:
        dx = x2 - x1
        dy = y2 - y1
        l2 = dx * dx + dy * dy
        if l2 < 1e-8:
            return (px - x1) ** 2 + (py - y1) ** 2
        t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / l2))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        return (px - proj_x) ** 2 + (py - proj_y) ** 2

    @staticmethod
    def _point_to_aabb_dist_sq(px: float, py: float, hx: float, hy: float) -> float:
        dx = max(0.0, abs(px) - hx)
        dy = max(0.0, abs(py) - hy)
        return dx * dx + dy * dy

    @staticmethod
    def _segment_intersects_aabb(
        x1: float, y1: float, x2: float, y2: float, hx: float, hy: float
    ) -> bool:
        if abs(x1) <= hx and abs(y1) <= hy:
            return True
        if abs(x2) <= hx and abs(y2) <= hy:
            return True
        dx = x2 - x1
        dy = y2 - y1
        p = [-dx, dx, -dy, dy]
        q = [x1 + hx, hx - x1, y1 + hy, hy - y1]
        u1, u2 = 0.0, 1.0
        for pi, qi in zip(p, q, strict=True):
            if pi == 0.0:
                if qi < 0.0:
                    return False
            elif pi < 0.0:
                t = qi / pi
                if t > u2:
                    return False
                if t > u1:
                    u1 = t
            else:
                t = qi / pi
                if t < u1:
                    return False
                if t < u2:
                    u2 = t
        return u1 <= u2

    @classmethod
    def _segment_aabb_dist(
        cls, x1: float, y1: float, x2: float, y2: float, hx: float, hy: float
    ) -> float:
        if cls._segment_intersects_aabb(x1, y1, x2, y2, hx, hy):
            return 0.0
        d_sq = min(
            cls._point_to_aabb_dist_sq(x1, y1, hx, hy),
            cls._point_to_aabb_dist_sq(x2, y2, hx, hy),
            cls._point_to_segment_dist_sq(-hx, -hy, x1, y1, x2, y2),
            cls._point_to_segment_dist_sq(-hx, hy, x1, y1, x2, y2),
            cls._point_to_segment_dist_sq(hx, -hy, x1, y1, x2, y2),
            cls._point_to_segment_dist_sq(hx, hy, x1, y1, x2, y2),
        )
        return math.sqrt(d_sq)

    def _check_wall_collision(self) -> bool:
        """Check if vehicle oriented bounding box intersects wall geometry (thickness 0.30m)."""
        if not self.track:
            return False

        hx = CAR_LENGTH / 2.0
        hy = CAR_WIDTH / 2.0
        # Wall box half-thickness is DEFAULT_WALL_THICKNESS / 2.0 (0.15m), plus 2cm contact margin
        r_eff = (DEFAULT_WALL_THICKNESS / 2.0) + 0.02

        ego_walls = self._transform_walls_to_ego()
        for (x1, y1), (x2, y2) in ego_walls:
            if self._segment_aabb_dist(x1, y1, x2, y2, hx, hy) <= r_eff:
                return True

        return False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Synchronize track geometry
        if self.sim_mode == "gazebo" and self.track_path.exists():
            self.track = ProceduralTrack.load_json(self.track_path)
        elif (
            options
            and "track" in options
            and isinstance(options["track"], ProceduralTrack)
        ):
            self.track = options["track"]
        else:
            preset = (
                str(options["preset"])
                if options and "preset" in options
                else "demo_90deg"
            )
            track_seed = int(self.rng.integers(0, 1_000_000))
            self.track = self.track_gen.generate_track(
                seed=track_seed,
                length=self.track_length,
                road_width=self.road_width,
                num_obstacles=self.num_obstacles,
                preset=preset,
            )

        self.step_count = 0
        self.mission_phase = PHASE_A_TO_B
        self.hold_steps = 0
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

        if self.sim_mode == "gazebo":
            if self.ros_bridge is None:
                raise RuntimeError(
                    "sim_mode is 'gazebo' but ros_bridge is not initialized."
                )
            # Wait up to 1.0s for background executor to receive initial state from Gazebo
            t_start = time.monotonic()
            while time.monotonic() - t_start < 1.0:
                odom, scans = self.ros_bridge.get_state()
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
                    self.car_yaw_rate = odom["yaw_rate"]
                if scans:
                    self._current_scans = scans
                if odom is not None and len(scans) == 4:
                    break
                time.sleep(0.02)

        goal_state = self._compute_goal_state()
        self.prev_dist_to_goal = goal_state.distance

        obs = self._get_observation()
        info: dict[str, Any] = {
            "phase": self.mission_phase,
            "pose": self.pose,
            "dist_to_goal": goal_state.distance,
            "speed": self.car_vx,
            "lidar_source": "ros_gz" if (self.sim_mode == "gazebo" and len(self._current_scans) > 0) else "mock",
            "active_pods": len(self._current_scans) if self.sim_mode == "gazebo" else 4,
            "raw_lidar_points_ego": self.latest_raw_lidar_points_ego,
            "raw_pod_beams": self.latest_raw_pod_beams,
        }
        return obs, info

    def _get_observation(self) -> np.ndarray:
        kinematics = EgoKinematics(
            vx=self.car_vx,
            vy=self.car_vy,
            steering_angle=self.car_steer,
            yaw_rate=self.car_yaw_rate,
            ax=self.car_ax,
            ay=self.car_vx * self.car_yaw_rate,
        )
        goal_state = self._compute_goal_state()

        tracked_obstacles = self.mock_perception.filter_and_noise(
            self.car_x,
            self.car_y,
            self.car_yaw,
            self.track.obstacles if self.track else [],
        )

        if self.sim_mode == "gazebo":
            if self.ros_bridge is None:
                raise RuntimeError(
                    "sim_mode is 'gazebo' but ros_bridge is not initialized."
                )
            return self.map_builder.build_observation(
                kinematics=kinematics,
                goal_state=goal_state,
                obstacles=tracked_obstacles,
                pod_scans=self._current_scans,
            )

        # Mock perception fallback for pure mock simulation
        wall_segments_ego = self._transform_walls_to_ego()

        return self.map_builder.build_observation(
            kinematics=kinematics,
            goal_state=goal_state,
            obstacles=tracked_obstacles,
            wall_segments_ego=wall_segments_ego,
        )

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.step_count += 1
        steer_cmd = float(np.clip(action[0], -1.0, 1.0))
        throttle_cmd = float(np.clip(action[1], -1.0, 1.0))

        if self.sim_mode == "gazebo" and self.ros_bridge is not None:
            target_steer = steer_cmd * MAX_STEER_RAD
            if throttle_cmd >= 0.0:
                if self.car_vx < -0.1:
                    # Vehicle is reversing; forward throttle acts as active brake to stop
                    target_speed = 0.0
                else:
                    target_speed = throttle_cmd * MAX_SPEED_MPS
            else:
                if self.car_vx > 0.1:
                    # Vehicle is moving forward; reverse throttle acts as active brake to stop
                    target_speed = 0.0
                else:
                    # Vehicle is stationary or reversing; engage reverse drive
                    target_speed = throttle_cmd * abs(MIN_SPEED_MPS)
            self.ros_bridge.publish_action(target_steer, target_speed)
            odom, scans = self.ros_bridge.get_state()
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
                self.car_yaw_rate = odom["yaw_rate"]
                self.car_steer = target_steer
            if scans:
                self._current_scans = scans
        else:
            # 1. Steering Dynamics (Ackermann angle with actuator lag)
            target_steer = steer_cmd * MAX_STEER_RAD
            self.car_steer += (target_steer - self.car_steer) * 0.35

            # 2. Longitudinal Dynamics (Separate forward acceleration and reverse/braking)
            if throttle_cmd >= 0.0:
                target_accel = throttle_cmd * MAX_ACCEL_MPSS
                self.car_ax = target_accel
                self.car_vx = min(MAX_SPEED_MPS, self.car_vx + self.car_ax * DT_SECONDS)
            else:
                # Negative action commands deceleration / reverse
                target_decel = abs(throttle_cmd) * MAX_BRAKE_MPSS
                self.car_ax = -target_decel
                self.car_vx = max(MIN_SPEED_MPS, self.car_vx + self.car_ax * DT_SECONDS)

            # 3. Non-Holonomic Bicycle / Ackermann Kinematics
            if abs(self.car_vx) > 1e-4:
                self.car_yaw_rate = (self.car_vx / WHEELBASE) * math.tan(self.car_steer)
            else:
                self.car_yaw_rate = 0.0

            self.car_yaw += self.car_yaw_rate * DT_SECONDS
            self.car_x += self.car_vx * math.cos(self.car_yaw) * DT_SECONDS
            self.car_y += self.car_vx * math.sin(self.car_yaw) * DT_SECONDS

        # 4. Update Dynamic Obstacles
        if self.sim_mode == "gazebo" and self.ros_bridge is not None:
            # Synchronize true obstacle states from Gazebo Harmonic
            obs_states = self.ros_bridge.get_obstacle_states()
            if self.track:
                for i, obs in enumerate(self.track.obstacles):
                    if obs.is_dynamic and i in obs_states:
                        st = obs_states[i]
                        obs.world_x = st["x"]
                        obs.world_y = st["y"]
                        obs.world_vx = st["vx"]
                        obs.world_vy = st["vy"]
                    elif obs.is_dynamic:
                        obs.step(DT_SECONDS)
        else:
            if self.track:
                for obs in self.track.obstacles:
                    obs.step(DT_SECONDS)

        # 5. Collision Checks
        wall_collision = self._check_wall_collision()
        obstacle_collision = False
        if self.track and self.track.obstacles:
            cos_y = math.cos(self.car_yaw)
            sin_y = math.sin(self.car_yaw)
            hx = CAR_LENGTH / 2.0
            hy = CAR_WIDTH / 2.0
            for obs in self.track.obstacles:
                dx = obs.world_x - self.car_x
                dy = obs.world_y - self.car_y
                ex = dx * cos_y + dy * sin_y
                ey = -dx * sin_y + dy * cos_y
                dist_to_box = math.sqrt(self._point_to_aabb_dist_sq(ex, ey, hx, hy))
                if dist_to_box <= (obs.radius + 0.02):
                    obstacle_collision = True
                    break

        if self.sim_mode == "gazebo" and self.ros_bridge is not None:
            gz_contact = self.ros_bridge.get_collision()
            collision = gz_contact or wall_collision or obstacle_collision
        else:
            collision = wall_collision or obstacle_collision

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

        # 7. Reward Formulation
        progress = (self.prev_dist_to_goal - curr_dist) * 4.0
        self.prev_dist_to_goal = curr_dist

        # Penalize excessive cross-track deviation
        r_centering = -0.15 * min(4.0, goal_state.cross_track_error**2)
        # Smooth steering penalty
        r_smooth = -0.04 * ((steer_cmd - self.last_steer_action) ** 2)

        reward = progress + r_centering + r_smooth + phase_bonus

        if collision:
            reward -= 100.0

        terminated = collision or mission_completed
        truncated = self.step_count >= self.max_episode_steps

        self.last_steer_action = steer_cmd
        self.last_throttle_action = throttle_cmd

        obs = self._get_observation()
        info: dict[str, Any] = {
            "collision": collision,
            "wall_collision": wall_collision,
            "obstacle_collision": obstacle_collision,
            "reached_target": reached_target,
            "mission_completed": mission_completed,
            "phase": self.mission_phase,
            "pose": self.pose,
            "speed": self.car_vx,
            "dist_to_goal": curr_dist,
            "lidar_source": "ros_gz" if (self.sim_mode == "gazebo" and len(self._current_scans) > 0) else "mock",
            "active_pods": len(self._current_scans) if self.sim_mode == "gazebo" else 4,
            "raw_lidar_points_ego": self.latest_raw_lidar_points_ego,
            "raw_pod_beams": self.latest_raw_pod_beams,
        }
        return obs, float(reward), terminated, truncated, info

    @property
    def latest_raw_lidar_points_ego(self) -> np.ndarray:
        """Raw (N, 2) Cartesian hit points in ego coordinates from live 4-pod LiDARs."""
        if self.sim_mode == "gazebo" and self._current_scans:
            return self.map_builder.extract_point_cloud_ego(self._current_scans)
        if self.map_builder.latest_mock_pod_scans:
            return self.map_builder.extract_point_cloud_ego(self.map_builder.latest_mock_pod_scans)
        return np.empty((0, 2), dtype=np.float32)

    @property
    def latest_raw_pod_beams(self) -> list[dict[str, Any]]:
        """Physical ray beam segments per sensor pod with exact mounting origins."""
        tracked_obs = (
            self.mock_perception.filter_and_noise(
                self.car_x,
                self.car_y,
                self.car_yaw,
                self.track.obstacles if self.track else [],
            )
            if self.track
            else []
        )
        if self.sim_mode == "gazebo" and self._current_scans:
            return self.map_builder.extract_pod_beam_segments(
                self._current_scans, obstacles=tracked_obs
            )
        if self.map_builder.latest_mock_pod_scans:
            return self.map_builder.extract_pod_beam_segments(
                self.map_builder.latest_mock_pod_scans, obstacles=tracked_obs
            )
        return []

    def close(self) -> None:
        if self.ros_bridge is not None:
            self.ros_bridge.close()
            self.ros_bridge = None
