"""Procedural track generator for autonomous car navigation.

Synthesizes arbitrary road corridors with splines, flanking walls, static/dynamic
obstacles, and forward/reverse waypoint trajectories (A -> B -> A). Also exports
procedural Gazebo Harmonic SDF worlds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from .mock_perception import SimulatedEntity

DEFAULT_WALL_HEIGHT: Final[float] = 1.20
DEFAULT_WALL_THICKNESS: Final[float] = 0.25


@dataclass(frozen=True)
class ProceduralTrack:
    """Represents a generated navigation track environment."""

    seed: int
    start_pose: tuple[float, float, float]  # (x, y, yaw) at Point A
    goal_pose: tuple[float, float, float]  # (x, y, yaw) at Point B
    waypoints_forward: np.ndarray  # Shape (K, 3) [x, y, yaw] from A to B
    waypoints_reverse: np.ndarray  # Shape (K, 3) [x, y, yaw] from B to A
    wall_segments: list[tuple[tuple[float, float], tuple[float, float]]]  # 2D line segments
    obstacles: list[SimulatedEntity]  # Static and dynamic obstacles
    road_width: float

    def export_gazebo_sdf(self, output_path: Path) -> None:
        """Export the procedural track as a valid Gazebo Harmonic SDF world."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        walls_sdf: list[str] = []
        for i, ((x1, y1), (x2, y2)) in enumerate(self.wall_segments):
            mx = (x1 + x2) / 2.0
            my = (y1 + y2) / 2.0
            dx = x2 - x1
            dy = y2 - y1
            length = math.hypot(dx, dy)
            yaw = math.atan2(dy, dx)
            z = DEFAULT_WALL_HEIGHT / 2.0

            wall_xml = f"""
    <!-- Wall {i} -->
    <model name="wall_{i}">
      <static>true</static>
      <pose>{mx:.3f} {my:.3f} {z:.3f} 0 0 {yaw:.4f}</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>{length:.3f} {DEFAULT_WALL_THICKNESS:.3f} {DEFAULT_WALL_HEIGHT:.3f}</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>{length:.3f} {DEFAULT_WALL_THICKNESS:.3f} {DEFAULT_WALL_HEIGHT:.3f}</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.15 0.15 1.0</ambient>
            <diffuse>0.85 0.2 0.2 1.0</diffuse>
          </material>
        </visual>
      </link>
    </model>"""
            walls_sdf.append(wall_xml)

        obstacles_sdf: list[str] = []
        for i, obs in enumerate(self.obstacles):
            ox, oy, oz = obs.world_x, obs.world_y, 0.4
            r = obs.radius
            obs_xml = f"""
    <!-- Obstacle {i}: {obs.class_name} -->
    <model name="obstacle_{i}">
      <static>true</static>
      <pose>{ox:.3f} {oy:.3f} {oz:.3f} 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <cylinder>
              <radius>{r:.3f}</radius>
              <length>0.80</length>
            </cylinder>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <cylinder>
              <radius>{r:.3f}</radius>
              <length>0.80</length>
            </cylinder>
          </geometry>
          <material>
            <ambient>0.9 0.6 0.1 1.0</ambient>
            <diffuse>0.95 0.65 0.15 1.0</diffuse>
          </material>
        </visual>
      </link>
    </model>"""
            obstacles_sdf.append(obs_xml)

        ax, ay, _ = self.start_pose
        bx, by, _ = self.goal_pose

        sdf_content = f"""<?xml version="1.0" ?>
<sdf version="1.8">
  <world name="procedural_car_track">
    <physics name="1ms" type="ignored">
      <max_step_size>0.005</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>

    <!-- Sun / Lighting -->
    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 30 0 0 0</pose>
      <diffuse>0.9 0.9 0.9 1</diffuse>
      <specular>0.3 0.3 0.3 1</specular>
      <direction>-0.5 0.1 -0.9</direction>
    </light>

    <!-- Ground Plane -->
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>200 200</size>
            </plane>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>200 200</size>
            </plane>
          </geometry>
          <material>
            <ambient>0.25 0.28 0.25 1.0</ambient>
            <diffuse>0.35 0.40 0.32 1.0</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- Point A Beacon (Start) -->
    <model name="beacon_point_a">
      <static>true</static>
      <pose>{ax:.3f} {ay:.3f} 1.5 0 0 0</pose>
      <link name="link">
        <visual name="visual">
          <geometry>
            <sphere><radius>0.45</radius></sphere>
          </geometry>
          <material>
            <ambient>0.1 0.8 0.2 1.0</ambient>
            <diffuse>0.1 0.9 0.2 1.0</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- Point B Beacon (Goal) -->
    <model name="beacon_point_b">
      <static>true</static>
      <pose>{bx:.3f} {by:.3f} 1.5 0 0 0</pose>
      <link name="link">
        <visual name="visual">
          <geometry>
            <sphere><radius>0.45</radius></sphere>
          </geometry>
          <material>
            <ambient>0.2 0.4 0.95 1.0</ambient>
            <diffuse>0.2 0.5 1.0 1.0</diffuse>
          </material>
        </visual>
      </link>
    </model>
{"".join(walls_sdf)}
{"".join(obstacles_sdf)}

    <!-- Include Large-Scale Ackermann Car -->
    <include>
      <uri>model://tractor.sdf</uri>
      <pose>{ax:.3f} {ay:.3f} 0.4 0 0 {self.start_pose[2]:.4f}</pose>
    </include>
  </world>
</sdf>
"""
        output_path.write_text(sdf_content, encoding="utf-8")


class ProceduralTrackGenerator:
    """Generates varied, realistic road corridors with walls, curves, and obstacle layouts."""

    def __init__(self, default_road_width: float = 5.0) -> None:
        self.default_road_width = default_road_width

    def generate_track(
        self,
        seed: int | None = None,
        length: float = 90.0,
        road_width: float | None = None,
        num_obstacles: int = 6,
    ) -> ProceduralTrack:
        """Synthesize a complete track corridor from Point A to Point B.

        Args:
            seed: Random seed for deterministic generation.
            length: Approximate nominal path length in meters (default 90m).
            road_width: Width of drivable corridor in meters.
            num_obstacles: Number of static/dynamic obstacles to scatter.

        Returns:
            ProceduralTrack containing waypoints, wall line segments, and obstacles.
        """
        rng = np.random.default_rng(seed)
        r_width = road_width if road_width is not None else self.default_road_width

        # 1. Generate 4-6 smooth control points using random curvature changes
        num_ctrl = rng.integers(4, 6)
        ctrl_pts = [(0.0, 0.0)]
        curr_x, curr_y = 0.0, 0.0
        curr_heading = float(rng.uniform(-0.3, 0.3))

        seg_len = length / float(num_ctrl)
        for _ in range(num_ctrl):
            curr_x += seg_len * math.cos(curr_heading)
            curr_y += seg_len * math.sin(curr_heading)
            ctrl_pts.append((float(curr_x), float(curr_y)))
            # Max turn angle per segment kept within car turning radius capabilities (Ackermann R_min >= 4.0m)
            heading_delta = float(rng.uniform(-0.55, 0.55))
            curr_heading += heading_delta

        # 2. Interpolate smooth dense centerline spline points (0.5m spacing)
        ctrl_arr = np.array(ctrl_pts, dtype=np.float32)
        total_steps = int(length * 2.0)
        t_ctrl = np.linspace(0, 1, len(ctrl_pts))
        t_eval = np.linspace(0, 1, total_steps)

        xs = np.interp(t_eval, t_ctrl, ctrl_arr[:, 0])
        ys = np.interp(t_eval, t_ctrl, ctrl_arr[:, 1])

        # Smooth using a rolling boxcar window to eliminate sharp kinks
        window_size = 9
        xs = np.convolve(xs, np.ones(window_size) / window_size, mode="same")
        ys = np.convolve(ys, np.ones(window_size) / window_size, mode="same")

        # Compute headings along centerline
        dxs = np.gradient(xs)
        dys = np.gradient(ys)
        yaws = np.arctan2(dys, dxs)

        waypoints_forward = np.column_stack([xs, ys, yaws]).astype(np.float32)

        # Reverse waypoints for return phase (B -> A)
        # Heading is flipped 180 degrees for forward-facing return drive
        rev_xs = np.flip(xs)
        rev_ys = np.flip(ys)
        rev_dxs = np.gradient(rev_xs)
        rev_dys = np.gradient(rev_ys)
        rev_yaws = np.arctan2(rev_dys, rev_dxs)
        waypoints_reverse = np.column_stack([rev_xs, rev_ys, rev_yaws]).astype(np.float32)

        start_pose = (float(xs[0]), float(ys[0]), float(yaws[0]))
        goal_pose = (float(xs[-1]), float(ys[-1]), float(yaws[-1]))

        # 3. Generate flanking corridor walls
        half_w = r_width / 2.0
        left_wall_pts: list[tuple[float, float]] = []
        right_wall_pts: list[tuple[float, float]] = []

        step_decimate = 6  # Subsample spline to create clean wall segments
        for k in range(0, total_steps, step_decimate):
            px, py, psi = xs[k], ys[k], yaws[k]
            # Left boundary: +half_w along normal (psi + pi/2)
            lx = px - half_w * math.sin(psi)
            ly = py + half_w * math.cos(psi)
            # Right boundary: -half_w along normal (psi - pi/2)
            rx = px + half_w * math.sin(psi)
            ry = py - half_w * math.cos(psi)
            left_wall_pts.append((lx, ly))
            right_wall_pts.append((rx, ry))

        # Convert boundary points into discrete wall segments with intermittent gaps/gates
        wall_segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
        for i in range(len(left_wall_pts) - 1):
            # Introduce an occasional open section / gate on the left (e.g. at 70% of track length)
            if not (0.65 <= (i / len(left_wall_pts)) <= 0.78):
                wall_segments.append((left_wall_pts[i], left_wall_pts[i + 1]))
            # Right wall is continuous perimeter
            wall_segments.append((right_wall_pts[i], right_wall_pts[i + 1]))

        # 4. Procedurally place static and dynamic obstacles along corridor
        obstacles: list[SimulatedEntity] = []
        obs_classes = [
            ("Car", 1, 1.1),
            ("Person", 2, 0.4),
            ("Traffic Barrel", 3, 0.45),
            ("Tree", 3, 0.55),
            ("Bench", 3, 0.6),
        ]

        # Place obstacles between 15% and 85% of track progress
        cand_indices = rng.choice(
            np.arange(int(total_steps * 0.15), int(total_steps * 0.85)),
            size=min(num_obstacles, int(total_steps * 0.5)),
            replace=False,
        )

        for obs_idx in cand_indices:
            cls_name, cls_id, rad = obs_classes[int(rng.integers(0, len(obs_classes)))]
            cx, cy, c_yaw = xs[obs_idx], ys[obs_idx], yaws[obs_idx]

            # Place obstacle with random lateral offset within the drivable lane
            # Leave at least CAR_WIDTH + 0.6m passing clearance
            lateral_offset = float(rng.uniform(-half_w + rad + 0.3, half_w - rad - 0.3))
            ox = cx - lateral_offset * math.sin(c_yaw)
            oy = cy + lateral_offset * math.cos(c_yaw)

            # Some obstacles are dynamic pedestrians crossing the road
            vx, vy = 0.0, 0.0
            if cls_name == "Person" and rng.random() > 0.4:
                # Slow walking speed perpendicular to road (0.5 to 0.9 m/s)
                cross_speed = float(rng.uniform(0.5, 0.9))
                vx = -cross_speed * math.sin(c_yaw)
                vy = cross_speed * math.cos(c_yaw)

            obstacles.append(
                SimulatedEntity(
                    name=f"obs_{len(obstacles)}_{cls_name}",
                    class_id=cls_id,
                    class_name=cls_name,
                    world_x=ox,
                    world_y=oy,
                    world_vx=vx,
                    world_vy=vy,
                    radius=rad,
                )
            )

        return ProceduralTrack(
            seed=seed if seed is not None else 0,
            start_pose=start_pose,
            goal_pose=goal_pose,
            waypoints_forward=waypoints_forward,
            waypoints_reverse=waypoints_reverse,
            wall_segments=wall_segments,
            obstacles=obstacles,
            road_width=r_width,
        )
