"""Procedural track generator for autonomous car navigation.

Synthesizes arbitrary road corridors with guaranteed-passable obstacle spacing,
Catmull-Rom splines, flanking walls, an open destination courtyard at Point B,
and forward/reverse waypoint trajectories (A -> B -> A). Also exports
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
DEFAULT_ROAD_WIDTH: Final[float] = 7.0  # Generous two-lane road width (meters)


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
            if length < 0.1:
                continue
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
            <ambient>0.85 0.15 0.15 1.0</ambient>
            <diffuse>0.90 0.20 0.20 1.0</diffuse>
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
      <pose>0 0 35 0 0 0</pose>
      <diffuse>0.95 0.95 0.95 1</diffuse>
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
              <size>250 250</size>
            </plane>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>250 250</size>
            </plane>
          </geometry>
          <material>
            <ambient>0.22 0.25 0.22 1.0</ambient>
            <diffuse>0.32 0.38 0.30 1.0</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- Point A Beacon (Start) -->
    <model name="beacon_point_a">
      <static>true</static>
      <pose>{ax:.3f} {ay:.3f} 1.8 0 0 0</pose>
      <link name="link">
        <visual name="visual">
          <geometry>
            <sphere><radius>0.50</radius></sphere>
          </geometry>
          <material>
            <ambient>0.1 0.85 0.2 1.0</ambient>
            <diffuse>0.1 0.95 0.2 1.0</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- Point B Beacon (Goal) -->
    <model name="beacon_point_b">
      <static>true</static>
      <pose>{bx:.3f} {by:.3f} 1.8 0 0 0</pose>
      <link name="link">
        <visual name="visual">
          <geometry>
            <sphere><radius>0.50</radius></sphere>
          </geometry>
          <material>
            <ambient>0.2 0.45 0.95 1.0</ambient>
            <diffuse>0.2 0.55 1.0 1.0</diffuse>
          </material>
        </visual>
      </link>
    </model>
{"".join(walls_sdf)}
{"".join(obstacles_sdf)}

    <!-- Include Large-Scale Ackermann Car -->
    <include>
      <uri>model://tractor.sdf</uri>
      <pose>{ax:.3f} {ay:.3f} 0.40 0 0 {self.start_pose[2]:.4f}</pose>
    </include>
  </world>
</sdf>
"""
        output_path.write_text(sdf_content, encoding="utf-8")


class ProceduralTrackGenerator:
    """Generates guaranteed-solvable road corridors with walls, curves, and obstacle layouts."""

    def __init__(self, default_road_width: float = DEFAULT_ROAD_WIDTH) -> None:
        self.default_road_width = default_road_width

    def _catmull_rom_spline(self, control_pts: np.ndarray, total_points: int) -> np.ndarray:
        """Compute smooth Catmull-Rom spline passing exactly through all control points."""
        # Duplicate start and end points for endpoint tangency
        pts = np.vstack([control_pts[0], control_pts, control_pts[-1]])
        curve: list[np.ndarray] = []
        n_segs = len(pts) - 3
        pts_per_seg = max(4, total_points // n_segs)

        for i in range(n_segs):
            p0, p1, p2, p3 = pts[i : i + 4]
            for u in np.linspace(0.0, 1.0, pts_per_seg, endpoint=False):
                u2 = u * u
                u3 = u2 * u
                pt = 0.5 * (
                    (2.0 * p1)
                    + (-p0 + p2) * u
                    + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * u2
                    + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * u3
                )
                curve.append(pt)

        curve.append(pts[-2])
        return np.array(curve, dtype=np.float32)

    def generate_track(
        self,
        seed: int | None = None,
        length: float = 95.0,
        road_width: float | None = None,
        num_obstacles: int = 4,
    ) -> ProceduralTrack:
        """Synthesize a complete track corridor from Point A to Point B.

        Guarantees:
        1. Exact endpoint preservation (no boundary distortion at Point A or Point B).
        2. Drivable corridor clearance of at least 4.5m around EVERY obstacle.
        3. Longitudinal obstacle separation of at least 18m to prevent chicanes or blockades.
        4. Spacious open turning courtyard (16m wide) at Point B.
        """
        rng = np.random.default_rng(seed)
        r_width = road_width if road_width is not None else self.default_road_width

        # 1. Generate smooth, realistic control points
        num_ctrl = rng.integers(4, 6)
        ctrl_pts: list[tuple[float, float]] = [(0.0, 0.0)]
        curr_x, curr_y = 0.0, 0.0
        curr_heading = float(rng.uniform(-0.2, 0.2))

        seg_len = length / float(num_ctrl)
        for _ in range(num_ctrl):
            curr_x += seg_len * math.cos(curr_heading)
            curr_y += seg_len * math.sin(curr_heading)
            ctrl_pts.append((float(curr_x), float(curr_y)))
            # Mild turning angle per segment to guarantee R >= 5.0m
            heading_delta = float(rng.uniform(-0.40, 0.40))
            curr_heading += heading_delta

        ctrl_arr = np.array(ctrl_pts, dtype=np.float32)
        total_steps = int(length * 2.0)
        spline_pts = self._catmull_rom_spline(ctrl_arr, total_steps)

        xs = spline_pts[:, 0]
        ys = spline_pts[:, 1]
        n_pts = len(xs)

        # Accurate heading angles along spline
        dxs = np.gradient(xs)
        dys = np.gradient(ys)
        yaws = np.arctan2(dys, dxs)

        waypoints_forward = np.column_stack([xs, ys, yaws]).astype(np.float32)

        # Reverse waypoints for return phase (B -> A)
        rev_xs = np.flip(xs)
        rev_ys = np.flip(ys)
        rev_dxs = np.gradient(rev_xs)
        rev_dys = np.gradient(rev_ys)
        rev_yaws = np.arctan2(rev_dys, rev_dxs)
        waypoints_reverse = np.column_stack([rev_xs, rev_ys, rev_yaws]).astype(np.float32)

        start_pose = (float(xs[0]), float(ys[0]), float(yaws[0]))
        goal_pose = (float(xs[-1]), float(ys[-1]), float(yaws[-1]))

        # 2. Generate Flanking Walls with Point B Courtyard
        # In the final 20% of the track, expand road into a 16m wide turning courtyard
        left_wall_pts: list[tuple[float, float]] = []
        right_wall_pts: list[tuple[float, float]] = []

        step_decimate = 5
        court_start_idx = int(n_pts * 0.80)

        for k in range(0, n_pts, step_decimate):
            px, py, psi = xs[k], ys[k], yaws[k]

            if k < court_start_idx:
                current_w = r_width
            else:
                # Smoothly expand road width from r_width (7m) to 16m courtyard
                t_exp = (k - court_start_idx) / max(1, n_pts - 1 - court_start_idx)
                current_w = r_width + t_exp * (16.0 - r_width)

            half_w = current_w / 2.0
            # Left boundary: +half_w along normal (psi + pi/2)
            lx = px - half_w * math.sin(psi)
            ly = py + half_w * math.cos(psi)
            # Right boundary: -half_w along normal (psi - pi/2)
            rx = px + half_w * math.sin(psi)
            ry = py - half_w * math.cos(psi)
            left_wall_pts.append((float(lx), float(ly)))
            right_wall_pts.append((float(rx), float(ry)))

        # Ensure last point is exactly captured
        last_px, last_py, last_psi = xs[-1], ys[-1], yaws[-1]
        last_half_w = 16.0 / 2.0
        last_l = (
            float(last_px - last_half_w * math.sin(last_psi)),
            float(last_py + last_half_w * math.cos(last_psi)),
        )
        last_r = (
            float(last_px + last_half_w * math.sin(last_psi)),
            float(last_py - last_half_w * math.cos(last_psi)),
        )
        left_wall_pts.append(last_l)
        right_wall_pts.append(last_r)

        # Convert boundary points into discrete wall segments
        wall_segments: list[tuple[tuple[float, float], tuple[float, float]]] = []

        # Left wall: includes an open gate entrance near 60%-70% of the track
        for i in range(len(left_wall_pts) - 1):
            progress_ratio = i / float(len(left_wall_pts))
            if not (0.60 <= progress_ratio <= 0.72):
                wall_segments.append((left_wall_pts[i], left_wall_pts[i + 1]))

        # Right wall: continuous perimeter
        for i in range(len(right_wall_pts) - 1):
            wall_segments.append((right_wall_pts[i], right_wall_pts[i + 1]))

        # Back enclosing wall at Point B courtyard (behind Point B, with 6m buffer)
        court_back_dist = 6.0
        back_center_x = last_px + court_back_dist * math.cos(last_psi)
        back_center_y = last_py + court_back_dist * math.sin(last_psi)
        back_l = (
            float(back_center_x - last_half_w * math.sin(last_psi)),
            float(back_center_y + last_half_w * math.cos(last_psi)),
        )
        back_r = (
            float(back_center_x + last_half_w * math.sin(last_psi)),
            float(back_center_y - last_half_w * math.cos(last_psi)),
        )

        wall_segments.append((last_l, back_l))
        wall_segments.append((back_l, back_r))
        wall_segments.append((back_r, last_r))

        # 3. Procedurally Place Obstacles with GUARANTEED Clearance
        # - Strictly separated by at least 18m along path
        # - Strictly hugged to left curb or right curb (never centered)
        # - Leaves at least 4.5m of open drivable road width on opposite side
        obstacles: list[SimulatedEntity] = []
        obs_palette = [
            ("Car", 1, 0.95),  # Parked vehicle on curb
            ("Person", 2, 0.40),  # Pedestrian on shoulder
            ("Traffic Barrel", 3, 0.45),  # Work barrel along curb
            ("Tree", 3, 0.50),  # Tree on verge
        ]

        # Obstacles placed only in the road section (between 15% and 75% progress)
        road_start_idx = int(n_pts * 0.15)
        road_end_idx = int(n_pts * 0.75)
        min_step_gap = int(18.0 * 2.0)  # at least 18m apart

        curr_obs_idx = road_start_idx + rng.integers(10, 20)
        side_toggle = 1.0  # Alternate sides so vehicle has clear slalom line

        while curr_obs_idx < road_end_idx and len(obstacles) < num_obstacles:
            cls_name, cls_id, rad = obs_palette[int(rng.integers(0, len(obs_palette)))]
            cx = float(xs[curr_obs_idx])
            cy = float(ys[curr_obs_idx])
            c_yaw = float(yaws[curr_obs_idx])

            # Hug obstacle to the chosen curb:
            # lateral_offset places obstacle near wall, leaving entire other lane wide open!
            half_w = r_width / 2.0
            lateral_offset = side_toggle * (half_w - rad - 0.35)

            ox = cx - lateral_offset * math.sin(c_yaw)
            oy = cy + lateral_offset * math.cos(c_yaw)

            # Some pedestrians cross slowly from shoulder to curb
            vx, vy = 0.0, 0.0
            if cls_name == "Person" and rng.random() > 0.5:
                cross_speed = float(rng.uniform(0.5, 0.8))
                vx = -cross_speed * math.sin(c_yaw) * side_toggle
                vy = cross_speed * math.cos(c_yaw) * side_toggle

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

            # Move to next obstacle with guaranteed longitudinal spacing
            curr_obs_idx += min_step_gap + rng.integers(10, 30)
            side_toggle *= -1.0  # Alternate sides

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
