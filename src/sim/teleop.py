"""Interactive pygame-based teleoperation and 360-degree perception viewer for CarGazeboEnv.

Supports driving the large-scale non-holonomic Ackermann vehicle along procedurally
generated corridors (A -> B -> A), viewing 360-degree LiDAR returns, and exporting
worlds to 3D Gazebo Harmonic.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Final


# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pygame  # noqa: E402

from src.sim.car_gazebo_env import (  # noqa: E402
    CarGazeboEnv,
)
from src.translate.config import (  # noqa: E402
    CAR_LENGTH,
    CAR_WIDTH,
    FRONT_LIDAR_X,
    LIDAR_MAX_RANGE,
    LIDAR_NUM_RAYS,
    MAX_SPEED_MPS,
    MIN_SPEED_MPS,
    REAR_LIDAR_X,
    WHEELBASE,
)
from src.translate.fusion_types import (  # noqa: E402
    CLASS_FREE,
    CLASS_PEDESTRIAN,
    CLASS_VEHICLE,
)

# Window Configuration
WINDOW_W: Final[int] = 1280
WINDOW_H: Final[int] = 840
VIEW_W: Final[int] = 880
PANEL_W: Final[int] = 400
FPS: Final[int] = 30

# LiDAR Display Modes
LIDAR_MODE_PHYSICAL_BEAMS: Final[str] = "physical"
LIDAR_MODE_POLICY_GRID: Final[str] = "policy"
LIDAR_MODE_POINT_CLOUD: Final[str] = "point_cloud"

# Colors (Dark Modern Theme)
BG_COLOR: Final[tuple[int, int, int]] = (18, 22, 28)
PANEL_BG: Final[tuple[int, int, int]] = (26, 32, 42)
ROAD_COLOR: Final[tuple[int, int, int]] = (36, 44, 56)
LANE_COLOR: Final[tuple[int, int, int]] = (230, 195, 45)
WALL_COLOR: Final[tuple[int, int, int]] = (215, 55, 55)
GRID_COLOR: Final[tuple[int, int, int]] = (30, 38, 48)
TEXT_WHITE: Final[tuple[int, int, int]] = (240, 245, 250)
TEXT_MUTED: Final[tuple[int, int, int]] = (140, 155, 175)
TEXT_GREEN: Final[tuple[int, int, int]] = (70, 220, 120)
TEXT_RED: Final[tuple[int, int, int]] = (240, 70, 70)
TEXT_CYAN: Final[tuple[int, int, int]] = (60, 200, 240)
TEXT_YELLOW: Final[tuple[int, int, int]] = (245, 200, 50)

# Entity Colors
CAR_BODY_COLOR: Final[tuple[int, int, int]] = (
    235,
    130,
    30,
)  # Orange matching test drive footprint
CAR_CABIN_COLOR: Final[tuple[int, int, int]] = (195, 100, 20)
TIRE_COLOR: Final[tuple[int, int, int]] = (20, 20, 24)
OBS_VEHICLE_COLOR: Final[tuple[int, int, int]] = (60, 140, 240)
OBS_PED_COLOR: Final[tuple[int, int, int]] = (245, 170, 40)
OBS_STATIC_COLOR: Final[tuple[int, int, int]] = (235, 75, 75)

# Viewport parameters
PIXELS_PER_METER: Final[float] = 14.0  # 1 meter = 14 pixels


class CarTeleopApp:
    """Interactive GUI application for controlling the autonomous car in 3D simulation."""

    def __init__(self, backend: str = "sim", preset: str = "demo_90deg") -> None:
        pygame.init()
        pygame.display.set_caption("Los Podadores — Ackermann Car Teleop & 360° LiDAR")
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        self.clock = pygame.time.Clock()
        self.backend = backend
        self.preset = preset

        self.font_title = pygame.font.SysFont("monospace", 17, bold=True)
        self.font_body = pygame.font.SysFont("monospace", 13)
        self.font_small = pygame.font.SysFont("monospace", 11)

        sim_mode = "gazebo" if self.backend == "gazebo" else "mock"
        self.env = CarGazeboEnv(
            sim_mode=sim_mode,
            num_obstacles=4,
            road_width=7.0,
            allow_reverse=True,
        )
        reset_opts = {"preset": self.preset} if self.backend == "sim" else None
        self.obs, self.info = self.env.reset(options=reset_opts)

        self.steer_action: float = 0.0
        self.throttle_action: float = 0.0
        self.paused: bool = False
        self.lidar_render_mode: str = LIDAR_MODE_PHYSICAL_BEAMS
        self.total_reward: float = 0.0
        self.trail: list[tuple[float, float]] = []
        self.collision_flash_timer: int = 0
        if self.backend == "gazebo":
            self.status_msg = f"Gazebo 3D | Preset: {self.preset}"
        else:
            self.status_msg = f"Initialized ({self.preset}). Drive with WASD / Arrows."
        self.status_timer: int = 75

    def world_to_screen(
        self, wx: float, wy: float, car_x: float, car_y: float
    ) -> tuple[int, int]:
        """Convert world coordinates (meters) to screen pixel coordinates centered on car."""
        cx = VIEW_W / 2.0
        cy = WINDOW_H / 2.0
        sx = int(cx + (wx - car_x) * PIXELS_PER_METER)
        sy = int(cy - (wy - car_y) * PIXELS_PER_METER)
        return sx, sy

    def handle_input(self) -> bool:
        """Read keyboard state and update teleop action commands."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return False
                if event.key == pygame.K_l:
                    if self.lidar_render_mode == LIDAR_MODE_PHYSICAL_BEAMS:
                        self.lidar_render_mode = LIDAR_MODE_POLICY_GRID
                        mode_str = "RL Policy Bins (72-Ray Centered Grid)"
                    elif self.lidar_render_mode == LIDAR_MODE_POLICY_GRID:
                        self.lidar_render_mode = LIDAR_MODE_POINT_CLOUD
                        mode_str = "Contact Point Cloud (Hits Only)"
                    else:
                        self.lidar_render_mode = LIDAR_MODE_PHYSICAL_BEAMS
                        mode_str = "Physical LiDAR Beams (4 Pod Mounting Origins)"
                    self.status_msg = f"LiDAR: {mode_str}"
                    self.status_timer = 50
                if event.key == pygame.K_r:
                    self.obs, self.info = self.env.reset()
                    self.total_reward = 0.0
                    self.trail.clear()
                    if self.backend == "gazebo":
                        self.status_msg = "Teleported vehicle to start pose in Gazebo 3D."
                    else:
                        self.status_msg = "Reset vehicle and track."
                    self.status_timer = 50
                if event.key == pygame.K_p:
                    self.paused = not self.paused
                if event.key == pygame.K_n:
                    self.obs, self.info = self.env.reset()
                    self.trail.clear()
                    self.status_msg = "Generated new procedural track!"
                    self.status_timer = 60
                if event.key == pygame.K_e:
                    export_p = (
                        PROJECT_ROOT / "src" / "v2" / "model" / "procedural_track.sdf"
                    )
                    if self.env.track:
                        self.env.track.export_gazebo_sdf(export_p)
                        self.status_msg = f"Exported Gazebo SDF: {export_p.name}"
                        self.status_timer = 90
                if event.key == pygame.K_TAB:
                    # Manually cycle mission phase
                    self.env.mission_phase = (self.env.mission_phase + 1) % 3
                    phase_name = ["A -> B", "REMOTE HOLD", "B -> A"][
                        self.env.mission_phase
                    ]
                    self.status_msg = f"Switched mission phase: {phase_name}"
                    self.status_timer = 60

        keys = pygame.key.get_pressed()

        # Steering: Left/Right
        target_steer = 0.0
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            target_steer = 1.0  # Turn Left (+steer)
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            target_steer = -1.0  # Turn Right (-steer)

        if target_steer == 0.0:
            self.steer_action += (0.0 - self.steer_action) * 0.40
            if abs(self.steer_action) < 0.02:
                self.steer_action = 0.0
        else:
            self.steer_action += (target_steer - self.steer_action) * 0.35

        # Throttle/Brake/Reverse: Up/Down
        target_throttle = 0.0
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            target_throttle = 1.0  # Forward throttle
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            target_throttle = -1.0  # Reverse / Braking
        elif keys[pygame.K_SPACE]:
            target_throttle = -1.0
            self.steer_action = 0.0

        if target_throttle == 0.0:
            # Immediate zero throttle when no drive key is pressed
            self.throttle_action = 0.0
        else:
            self.throttle_action += (target_throttle - self.throttle_action) * 0.40
        return True

    def draw_world(self) -> None:
        """Render procedural road, red walls, obstacles, vehicle, and 360-degree LiDAR."""
        view_rect = pygame.Rect(0, 0, VIEW_W, WINDOW_H)
        pygame.draw.rect(self.screen, BG_COLOR, view_rect)

        car_x, car_y, car_yaw = self.env.car_x, self.env.car_y, self.env.car_yaw

        # 1. Background Grid
        grid_m = 5.0
        min_gx = int((car_x - 40.0) / grid_m) * int(grid_m)
        max_gx = int((car_x + 40.0) / grid_m) * int(grid_m)
        min_gy = int((car_y - 30.0) / grid_m) * int(grid_m)
        max_gy = int((car_y + 30.0) / grid_m) * int(grid_m)

        for gx in range(min_gx, max_gx + int(grid_m), int(grid_m)):
            p1 = self.world_to_screen(float(gx), float(min_gy), car_x, car_y)
            p2 = self.world_to_screen(float(gx), float(max_gy), car_x, car_y)
            pygame.draw.line(self.screen, GRID_COLOR, p1, p2, 1)

        for gy in range(min_gy, max_gy + int(grid_m), int(grid_m)):
            p1 = self.world_to_screen(float(min_gx), float(gy), car_x, car_y)
            p2 = self.world_to_screen(float(max_gx), float(gy), car_x, car_y)
            pygame.draw.line(self.screen, GRID_COLOR, p1, p2, 1)

        # 2. Draw Active Waypoint Path
        if self.env.track:
            wps = self.env._get_active_waypoints()
            if len(wps) > 1:
                pts = [self.world_to_screen(wp[0], wp[1], car_x, car_y) for wp in wps]
                pygame.draw.lines(self.screen, (55, 70, 90), False, pts, 4)

            # Draw Point A (Start) and Point B (Goal) beacons
            ax, ay, _ = self.env.track.start_pose
            bx, by, _ = self.env.track.goal_pose
            ap = self.world_to_screen(ax, ay, car_x, car_y)
            bp = self.world_to_screen(bx, by, car_x, car_y)
            pygame.draw.circle(self.screen, (40, 200, 90), ap, 10)
            pygame.draw.circle(self.screen, (30, 140, 240), bp, 10)
            self.screen.blit(
                self.font_small.render("Point A", True, TEXT_WHITE),
                (ap[0] + 12, ap[1] - 8),
            )
            self.screen.blit(
                self.font_small.render("Point B", True, TEXT_WHITE),
                (bp[0] + 12, bp[1] - 8),
            )

        # 3. Draw Flanking Red Walls
        if self.env.track:
            for (wx1, wy1), (wx2, wy2) in self.env.track.wall_segments:
                p1 = self.world_to_screen(wx1, wy1, car_x, car_y)
                p2 = self.world_to_screen(wx2, wy2, car_x, car_y)
                pygame.draw.line(self.screen, WALL_COLOR, p1, p2, 5)

        # 4. Draw Vehicle Breadcrumb Trail
        self.trail.append((car_x, car_y))
        if len(self.trail) > 400:
            self.trail.pop(0)
        if len(self.trail) > 1:
            trail_pts = [
                self.world_to_screen(tx, ty, car_x, car_y) for tx, ty in self.trail
            ]
            pygame.draw.lines(self.screen, (60, 130, 90), False, trail_pts, 2)

        # 5. Draw 360-degree LiDAR Returns (Physical Beams / Policy Bins / Point Cloud)
        cos_y = math.cos(car_yaw)
        sin_y = math.sin(car_yaw)

        if self.lidar_render_mode == LIDAR_MODE_PHYSICAL_BEAMS:
            # 5a. Physical LiDAR Beams: originating from the 4 actual hardware pod mounting locations!
            pod_beams = self.info.get("raw_pod_beams", None)
            if pod_beams is None or len(pod_beams) == 0:
                pod_beams = getattr(self.env, "latest_raw_pod_beams", [])

            for pod in pod_beams:
                x_pod, y_pod = pod["origin_ego"]
                origin_wx = car_x + x_pod * cos_y - y_pod * sin_y
                origin_wy = car_y + x_pod * sin_y + y_pod * cos_y
                origin_screen = self.world_to_screen(origin_wx, origin_wy, car_x, car_y)

                endpoints_ego = pod["endpoints_ego"]
                classes = pod["classes"]
                is_hit = pod["is_hit"]

                for (xe, ye), h_cls, hit in zip(endpoints_ego, classes, is_hit):
                    end_wx = car_x + xe * cos_y - ye * sin_y
                    end_wy = car_y + xe * sin_y + ye * cos_y
                    end_screen = self.world_to_screen(end_wx, end_wy, car_x, car_y)

                    if hit:
                        if h_cls == CLASS_VEHICLE:
                            color = OBS_VEHICLE_COLOR
                        elif h_cls == CLASS_PEDESTRIAN:
                            color = OBS_PED_COLOR
                        else:
                            color = WALL_COLOR
                        pygame.draw.line(self.screen, color, origin_screen, end_screen, 1)
                        pygame.draw.circle(self.screen, color, end_screen, 2)
                    else:
                        # Subtle unoccluded beam
                        pygame.draw.line(self.screen, (22, 38, 30), origin_screen, end_screen, 1)

        elif self.lidar_render_mode == LIDAR_MODE_POLICY_GRID:
            # 5b. Policy observation rays (72 ego-centroid spatial sectors around vehicle center)
            car_screen = self.world_to_screen(car_x, car_y, car_x, car_y)
            norm_dists = self.obs[12:84]
            hit_classes = self.obs[84:156]
            ray_angles = np.linspace(-math.pi, math.pi, LIDAR_NUM_RAYS, endpoint=False)
            for i in range(LIDAR_NUM_RAYS):
                r_dist = float(norm_dists[i]) * LIDAR_MAX_RANGE
                r_ang = float(ray_angles[i]) + car_yaw
                end_wx = car_x + r_dist * math.cos(r_ang)
                end_wy = car_y + r_dist * math.sin(r_ang)
                end_screen = self.world_to_screen(end_wx, end_wy, car_x, car_y)

                h_cls = int(hit_classes[i])
                if h_cls == CLASS_FREE:
                    color = (25, 45, 35)
                elif h_cls == CLASS_VEHICLE:
                    color = OBS_VEHICLE_COLOR
                elif h_cls == CLASS_PEDESTRIAN:
                    color = OBS_PED_COLOR
                else:
                    color = WALL_COLOR

                pygame.draw.line(self.screen, color, car_screen, end_screen, 1)

        else:
            # 5c. High-Resolution Contact Point Cloud Only
            raw_pts = self.info.get("raw_lidar_points_ego", None)
            if raw_pts is None or len(raw_pts) == 0:
                raw_pts = getattr(self.env, "latest_raw_lidar_points_ego", np.empty((0, 2)))

            if len(raw_pts) > 0:
                wx_pts = car_x + raw_pts[:, 0] * cos_y - raw_pts[:, 1] * sin_y
                wy_pts = car_y + raw_pts[:, 0] * sin_y + raw_pts[:, 1] * cos_y
                for wx, wy in zip(wx_pts, wy_pts):
                    sp = self.world_to_screen(wx, wy, car_x, car_y)
                    pygame.draw.circle(self.screen, (255, 95, 85), sp, 2)

        # 6. Draw Obstacles (Box/Cylinder, Static/Dynamic)
        if self.env.track:
            for obs in self.env.track.obstacles:
                if obs.shape == "box" and obs.length > 0.0 and obs.width > 0.0:
                    hl = obs.length / 2.0
                    hw = obs.width / 2.0
                    cos_o = math.cos(obs.yaw)
                    sin_o = math.sin(obs.yaw)
                    corners = [
                        (
                            obs.world_x + hl * cos_o - hw * sin_o,
                            obs.world_y + hl * sin_o + hw * cos_o,
                        ),
                        (
                            obs.world_x + hl * cos_o + hw * sin_o,
                            obs.world_y + hl * sin_o - hw * cos_o,
                        ),
                        (
                            obs.world_x - hl * cos_o + hw * sin_o,
                            obs.world_y - hl * sin_o - hw * cos_o,
                        ),
                        (
                            obs.world_x - hl * cos_o - hw * sin_o,
                            obs.world_y - hl * sin_o + hw * cos_o,
                        ),
                    ]
                    screen_pts = [
                        self.world_to_screen(cx, cy, car_x, car_y) for cx, cy in corners
                    ]
                    c = (
                        (190, 50, 40)
                        if obs.class_name == "Shipping Container"
                        else OBS_VEHICLE_COLOR
                    )
                    pygame.draw.polygon(self.screen, c, screen_pts)
                    pygame.draw.polygon(self.screen, TEXT_WHITE, screen_pts, 2)
                else:
                    op = self.world_to_screen(obs.world_x, obs.world_y, car_x, car_y)
                    r_px = max(4, int(obs.radius * PIXELS_PER_METER))
                    c = (
                        OBS_VEHICLE_COLOR
                        if obs.class_name == "Car"
                        else OBS_PED_COLOR
                        if obs.class_name == "Person"
                        else OBS_STATIC_COLOR
                    )
                    pygame.draw.circle(self.screen, c, op, r_px)
                    pygame.draw.circle(self.screen, TEXT_WHITE, op, r_px, 1)

                # Draw velocity arrow for dynamic obstacles
                if obs.is_dynamic:
                    op = self.world_to_screen(obs.world_x, obs.world_y, car_x, car_y)
                    tip_x = obs.world_x + obs.world_vx * 1.5
                    tip_y = obs.world_y + obs.world_vy * 1.5
                    tip_s = self.world_to_screen(tip_x, tip_y, car_x, car_y)
                    pygame.draw.line(self.screen, (255, 235, 60), op, tip_s, 3)

        # 7. Draw Scaled Vehicle (3.8m length, 1.9m width, Ackermann wheels)
        self._draw_vehicle(car_x, car_y, car_yaw, self.env.car_steer)

    def _draw_vehicle(
        self, car_x: float, car_y: float, car_yaw: float, steer: float
    ) -> None:
        """Render the 3.8m car body and steerable front wheels."""
        cos_y = math.cos(car_yaw)
        sin_y = math.sin(car_yaw)

        # 4 corners of the car body in ego frame (+X forward, +Y left)
        hl = CAR_LENGTH / 2.0  # 1.90m
        hw = CAR_WIDTH / 2.0  # 0.95m
        corners = [
            (+hl, +hw),
            (+hl, -hw),
            (-hl, -hw),
            (-hl, +hw),
        ]

        screen_corners: list[tuple[int, int]] = []
        for ex, ey in corners:
            wx = car_x + ex * cos_y - ey * sin_y
            wy = car_y + ex * sin_y + ey * cos_y
            screen_corners.append(self.world_to_screen(wx, wy, car_x, car_y))

        pygame.draw.polygon(self.screen, CAR_BODY_COLOR, screen_corners)
        pygame.draw.polygon(self.screen, (255, 170, 70), screen_corners, 2)

        # Draw Cabin / Hood contrast
        hood_front = [
            (+hl * 0.9, +hw * 0.7),
            (+hl * 0.9, -hw * 0.7),
            (+0.2, -hw * 0.7),
            (+0.2, +hw * 0.7),
        ]
        hood_pts = []
        for ex, ey in hood_front:
            wx = car_x + ex * cos_y - ey * sin_y
            wy = car_y + ex * sin_y + ey * cos_y
            hood_pts.append(self.world_to_screen(wx, wy, car_x, car_y))
        pygame.draw.polygon(self.screen, CAR_CABIN_COLOR, hood_pts)

        # Draw 4 Wheels
        wheel_l = 0.76  # 2 * 0.38m radius
        wheel_w = 0.28
        axle_f = WHEELBASE / 2.0  # +1.25m
        axle_r = -WHEELBASE / 2.0  # -1.25m
        track_h = 0.80  # +/- 0.80m

        wheel_configs = [
            (axle_f, +track_h, steer),  # Front Left (Steerable)
            (axle_f, -track_h, steer),  # Front Right (Steerable)
            (axle_r, +track_h, 0.0),  # Rear Left (Fixed)
            (axle_r, -track_h, 0.0),  # Rear Right (Fixed)
        ]

        for wx_ego, wy_ego, w_steer in wheel_configs:
            total_steer = car_yaw + w_steer
            cos_w = math.cos(total_steer)
            sin_w = math.sin(total_steer)

            # Center of wheel in world
            wc_x = car_x + wx_ego * cos_y - wy_ego * sin_y
            wc_y = car_y + wx_ego * sin_y + wy_ego * cos_y

            w_pts = [
                (+wheel_l / 2, +wheel_w / 2),
                (+wheel_l / 2, -wheel_w / 2),
                (-wheel_l / 2, -wheel_w / 2),
                (-wheel_l / 2, +wheel_w / 2),
            ]
            scr_w_pts = []
            for px, py in w_pts:
                p_wx = wc_x + px * cos_w - py * sin_w
                p_wy = wc_y + px * sin_w + py * cos_w
                scr_w_pts.append(self.world_to_screen(p_wx, p_wy, car_x, car_y))

            pygame.draw.polygon(self.screen, TIRE_COLOR, scr_w_pts)
            pygame.draw.polygon(self.screen, (80, 80, 90), scr_w_pts, 1)

        # Draw Sensor Pod Indicators (Exact physical CAD mounting locations)
        pod_offsets = [
            (FRONT_LIDAR_X, 0.0, (50, 220, 100), "F"),   # Front bumper
            (REAR_LIDAR_X, 0.0, (220, 50, 50), "R"),    # Rear hitch
            (+0.50, -0.70, (50, 180, 240), "R"), # Right flank
            (+0.50, +0.70, (50, 180, 240), "L"), # Left flank
        ]
        for px, py, col, _ in pod_offsets:
            p_wx = car_x + px * cos_y - py * sin_y
            p_wy = car_y + px * sin_y + py * cos_y
            p_scr = self.world_to_screen(p_wx, p_wy, car_x, car_y)
            pygame.draw.circle(self.screen, (15, 20, 26), p_scr, 6)
            pygame.draw.circle(self.screen, col, p_scr, 4)
            pygame.draw.circle(self.screen, (255, 255, 255), p_scr, 1)

        # 7. Collision Alert Visual Overlay
        if self.info.get("collision") or self.collision_flash_timer > 0:
            pygame.draw.rect(
                self.screen, (220, 45, 45), (0, 0, VIEW_W, WINDOW_H), 8
            )
            warn_lbl = self.font_title.render("COLLISION DETECTED!", True, (255, 60, 60))
            self.screen.blit(warn_lbl, (VIEW_W // 2 - warn_lbl.get_width() // 2, 24))

    def draw_hud(self) -> None:
        """Render side panel with instrumentation, 360-degree radar minimap, and controls."""
        panel_rect = pygame.Rect(VIEW_W, 0, PANEL_W, WINDOW_H)
        pygame.draw.rect(self.screen, PANEL_BG, panel_rect)
        pygame.draw.line(self.screen, (50, 60, 75), (VIEW_W, 0), (VIEW_W, WINDOW_H), 2)

        x = VIEW_W + 24
        y = 20

        # Title & Backend
        title = self.font_title.render("AUTONOMOUS CAR TELEOP", True, TEXT_CYAN)
        self.screen.blit(title, (x, y))
        y += 24

        if self.backend == "gazebo":
            backend_lbl = self.font_small.render(
                "BACKEND: GAZEBO 3D (ROS 2 BRIDGE) | ACKERMANN", True, TEXT_GREEN
            )
            n_pods = self.info.get("active_pods", 0)
            lidar_sub_lbl = self.font_small.render(
                f"LIDAR: ROS 2 LIVE ({n_pods} Pods / 660 Beams, 20Hz)",
                True,
                TEXT_CYAN,
            )
        else:
            backend_lbl = self.font_small.render(
                f"BACKEND: {self.backend.upper()} | NON-HOLONOMIC ACKERMANN",
                True,
                TEXT_MUTED,
            )
            lidar_sub_lbl = self.font_small.render(
                "LIDAR: MOCK ANALYTICAL RAYCASTER (72 Rays)",
                True,
                TEXT_MUTED,
            )
        self.screen.blit(backend_lbl, (x, y))
        y += 18
        self.screen.blit(lidar_sub_lbl, (x, y))
        y += 24

        # Mission Phase Header
        phase_idx = self.env.mission_phase
        phase_names = [
            "PHASE 0: NAV A -> B",
            "PHASE 1: REMOTE HOLD (POINT B)",
            "PHASE 2: NAV B -> A",
        ]
        phase_colors = [TEXT_GREEN, TEXT_YELLOW, TEXT_CYAN]
        phase_text = self.font_body.render(
            phase_names[phase_idx], True, phase_colors[phase_idx]
        )
        pygame.draw.rect(
            self.screen, (35, 45, 60), (x - 6, y - 4, PANEL_W - 36, 26), border_radius=4
        )
        self.screen.blit(phase_text, (x, y))
        y += 36

        # Telemetry Gauges
        speed = self.env.car_vx
        steer_deg = math.degrees(self.env.car_steer)
        goal_dist = self.info.get("dist_to_goal", 0.0)

        self._draw_gauge(
            "SPEED (m/s):",
            speed,
            MIN_SPEED_MPS,
            MAX_SPEED_MPS,
            x,
            y,
            TEXT_GREEN if speed >= 0 else TEXT_YELLOW,
        )
        y += 50
        self._draw_gauge("STEER (deg):", steer_deg, -32.0, 32.0, x, y, TEXT_CYAN)
        y += 50

        # Metrics Table
        is_colliding = bool(self.info.get("collision")) or (self.collision_flash_timer > 0)
        metrics = [
            f"Distance to Target: {goal_dist:5.1f} m",
            f"Heading:            {math.degrees(self.env.car_yaw):+5.1f}°",
            f"Yaw Rate:           {math.degrees(self.env.car_yaw_rate):+5.1f}°/s",
            f"Linear Accel (ax):  {self.env.car_ax:+5.2f} m/s²",
            f"Lateral Slip (vy):  {self.env.car_vy:+5.2f} m/s",
            f"Knuckle Steer:      {steer_deg:+5.1f}°",
            f"Collision Status:   {'COLLISION!' if is_colliding else 'CLEAR'}",
            f"Simulation Steps:   {self.env.step_count:d}",
        ]
        for m in metrics:
            color = TEXT_RED if "COLLISION!" in m else TEXT_WHITE
            lbl = self.font_body.render(m, True, color)
            self.screen.blit(lbl, (x, y))
            y += 20
        y += 6

        # 360-Degree Surround Radar Minimap
        radar_cx = x + (PANEL_W - 48) // 2
        radar_cy = y + 75
        radar_rad = 65
        pygame.draw.circle(self.screen, (15, 20, 26), (radar_cx, radar_cy), radar_rad)
        pygame.draw.circle(
            self.screen, (40, 50, 65), (radar_cx, radar_cy), radar_rad, 1
        )
        pygame.draw.circle(
            self.screen, (30, 40, 52), (radar_cx, radar_cy), radar_rad // 2, 1
        )
        pygame.draw.line(
            self.screen,
            (45, 55, 70),
            (radar_cx - radar_rad, radar_cy),
            (radar_cx + radar_rad, radar_cy),
            1,
        )
        pygame.draw.line(
            self.screen,
            (45, 55, 70),
            (radar_cx, radar_cy - radar_rad),
            (radar_cx, radar_cy + radar_rad),
            1,
        )

        # Draw car icon in center of radar
        pygame.draw.rect(
            self.screen,
            CAR_BODY_COLOR,
            (radar_cx - 4, radar_cy - 8, 8, 16),
            border_radius=2,
        )

        # Plot 360-degree LiDAR returns on radar
        norm_dists = self.obs[12:84]
        hit_classes = self.obs[84:156]
        ray_angles = np.linspace(-math.pi, math.pi, LIDAR_NUM_RAYS, endpoint=False)

        raw_pts = self.info.get("raw_lidar_points_ego", None)
        if raw_pts is None or len(raw_pts) == 0:
            raw_pts = getattr(self.env, "latest_raw_lidar_points_ego", np.empty((0, 2)))

        if len(raw_pts) > 0 and self.lidar_render_mode != LIDAR_MODE_POLICY_GRID:
            dists = np.hypot(raw_pts[:, 0], raw_pts[:, 1])
            valid = dists <= LIDAR_MAX_RANGE
            d_norm = dists[valid] / LIDAR_MAX_RANGE
            ego_ang = np.arctan2(raw_pts[valid, 1], raw_pts[valid, 0])
            radar_ang = ego_ang - math.pi / 2.0
            for d_r, ang in zip(d_norm, radar_ang):
                rx = radar_cx + int(d_r * radar_rad * math.cos(ang))
                ry = radar_cy + int(d_r * radar_rad * math.sin(ang))
                pygame.draw.circle(self.screen, (255, 90, 80), (rx, ry), 1)
        else:
            for i in range(LIDAR_NUM_RAYS):
                d_ratio = float(norm_dists[i])
                if d_ratio < 0.98:
                    ang = float(ray_angles[i]) - math.pi / 2.0  # +X forward is up on radar
                    rx = radar_cx + int(d_ratio * radar_rad * math.cos(ang))
                    ry = radar_cy + int(d_ratio * radar_rad * math.sin(ang))
                    h_cls = int(hit_classes[i])
                    col = (
                        OBS_VEHICLE_COLOR
                        if h_cls == CLASS_VEHICLE
                        else OBS_PED_COLOR
                        if h_cls == CLASS_PEDESTRIAN
                        else WALL_COLOR
                    )
                    pygame.draw.circle(self.screen, col, (rx, ry), 2)

        lidar_src = self.info.get("lidar_source", "mock")
        if lidar_src == "ros_gz":
            n_hits = len(raw_pts) if raw_pts is not None else 0
            n_pods = self.info.get("active_pods", 4)
            radar_lbl = self.font_small.render(
                f"ROS 2 LiDAR ({n_pods}p, {n_hits}pts)", True, TEXT_CYAN
            )
        else:
            radar_lbl = self.font_small.render("360° LIDAR (4 Pods, 15m)", True, TEXT_MUTED)
        self.screen.blit(radar_lbl, (radar_cx - 65, radar_cy + radar_rad + 6))

        # Mode Indicator
        if self.lidar_render_mode == LIDAR_MODE_PHYSICAL_BEAMS:
            mode_badge = "VIEW: PHYSICAL BEAMS (4-Pod Chassis Origins)"
            mode_col = TEXT_CYAN
        elif self.lidar_render_mode == LIDAR_MODE_POLICY_GRID:
            mode_badge = "VIEW: RL POLICY BINS (72-Ray Centered Grid)"
            mode_col = TEXT_YELLOW
        else:
            mode_badge = "VIEW: CONTACT POINT CLOUD (660 Hits)"
            mode_col = (255, 120, 100)
        mode_txt = self.font_small.render(mode_badge, True, mode_col)
        self.screen.blit(mode_txt, (x, radar_cy + radar_rad + 24))
        y += 185

        # Status Message Banner
        if self.status_timer > 0:
            self.status_timer -= 1
            msg_box = pygame.Rect(x - 6, y, PANEL_W - 36, 26)
            pygame.draw.rect(self.screen, (25, 40, 35), msg_box, border_radius=4)
            self.screen.blit(
                self.font_small.render(self.status_msg, True, TEXT_GREEN), (x, y + 5)
            )
        y += 35

        # Controls Guide
        controls = [
            "KEYBOARD CONTROLS:",
            "  [W / ↑]  Accelerate (0 -> 3.0 m/s)",
            "  [S / ↓]  Brake / Reverse (down to -1.2)",
            "  [A / D]  Steer Left / Right (Ackermann)",
            "  [Space]  Handbrake (instant stop)",
            "  [Tab]    Cycle Phase (A->B, Hold, B->A)",
            "  [L]      LiDAR View: Physical / Policy / Cloud",
            "  [N]      Generate New Procedural Track",
            "  [E]      Export Track to Gazebo 3D SDF",
            "  [R]      Reset Vehicle & Track",
            "  [P]      Pause Simulation",
            "  [Esc]    Quit",
        ]
        for c in controls:
            color = TEXT_YELLOW if "KEYBOARD" in c else TEXT_MUTED
            lbl = self.font_small.render(c, True, color)
            self.screen.blit(lbl, (x, y))
            y += 18

    def _draw_gauge(
        self,
        label: str,
        val: float,
        min_v: float,
        max_v: float,
        x: int,
        y: int,
        color: tuple[int, int, int],
    ) -> None:
        """Draw an analog gauge bar with center zero."""
        lbl = self.font_body.render(f"{label} {val:+4.2f}", True, TEXT_MUTED)
        self.screen.blit(lbl, (x, y))

        bar_w = 340
        bar_h = 14
        bar_y = y + 20
        pygame.draw.rect(
            self.screen, (15, 20, 26), (x, bar_y, bar_w, bar_h), border_radius=3
        )
        pygame.draw.line(
            self.screen,
            (80, 95, 115),
            (x + bar_w // 2, bar_y - 2),
            (x + bar_w // 2, bar_y + bar_h + 2),
            2,
        )

        norm_val = np.clip(val, min_v, max_v)
        center_x = x + bar_w // 2
        fill_w = int((norm_val / (max_v - min_v) * 2.0) * (bar_w // 2))

        if fill_w > 0:
            pygame.draw.rect(
                self.screen, color, (center_x, bar_y + 2, fill_w, bar_h - 4)
            )
        elif fill_w < 0:
            pygame.draw.rect(
                self.screen, color, (center_x + fill_w, bar_y + 2, -fill_w, bar_h - 4)
            )

    def run(self) -> None:
        """Main loop: handle inputs, step physics, and render."""
        running = True
        while running:
            running = self.handle_input()

            if not self.paused:
                action = np.array(
                    [self.steer_action, self.throttle_action], dtype=np.float32
                )
                self.obs, reward, terminated, truncated, self.info = self.env.step(
                    action
                )
                self.total_reward += reward

                if self.info.get("collision"):
                    self.collision_flash_timer = 45  # 1.5s visual latch at 30 FPS
                    self.status_msg = "CRASH DETECTED! Collision registered."
                    self.status_timer = 60
                elif self.collision_flash_timer > 0:
                    self.collision_flash_timer -= 1

                if terminated or truncated:
                    was_collision = bool(self.info.get("collision"))
                    self.obs, self.info = self.env.reset()
                    if was_collision:
                        self.info["collision"] = True
                        self.collision_flash_timer = 45
                    self.total_reward = 0.0
                    self.trail.clear()
                    if self.backend == "gazebo":
                        self.status_msg = "Episode reset. Vehicle returned to start in Gazebo."
                    else:
                        self.status_msg = "Episode reset. Vehicle returned to start."
                    self.status_timer = 60

            self.draw_world()
            self.draw_hud()

            pygame.display.flip()
            self.clock.tick(FPS)

        self.env.close()
        pygame.quit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Car Teleop & 360 perception viewer")
    parser.add_argument(
        "--backend",
        choices=["sim", "gazebo"],
        default="sim",
        help="Backend simulation engine",
    )
    parser.add_argument(
        "--preset",
        choices=["demo_90deg", "multi_zone_plaza", "random"],
        default="demo_90deg",
        help="Procedural track layout preset",
    )
    args = parser.parse_args()

    app = CarTeleopApp(backend=args.backend, preset=args.preset)
    app.run()


if __name__ == "__main__":
    main()
