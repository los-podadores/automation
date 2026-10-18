"""Headless 2D Pygame renderer for CarGazeboEnv.

Renders high-fidelity top-down RGB frames containing procedural road corridors,
active waypoint trajectories, walls, static/dynamic obstacles, steerable Ackermann
vehicle footprint, 360-degree LiDAR returns, and telemetry overlays.
"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING, Final

import numpy as np

if TYPE_CHECKING:
    from .car_gazebo_env import CarGazeboEnv

# Ensure headless execution
os.environ["SDL_VIDEODRIVER"] = "dummy"

import pygame  # noqa: E402

from src.translate.config import (  # noqa: E402
    CAR_LENGTH,
    CAR_WIDTH,
    FRONT_TRACK_WIDTH,
    FRONT_WHEEL_RADIUS,
    LIDAR_MAX_RANGE,
    LIDAR_NUM_RAYS,
    REAR_TRACK_WIDTH,
    REAR_WHEEL_RADIUS,
    VEHICLE_SIZE_SCALE,
    WHEELBASE,
)
from src.translate.fusion_types import CLASS_FREE  # noqa: E402

# Colors
BG_COLOR: Final[tuple[int, int, int]] = (18, 22, 28)
ROAD_COLOR: Final[tuple[int, int, int]] = (36, 44, 56)
WALL_COLOR: Final[tuple[int, int, int]] = (220, 55, 55)
GRID_COLOR: Final[tuple[int, int, int]] = (30, 38, 48)
TEXT_WHITE: Final[tuple[int, int, int]] = (240, 245, 250)
TEXT_GREEN: Final[tuple[int, int, int]] = (70, 220, 120)
TEXT_RED: Final[tuple[int, int, int]] = (240, 70, 70)
TEXT_CYAN: Final[tuple[int, int, int]] = (60, 200, 240)
TEXT_YELLOW: Final[tuple[int, int, int]] = (245, 200, 50)
CAR_BODY_COLOR: Final[tuple[int, int, int]] = (235, 130, 30)
CAR_CABIN_COLOR: Final[tuple[int, int, int]] = (195, 100, 20)
TIRE_COLOR: Final[tuple[int, int, int]] = (20, 20, 24)
OBS_VEHICLE_COLOR: Final[tuple[int, int, int]] = (60, 140, 240)
OBS_PED_COLOR: Final[tuple[int, int, int]] = (245, 170, 40)
OBS_STATIC_COLOR: Final[tuple[int, int, int]] = (235, 75, 75)


class CarEnvRenderer:
    """Renders top-down RGB frames of CarGazeboEnv state for evaluation videos."""

    def __init__(
        self,
        width: int = 880,
        height: int = 640,
        pixels_per_meter: float = 14.0,
    ) -> None:
        self.width = width
        self.height = height
        self.ppm = pixels_per_meter

        pygame.init()
        self.surface = pygame.Surface((self.width, self.height))
        self.font_title = pygame.font.SysFont("monospace", 15, bold=True)
        self.font_body = pygame.font.SysFont("monospace", 12)
        self.trail: list[tuple[float, float]] = []

    def world_to_screen(
        self, wx: float, wy: float, car_x: float, car_y: float
    ) -> tuple[int, int]:
        cx = self.width / 2.0
        cy = self.height / 2.0
        sx = int(cx + (wx - car_x) * self.ppm)
        sy = int(cy - (wy - car_y) * self.ppm)
        return sx, sy

    def reset_trail(self) -> None:
        self.trail.clear()

    def render_frame(self, env: CarGazeboEnv) -> np.ndarray:
        """Render a single frame and return as (height, width, 3) uint8 numpy array."""
        self.surface.fill(BG_COLOR)
        car_x, car_y, car_yaw = env.car_x, env.car_y, env.car_yaw

        # 1. Background Grid
        grid_m = 5.0
        min_gx = int((car_x - 35.0) / grid_m) * int(grid_m)
        max_gx = int((car_x + 35.0) / grid_m) * int(grid_m)
        min_gy = int((car_y - 25.0) / grid_m) * int(grid_m)
        max_gy = int((car_y + 25.0) / grid_m) * int(grid_m)

        for gx in range(min_gx, max_gx + int(grid_m), int(grid_m)):
            p1 = self.world_to_screen(float(gx), float(min_gy), car_x, car_y)
            p2 = self.world_to_screen(float(gx), float(max_gy), car_x, car_y)
            pygame.draw.line(self.surface, GRID_COLOR, p1, p2, 1)

        for gy in range(min_gy, max_gy + int(grid_m), int(grid_m)):
            p1 = self.world_to_screen(float(min_gx), float(gy), car_x, car_y)
            p2 = self.world_to_screen(float(max_gx), float(gy), car_x, car_y)
            pygame.draw.line(self.surface, GRID_COLOR, p1, p2, 1)

        # 2. Waypoints & Goal/Start Beacons
        if env.track:
            wps = env._get_active_waypoints()
            if len(wps) > 1:
                pts = [self.world_to_screen(wp[0], wp[1], car_x, car_y) for wp in wps]
                pygame.draw.lines(self.surface, (55, 75, 95), False, pts, 4)

            ax, ay, _ = env.track.start_pose
            bx, by, _ = env.track.goal_pose
            ap = self.world_to_screen(ax, ay, car_x, car_y)
            bp = self.world_to_screen(bx, by, car_x, car_y)
            pygame.draw.circle(self.surface, (40, 200, 90), ap, 8)
            pygame.draw.circle(self.surface, (30, 140, 240), bp, 8)

        # 3. Flanking Corridor Walls
        if env.track:
            for (wx1, wy1), (wx2, wy2) in env.track.wall_segments:
                p1 = self.world_to_screen(wx1, wy1, car_x, car_y)
                p2 = self.world_to_screen(wx2, wy2, car_x, car_y)
                pygame.draw.line(self.surface, WALL_COLOR, p1, p2, 5)

        # 4. Trail
        self.trail.append((car_x, car_y))
        if len(self.trail) > 300:
            self.trail.pop(0)
        if len(self.trail) > 1:
            t_pts = [self.world_to_screen(tx, ty, car_x, car_y) for tx, ty in self.trail]
            pygame.draw.lines(self.surface, (60, 140, 95), False, t_pts, 2)

        # 5. Obstacles
        if env.track:
            for obs in env.track.obstacles:
                if obs.shape == "box" and obs.length > 0.0 and obs.width > 0.0:
                    hl = obs.length / 2.0
                    hw = obs.width / 2.0
                    cos_o = math.cos(obs.yaw)
                    sin_o = math.sin(obs.yaw)
                    corners = [
                        (obs.world_x + hl * cos_o - hw * sin_o, obs.world_y + hl * sin_o + hw * cos_o),
                        (obs.world_x + hl * cos_o + hw * sin_o, obs.world_y + hl * sin_o - hw * cos_o),
                        (obs.world_x - hl * cos_o + hw * sin_o, obs.world_y - hl * sin_o - hw * cos_o),
                        (obs.world_x - hl * cos_o - hw * sin_o, obs.world_y - hl * sin_o + hw * cos_o),
                    ]
                    s_pts = [self.world_to_screen(cx, cy, car_x, car_y) for cx, cy in corners]
                    pygame.draw.polygon(self.surface, OBS_VEHICLE_COLOR, s_pts)
                    pygame.draw.polygon(self.surface, TEXT_WHITE, s_pts, 2)
                else:
                    op = self.world_to_screen(obs.world_x, obs.world_y, car_x, car_y)
                    r_px = max(4, int(obs.radius * self.ppm))
                    c = OBS_PED_COLOR if obs.class_name == "Person" else OBS_STATIC_COLOR
                    pygame.draw.circle(self.surface, c, op, r_px)
                    pygame.draw.circle(self.surface, TEXT_WHITE, op, r_px, 1)

        # 6. Segmented 3D LiDAR Cloud (front+rear, viz only)
        cos_y = math.cos(car_yaw)
        sin_y = math.sin(car_yaw)
        raw_pts = env.latest_raw_lidar_points_ego
        for xe, ye in np.asarray(raw_pts).reshape(-1, 2):
            end_wx = car_x + xe * cos_y - ye * sin_y
            end_wy = car_y + xe * sin_y + ye * cos_y
            end_s = self.world_to_screen(end_wx, end_wy, car_x, car_y)
            pygame.draw.circle(self.surface, WALL_COLOR, end_s, 2)

        # 7. Scaled Ackermann Vehicle
        self._draw_vehicle(env, car_x, car_y, car_yaw)

        # 8. HUD Telemetry Banner
        self._draw_hud_banner(env)

        # Convert surface to (H, W, 3) numpy array
        raw_arr = pygame.surfarray.array3d(self.surface)
        return np.transpose(raw_arr, (1, 0, 2)).astype(np.uint8)

    def _draw_vehicle(
        self, env: CarGazeboEnv, car_x: float, car_y: float, car_yaw: float
    ) -> None:
        cos_y = math.cos(car_yaw)
        sin_y = math.sin(car_yaw)
        hl = CAR_LENGTH / 2.0
        hw = CAR_WIDTH / 2.0
        corners = [(+hl, +hw), (+hl, -hw), (-hl, -hw), (-hl, +hw)]
        screen_corners = [
            self.world_to_screen(car_x + ex * cos_y - ey * sin_y, car_y + ex * sin_y + ey * cos_y, car_x, car_y)
            for ex, ey in corners
        ]
        pygame.draw.polygon(self.surface, CAR_BODY_COLOR, screen_corners)
        pygame.draw.polygon(self.surface, (255, 175, 75), screen_corners, 2)

        # Wheels (scaled with vehicle: per-axle diameter x width)
        axle_f, axle_r = WHEELBASE / 2.0, -WHEELBASE / 2.0
        track_f = FRONT_TRACK_WIDTH / 2.0
        track_r = REAR_TRACK_WIDTH / 2.0
        wheel_configs = [
            (axle_f, +track_f, env.car_steer, 2.0 * FRONT_WHEEL_RADIUS, 0.16 * VEHICLE_SIZE_SCALE),
            (axle_f, -track_f, env.car_steer, 2.0 * FRONT_WHEEL_RADIUS, 0.16 * VEHICLE_SIZE_SCALE),
            (axle_r, +track_r, 0.0, 2.0 * REAR_WHEEL_RADIUS, 0.27 * VEHICLE_SIZE_SCALE),
            (axle_r, -track_r, 0.0, 2.0 * REAR_WHEEL_RADIUS, 0.27 * VEHICLE_SIZE_SCALE),
        ]
        for wx_ego, wy_ego, w_steer, wheel_l, wheel_w in wheel_configs:
            total_steer = car_yaw + w_steer
            cos_w = math.cos(total_steer)
            sin_w = math.sin(total_steer)
            wc_x = car_x + wx_ego * cos_y - wy_ego * sin_y
            wc_y = car_y + wx_ego * sin_y + wy_ego * cos_y
            w_pts = [
                (+wheel_l / 2, +wheel_w / 2),
                (+wheel_l / 2, -wheel_w / 2),
                (-wheel_l / 2, -wheel_w / 2),
                (-wheel_l / 2, +wheel_w / 2),
            ]
            scr_w_pts = [
                self.world_to_screen(wc_x + px * cos_w - py * sin_w, wc_y + px * sin_w + py * cos_w, car_x, car_y)
                for px, py in w_pts
            ]
            pygame.draw.polygon(self.surface, TIRE_COLOR, scr_w_pts)

    def _draw_hud_banner(self, env: CarGazeboEnv) -> None:
        banner_rect = pygame.Rect(10, 10, self.width - 20, 32)
        pygame.draw.rect(self.surface, (25, 32, 42), banner_rect, border_radius=4)
        pygame.draw.rect(self.surface, (45, 55, 70), banner_rect, 1, border_radius=4)

        phase_names = ["A -> B", "HOLD", "B -> A"]
        phase_str = phase_names[env.mission_phase] if env.mission_phase < len(phase_names) else "?"
        goal_st = env._compute_goal_state()

        text = (
            f"Step: {env.step_count:3d} | "
            f"Phase: {phase_str} | "
            f"Speed: {env.car_vx:+4.2f} m/s | "
            f"Steer: {math.degrees(env.car_steer):+4.1f}° | "
            f"Dist: {goal_st.distance:5.1f}m | "
            f"CTE: {goal_st.cross_track_error:+4.2f}m"
        )
        lbl = self.font_body.render(text, True, TEXT_WHITE)
        self.surface.blit(lbl, (20, 18))
