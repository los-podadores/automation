"""Interactive pygame-based teleoperation and perception viewer for CarGazeboEnv."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Final

# Sanitize sys.path to prevent foreign python version site-packages (e.g. Nix ROS python3.14 leaking into Python 3.12)
_curr_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
sys.path = [p for p in sys.path if "/lib/python" not in p or f"/lib/{_curr_ver}" in p]

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pygame

from src.sim.car_gazebo_env import CarGazeboEnv  # noqa: E402
from src.translate.fusion_types import (  # noqa: E402
    CLASS_FREE,
    CLASS_PEDESTRIAN,
    CLASS_STATIC_HAZARD,
    CLASS_VEHICLE,
)

# Window Configuration
WINDOW_W: Final[int] = 1200
WINDOW_H: Final[int] = 800
VIEW_W: Final[int] = 800
PANEL_W: Final[int] = 400
FPS: Final[int] = 30

# Colors (Dark Modern Theme)
BG_COLOR: Final[tuple[int, int, int]] = (20, 24, 30)
PANEL_BG: Final[tuple[int, int, int]] = (28, 34, 44)
ROAD_COLOR: Final[tuple[int, int, int]] = (40, 48, 60)
LANE_COLOR: Final[tuple[int, int, int]] = (240, 200, 50)
CURB_COLOR: Final[tuple[int, int, int]] = (180, 50, 50)
GRID_COLOR: Final[tuple[int, int, int]] = (35, 42, 54)
TEXT_WHITE: Final[tuple[int, int, int]] = (240, 245, 250)
TEXT_MUTED: Final[tuple[int, int, int]] = (140, 155, 175)
TEXT_GREEN: Final[tuple[int, int, int]] = (70, 220, 120)
TEXT_RED: Final[tuple[int, int, int]] = (240, 70, 70)
TEXT_CYAN: Final[tuple[int, int, int]] = (60, 200, 240)

# Entity Colors
CAR_BODY_COLOR: Final[tuple[int, int, int]] = (45, 160, 85)
CAR_HOOD_COLOR: Final[tuple[int, int, int]] = (35, 130, 70)
TIRE_COLOR: Final[tuple[int, int, int]] = (15, 15, 18)
OBS_VEHICLE_COLOR: Final[tuple[int, int, int]] = (60, 140, 240)
OBS_PED_COLOR: Final[tuple[int, int, int]] = (245, 170, 40)
OBS_STATIC_COLOR: Final[tuple[int, int, int]] = (235, 75, 75)

# Viewport parameters
PIXELS_PER_METER: Final[float] = 16.0  # 1 meter = 16 pixels


class CarTeleopApp:
    """Interactive GUI application for controlling the autonomous car."""

    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("Los Podadores — Car Teleoperation & Perception Viewer")
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        self.clock = pygame.time.Clock()

        self.font_title = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_body = pygame.font.SysFont("monospace", 14)
        self.font_small = pygame.font.SysFont("monospace", 12)

        self.env = CarGazeboEnv(num_obstacles=6)
        self.obs, self.info = self.env.reset(seed=123)

        self.steer_action: float = 0.0
        self.throttle_action: float = 0.0
        self.paused: bool = False
        self.total_reward: float = 0.0
        self.trail: list[tuple[float, float]] = []

    def world_to_screen(self, wx: float, wy: float, car_x: float) -> tuple[int, int]:
        """Convert world coordinates (meters) to screen pixel coordinates."""
        # Camera tracks vehicle along X with forward offset
        cam_x = car_x - 4.0
        screen_x = int((wx - cam_x) * PIXELS_PER_METER + 50)
        # Y is lateral: +Y left, so screen_y is inverted
        screen_y = int((WINDOW_H / 2.0) - (wy * PIXELS_PER_METER))
        return screen_x, screen_y

    def handle_input(self) -> bool:
        """Read keyboard state and update teleop action commands."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return False
                if event.key == pygame.K_r:
                    self.obs, self.info = self.env.reset()
                    self.total_reward = 0.0
                    self.trail.clear()
                if event.key == pygame.K_p:
                    self.paused = not self.paused

        keys = pygame.key.get_pressed()

        # Steering: Left/Right
        target_steer = 0.0
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            target_steer = 1.0  # positive steering turns left (+Y)
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            target_steer = -1.0  # negative steering turns right (-Y)

        # Smooth steering transition
        self.steer_action += (target_steer - self.steer_action) * 0.25

        # Throttle/Brake: Up/Down
        target_throttle = 0.0
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            target_throttle = 1.0
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            target_throttle = -1.0
        elif keys[pygame.K_SPACE]:
            target_throttle = -1.0
            target_steer = 0.0

        # Smooth throttle transition
        self.throttle_action += (target_throttle - self.throttle_action) * 0.30

        return True

    def draw_world(self) -> None:
        """Render road, obstacles, vehicle, and perception rays on the left panel."""
        view_rect = pygame.Rect(0, 0, VIEW_W, WINDOW_H)
        pygame.draw.rect(self.screen, BG_COLOR, view_rect)

        car_x = self.env.car_x
        car_y = self.env.car_y
        car_yaw = self.env.car_yaw

        # 1. Draw Road and Lane Boundaries
        road_half_w = 4.0  # 4 meters each side
        top_left = self.world_to_screen(car_x - 10.0, road_half_w, car_x)
        bot_right = self.world_to_screen(car_x + 50.0, -road_half_w, car_x)
        road_h = abs(bot_right[1] - top_left[1])
        pygame.draw.rect(self.screen, ROAD_COLOR, (0, top_left[1], VIEW_W, road_h))

        # Lane markings (dashed center line)
        dash_len_m = 2.0
        gap_len_m = 2.0
        curr_x = math.floor((car_x - 10.0) / 4.0) * 4.0
        while curr_x < car_x + 50.0:
            p1 = self.world_to_screen(curr_x, 0.0, car_x)
            p2 = self.world_to_screen(curr_x + dash_len_m, 0.0, car_x)
            if 0 <= p1[0] <= VIEW_W or 0 <= p2[0] <= VIEW_W:
                pygame.draw.line(self.screen, LANE_COLOR, p1, p2, 2)
            curr_x += dash_len_m + gap_len_m

        # Curbs
        p_c1_a = self.world_to_screen(car_x - 10.0, road_half_w, car_x)
        p_c1_b = self.world_to_screen(car_x + 50.0, road_half_w, car_x)
        pygame.draw.line(self.screen, CURB_COLOR, p_c1_a, p_c1_b, 4)

        p_c2_a = self.world_to_screen(car_x - 10.0, -road_half_w, car_x)
        p_c2_b = self.world_to_screen(car_x + 50.0, -road_half_w, car_x)
        pygame.draw.line(self.screen, CURB_COLOR, p_c2_a, p_c2_b, 4)

        # 2. Draw 72 Radial Perception Rays (Observed by the Policy)
        num_rays = 72
        dist_slice = self.obs[10 : 10 + num_rays]
        cls_slice = self.obs[10 + num_rays : 10 + 2 * num_rays]
        ray_angles = np.linspace(-math.pi / 2.0, math.pi / 2.0, num_rays)
        car_pos_px = self.world_to_screen(car_x, car_y, car_x)

        for i in range(0, num_rays, 2):  # Draw every 2nd ray to keep view clean
            ang_world = car_yaw + ray_angles[i]
            r_m = float(dist_slice[i]) * 25.0
            ray_end_w = (car_x + r_m * math.cos(ang_world), car_y + r_m * math.sin(ang_world))
            ray_end_px = self.world_to_screen(ray_end_w[0], ray_end_w[1], car_x)

            cls_code = int(cls_slice[i])
            if cls_code == CLASS_FREE:
                r_color = (60, 75, 95)
            elif cls_code == CLASS_VEHICLE:
                r_color = (80, 160, 240)
            elif cls_code == CLASS_PEDESTRIAN:
                r_color = (240, 180, 50)
            else:
                r_color = (240, 80, 80)

            pygame.draw.line(self.screen, r_color, car_pos_px, ray_end_px, 1)

        # 3. Draw Trajectory Trail
        self.trail.append((car_x, car_y))
        if len(self.trail) > 150:
            self.trail.pop(0)
        if len(self.trail) >= 2:
            trail_pts = [self.world_to_screen(tx, ty, car_x) for tx, ty in self.trail]
            pygame.draw.lines(self.screen, (70, 160, 100), False, trail_pts, 2)

        # 4. Draw Simulated Obstacles
        for ent in self.env.entities:
            epx, epy = self.world_to_screen(ent.world_x, ent.world_y, car_x)
            r_px = max(4, int(ent.radius * PIXELS_PER_METER))

            if ent.class_name in ("Car", "Bus"):
                color = OBS_VEHICLE_COLOR
                # Draw vehicle box
                w_px = int(ent.radius * 2.0 * PIXELS_PER_METER)
                h_px = int(ent.radius * 1.2 * PIXELS_PER_METER)
                pygame.draw.rect(
                    self.screen,
                    color,
                    (epx - w_px // 2, epy - h_px // 2, w_px, h_px),
                    border_radius=4,
                )
            elif ent.class_name == "Person":
                color = OBS_PED_COLOR
                pygame.draw.circle(self.screen, color, (epx, epy), r_px)
            else:
                color = OBS_STATIC_COLOR
                pygame.draw.circle(self.screen, color, (epx, epy), r_px)

            # Label
            lbl = self.font_small.render(ent.class_name, True, TEXT_MUTED)
            self.screen.blit(lbl, (epx - lbl.get_width() // 2, epy - r_px - 14))

        # 5. Draw Ego Tractor Vehicle
        self._draw_tractor(car_x, car_y, car_yaw)

    def _draw_tractor(self, cx: float, cy: float, yaw: float) -> None:
        """Render the tractor with articulated front steering wheels."""
        c_px, c_py = self.world_to_screen(cx, cy, cx)

        # Chassis oriented polygon
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)

        def rot(dx: float, dy: float) -> tuple[int, int]:
            # dy is lateral (+left), so invert for screen coordinates
            sx = c_px + int((dx * cos_y - dy * sin_y) * PIXELS_PER_METER)
            sy = c_py - int((dx * sin_y + dy * cos_y) * PIXELS_PER_METER)
            return sx, sy

        # Chassis Box
        corners = [
            rot(0.54, 0.28),
            rot(0.54, -0.28),
            rot(-0.54, -0.28),
            rot(-0.54, 0.28),
        ]
        pygame.draw.polygon(self.screen, CAR_BODY_COLOR, corners)
        pygame.draw.polygon(self.screen, TEXT_WHITE, corners, 1)

        # Hood accent
        hood = [
            rot(0.54, 0.18),
            rot(0.54, -0.18),
            rot(0.10, -0.20),
            rot(0.10, 0.20),
        ]
        pygame.draw.polygon(self.screen, CAR_HOOD_COLOR, hood)

        # Rear drive wheels (Fixed orientation to chassis)
        wheel_w_rear = 0.25
        wheel_l_rear = 0.50
        for side_y in (0.32, -0.32):
            w_pts = [
                rot(-0.40 + wheel_l_rear / 2.0, side_y + wheel_w_rear / 2.0),
                rot(-0.40 + wheel_l_rear / 2.0, side_y - wheel_w_rear / 2.0),
                rot(-0.40 - wheel_l_rear / 2.0, side_y - wheel_w_rear / 2.0),
                rot(-0.40 - wheel_l_rear / 2.0, side_y + wheel_w_rear / 2.0),
            ]
            pygame.draw.polygon(self.screen, TIRE_COLOR, w_pts)

        # Front steering wheels (Articulated by steer angle delta)
        steer = self.env.car_steer
        steer_yaw = yaw + steer
        cos_s = math.cos(steer_yaw)
        sin_s = math.sin(steer_yaw)
        wheel_w_front = 0.16
        wheel_l_front = 0.37

        for side_y in (0.30, -0.30):
            hub_px, hub_py = rot(0.40, side_y)

            def rot_wheel(lx: float, ly: float) -> tuple[int, int]:
                wx = hub_px + int((lx * cos_s - ly * sin_s) * PIXELS_PER_METER)
                wy = hub_py - int((lx * sin_s + ly * cos_s) * PIXELS_PER_METER)
                return wx, wy

            fw_pts = [
                rot_wheel(wheel_l_front / 2.0, wheel_w_front / 2.0),
                rot_wheel(wheel_l_front / 2.0, -wheel_w_front / 2.0),
                rot_wheel(-wheel_l_front / 2.0, -wheel_w_front / 2.0),
                rot_wheel(-wheel_l_front / 2.0, wheel_w_front / 2.0),
            ]
            pygame.draw.polygon(self.screen, TIRE_COLOR, fw_pts)

        # 3 Forward-Biased Sensor Pods (Option B)
        # Pod 0: Center (Yellow)
        pygame.draw.circle(self.screen, (240, 220, 50), rot(0.55, 0.0), 4)
        # Pod 1: Left +45° (Cyan)
        pygame.draw.circle(self.screen, (60, 220, 240), rot(0.45, 0.30), 4)
        # Pod 2: Right -45° (Cyan)
        pygame.draw.circle(self.screen, (60, 220, 240), rot(0.45, -0.30), 4)

    def draw_hud(self) -> None:
        """Render live telemetry, gauges, and perception HUD on the right panel."""
        panel_rect = pygame.Rect(VIEW_W, 0, PANEL_W, WINDOW_H)
        pygame.draw.rect(self.screen, PANEL_BG, panel_rect)
        pygame.draw.line(self.screen, GRID_COLOR, (VIEW_W, 0), (VIEW_W, WINDOW_H), 2)

        x_pad = VIEW_W + 20
        y = 20

        # Title
        title = self.font_title.render("TELEMETRY & SENSORS", True, TEXT_WHITE)
        self.screen.blit(title, (x_pad, y))
        y += 35

        # Vehicle Kinematics
        speed_kmh = self.env.car_vx * 3.6
        steer_deg = math.degrees(self.env.car_steer)
        lat_err = self.env.car_y

        stats = [
            ("Speed:", f"{self.env.car_vx:4.2f} m/s ({speed_kmh:4.1f} km/h)", TEXT_CYAN),
            ("Steer Angle:", f"{steer_deg:+5.1f}°", TEXT_GREEN if abs(steer_deg) < 15 else TEXT_RED),
            ("Yaw Rate:", f"{self.env.car_yaw_rate:+5.2f} rad/s", TEXT_WHITE),
            ("Acceleration:", f"{self.env.car_ax:+5.2f} m/s²", TEXT_WHITE),
            ("Lateral Error:", f"{lat_err:+5.2f} m", TEXT_GREEN if abs(lat_err) < 1.0 else TEXT_RED),
            ("Heading Error:", f"{math.degrees(self.env.car_yaw):+5.1f}°", TEXT_WHITE),
            ("Episode Reward:", f"{self.total_reward:+6.1f}", TEXT_GREEN if self.total_reward > 0 else TEXT_RED),
            ("Step Count:", f"{self.env.step_count:4d} / 500", TEXT_WHITE),
        ]

        for label, val, color in stats:
            lbl_surf = self.font_body.render(f"{label:<16}", True, TEXT_MUTED)
            val_surf = self.font_body.render(val, True, color)
            self.screen.blit(lbl_surf, (x_pad, y))
            self.screen.blit(val_surf, (x_pad + 150, y))
            y += 24

        y += 15
        pygame.draw.line(self.screen, GRID_COLOR, (x_pad, y), (WINDOW_W - 20, y), 1)
        y += 15

        # Action Gauges
        gauge_lbl = self.font_title.render("ACTIVE CONTROLS", True, TEXT_WHITE)
        self.screen.blit(gauge_lbl, (x_pad, y))
        y += 30

        # Steering bar (-1 to +1)
        self._draw_bar(x_pad, y, "Steering (A / D):", self.steer_action, -1.0, 1.0, (80, 180, 240))
        y += 50
        # Throttle/Brake bar (-1 to +1)
        self._draw_bar(x_pad, y, "Throttle (W / S):", self.throttle_action, -1.0, 1.0, (80, 220, 120))
        y += 65

        pygame.draw.line(self.screen, GRID_COLOR, (x_pad, y), (WINDOW_W - 20, y), 1)
        y += 15

        # Perception Mini-Radar View
        radar_title = self.font_title.render("POLAR PERCEPTION SCAN", True, TEXT_WHITE)
        self.screen.blit(radar_title, (x_pad, y))
        y += 30

        radar_cx = x_pad + 180
        radar_cy = y + 100
        radar_r = 80

        # Draw radar arc
        pygame.draw.arc(
            self.screen,
            GRID_COLOR,
            (radar_cx - radar_r, radar_cy - radar_r, radar_r * 2, radar_r * 2),
            0,
            math.pi,
            2,
        )
        pygame.draw.line(self.screen, GRID_COLOR, (radar_cx - radar_r, radar_cy), (radar_cx + radar_r, radar_cy), 1)
        pygame.draw.line(self.screen, GRID_COLOR, (radar_cx, radar_cy), (radar_cx, radar_cy - radar_r), 1)

        # Plot 72 rays on radar
        num_rays = 72
        dist_slice = self.obs[10 : 10 + num_rays]
        cls_slice = self.obs[10 + num_rays : 10 + 2 * num_rays]
        angles = np.linspace(-math.pi / 2.0, math.pi / 2.0, num_rays)

        for i in range(num_rays):
            r_norm = float(dist_slice[i])
            if r_norm < 0.98:
                ang = angles[i] + math.pi / 2.0
                px = radar_cx + int(r_norm * radar_r * math.cos(ang))
                py = radar_cy - int(r_norm * radar_r * math.sin(ang))
                cls_c = int(cls_slice[i])
                if cls_c == CLASS_VEHICLE:
                    c = OBS_VEHICLE_COLOR
                elif cls_c == CLASS_PEDESTRIAN:
                    c = OBS_PED_COLOR
                elif cls_c == CLASS_STATIC_HAZARD:
                    c = OBS_STATIC_COLOR
                else:
                    c = TEXT_MUTED
                pygame.draw.circle(self.screen, c, (px, py), 3)

        # Draw vehicle icon at radar origin
        pygame.draw.circle(self.screen, (240, 240, 240), (radar_cx, radar_cy), 5)
        y += 190

        # Help Instructions
        help_lines = [
            "[W / UP]    Throttle forward",
            "[S / DOWN]  Brake / Reverse",
            "[A / LEFT]  Steer Left",
            "[D / RIGHT] Steer Right",
            "[SPACE]     Handbrake",
            "[R]         Reset environment",
            "[P]         Pause simulation",
            "[Q / ESC]   Quit",
        ]
        for line in help_lines:
            txt = self.font_small.render(line, True, TEXT_MUTED)
            self.screen.blit(txt, (x_pad, y))
            y += 18

    def _draw_bar(
        self,
        x: int,
        y: int,
        label: str,
        val: float,
        min_v: float,
        max_v: float,
        color: tuple[int, int, int],
    ) -> None:
        """Draw an analog gauge bar with center zero."""
        lbl = self.font_body.render(f"{label} {val:+4.2f}", True, TEXT_MUTED)
        self.screen.blit(lbl, (x, y))

        bar_w = 340
        bar_h = 14
        bar_y = y + 20
        pygame.draw.rect(self.screen, (15, 20, 26), (x, bar_y, bar_w, bar_h), border_radius=3)
        pygame.draw.line(self.screen, (80, 95, 115), (x + bar_w // 2, bar_y - 2), (x + bar_w // 2, bar_y + bar_h + 2), 2)

        norm_val = np.clip(val, min_v, max_v)
        center_x = x + bar_w // 2
        fill_w = int((norm_val / (max_v - min_v) * 2.0) * (bar_w // 2))

        if fill_w > 0:
            pygame.draw.rect(self.screen, color, (center_x, bar_y + 2, fill_w, bar_h - 4))
        elif fill_w < 0:
            pygame.draw.rect(self.screen, color, (center_x + fill_w, bar_y + 2, -fill_w, bar_h - 4))

    def run(self) -> None:
        """Main loop: handle inputs, step physics, and render."""
        running = True
        while running:
            running = self.handle_input()

            if not self.paused:
                action = np.array([self.steer_action, self.throttle_action], dtype=np.float32)
                self.obs, reward, terminated, truncated, self.info = self.env.step(action)
                self.total_reward += reward

                if terminated or truncated:
                    self.obs, self.info = self.env.reset()
                    self.total_reward = 0.0
                    self.trail.clear()

            self.draw_world()
            self.draw_hud()

            pygame.display.flip()
            self.clock.tick(FPS)

        pygame.quit()


def main() -> None:
    app = CarTeleopApp()
    app.run()


if __name__ == "__main__":
    main()
