"""Teleoperation and debug viewer for RobotCoverageEnv (pygame-based)."""

from __future__ import annotations

import math
import time
from collections import deque

import cv2
import numpy as np
import pygame
from robot_env import (
    CELLS_MISSED_THRESHOLD,
    MAP_SIZE,
    NUM_MAPS,
    REWARD_BASE_PENALTY,
    ROBOT_RADIUS_PX,
    RobotCoverageEnv,
)

ENV_SIZE: int = 700
PANEL_W: int = 660
WINDOW_W: int = ENV_SIZE + PANEL_W
WINDOW_H: int = 980
TELEOP_ROT_SCALE: float = 0.1

SENSOR_NAMES: list[str] = [
    "ray_front_left",
    "ray_front_center",
    "ray_front_right",
    "ray_side_left",
    "ray_side_right",
    "ray_rear",
    "last_v",
    "last_w",
    "homing_dist",
    "homing_cos",
    "homing_sin",
]

RAY_LABELS: list[str] = ["FL", "FC", "FR", "SL", "SR", "RR"]
RAY_COLORS: list[tuple[int, int, int]] = [
    (255, 165, 0),
    (0, 255, 200),
    (255, 165, 0),
    (100, 100, 255),
    (100, 100, 255),
    (200, 200, 200),
]
MAP_NAMES: list[str] = ["Coverage", "Obstacles", "Frontier"]
MAP_COLORS: list[tuple[int, int, int]] = [
    (100, 220, 100),
    (220, 80, 80),
    (220, 120, 255),
]

FPS_SMOOTHING: float = 0.95
SWATCH_SIZE: int = 64


# ------------------------------------------------------------------
# Drawing helpers
# ------------------------------------------------------------------


def draw_text(
    surface: pygame.Surface,
    text: str,
    x: int,
    y: int,
    font: pygame.font.Font,
    color: tuple[int, int, int] = (220, 220, 220),
) -> None:
    surface.blit(font.render(text, True, color), (x, y))


def draw_bar(
    surface: pygame.Surface,
    x: int,
    y: int,
    w: int,
    h: int,
    value: float,
    max_val: float,
    color: tuple[int, int, int],
    bg: tuple[int, int, int] = (50, 50, 50),
) -> None:
    pygame.draw.rect(surface, bg, (x, y, w, h))
    fill_w = max(0, min(w, int(w * value / max_val)))
    pygame.draw.rect(surface, color, (x, y, fill_w, h))
    pygame.draw.rect(surface, (100, 100, 100), (x, y, w, h), 1)


def draw_obs_maps(
    surface: pygame.Surface,
    obs: dict,
    panel_x: int,
    y: int,
    font: pygame.font.Font,
) -> int:
    """Draw the multi-scale observation maps.  Returns the updated *y*."""
    draw_text(
        surface,
        f"MULTI-SCALE MAPS ({NUM_MAPS} scales):",
        panel_x + 10,
        y,
        font,
        (180, 200, 255),
    )
    y += 18

    keys = ["coverage", "obstacles", "frontier"]
    map_px = 120
    col_gap = 12
    row_gap = 6

    for scale in range(NUM_MAPS):
        draw_text(surface, f"S{scale}:", panel_x + 10, y, font, (150, 150, 150))
        for i, (key, name, color) in enumerate(
            zip(keys, MAP_NAMES, MAP_COLORS, strict=True)
        ):
            cx = panel_x + 40 + i * (map_px + col_gap)
            if scale == 0:
                draw_text(surface, name, cx, y - 10, font, color)
            data = obs[key][scale]
            img = np.zeros((MAP_SIZE, MAP_SIZE, 3), dtype=np.uint8)
            mask = data > 0.01
            intensity = np.clip(data, 0, 1)
            for c in range(3):
                img[:, :, c] = (intensity * color[c]).astype(np.uint8)
            img[~mask] = 0
            img = cv2.resize(img, (map_px, map_px), interpolation=cv2.INTER_NEAREST)
            img = np.transpose(img, (1, 0, 2))
            surf = pygame.surfarray.make_surface(img)
            pygame.draw.rect(surf, (80, 80, 80), (0, 0, map_px, map_px), 1)
            surface.blit(surf, (cx, y))
        y += map_px + row_gap

    return y + 10


def draw_footprint_inset(
    surface: pygame.Surface,
    env: RobotCoverageEnv,
    panel_x: int,
    y: int,
    font: pygame.font.Font,
    inset_px: int = SWATCH_SIZE,
) -> int:
    """Draw the local robot footprint inset.  Returns the updated *y*."""
    draw_text(
        surface, "ROBOT FOOTPRINT (local):", panel_x + 10, y, font, (180, 200, 255)
    )
    y += 18
    footprint = env._draw_robot_footprint_local(
        env.agent_pos_m, env.agent_heading, local_size=inset_px
    )
    img = np.zeros((inset_px, inset_px, 3), dtype=np.uint8)
    img[footprint > 0] = [50, 50, 200]
    img = np.transpose(img, (1, 0, 2))
    surf = pygame.surfarray.make_surface(img)
    pygame.draw.rect(surf, (80, 80, 80), (0, 0, inset_px, inset_px), 1)
    surface.blit(surf, (panel_x + 10, y))
    draw_text(
        surface,
        f"{inset_px}x{inset_px}px  center=grid_px",
        panel_x + inset_px + 16,
        y + inset_px // 2 - 6,
        font,
        (150, 150, 150),
    )
    return y + inset_px + 12


# ------------------------------------------------------------------
# Panel sections (extracted from the monolithic main loop)
# ------------------------------------------------------------------


def _draw_header(
    screen: pygame.Surface,
    env: RobotCoverageEnv,
    panel_x: int,
    y: int,
    fps_smooth: float,
    avg_dt: float,
    font: pygame.font.Font,
) -> int:
    draw_text(
        screen,
        f"Phase: {env.phase}  Step: {env.current_step}  "
        f"FPS: {fps_smooth:.0f}  step: {avg_dt * 1000:.2f}ms",
        panel_x + 10,
        y,
        font,
        (255, 220, 100),
    )
    return y + 22


def _draw_help(
    screen: pygame.Surface, panel_x: int, y: int, font: pygame.font.Font
) -> int:
    draw_text(screen, "CONTROLS:", panel_x + 10, y, font, (180, 200, 255))
    y += 18
    for line in [
        " W/S: throttle fwd/back  A/D: steer",
        " R/C: reset  1-8: phase  SPACE: pause",
        " H: help   T: auto   ESC: quit",
        " D: collision dil  B: stamp bbox",
        " F: footprint    G: ray pixels",
        " V: coverable field",
    ]:
        draw_text(screen, line, panel_x + 10, y, font, (160, 160, 160))
        y += 15
    return y


def _draw_sensors(
    screen: pygame.Surface,
    obs: dict,
    panel_x: int,
    y: int,
    font: pygame.font.Font,
) -> int:
    sensors = obs["sensors"]
    draw_text(screen, "SENSORS:", panel_x + 10, y, font, (180, 200, 255))
    y += 18

    for i, _name in enumerate(SENSOR_NAMES[:6]):
        val = sensors[i]
        color = RAY_COLORS[i]
        draw_text(
            screen, f" {RAY_LABELS[i]:>2s}: {val:.3f}", panel_x + 10, y, font, color
        )
        draw_bar(screen, panel_x + 110, y + 2, 100, 10, val, 1.0, color)
        y += 16

    draw_text(
        screen, f" last_v:  {sensors[6]:.3f}", panel_x + 10, y, font, (200, 200, 200)
    )
    y += 16
    draw_text(
        screen, f" last_w:  {sensors[7]:.3f}", panel_x + 10, y, font, (200, 200, 200)
    )
    y += 16

    # Homing beacon values
    draw_text(
        screen, f" homing_d: {sensors[8]:.3f}", panel_x + 10, y, font, (180, 200, 255)
    )
    y += 16
    draw_text(
        screen, f" homing_cos: {sensors[9]:.3f}", panel_x + 10, y, font, (180, 200, 255)
    )
    y += 16
    draw_text(
        screen,
        f" homing_sin: {sensors[10]:.3f}",
        panel_x + 10,
        y,
        font,
        (180, 200, 255),
    )
    y += 22
    return y


def _draw_coverage(
    screen: pygame.Surface,
    env: RobotCoverageEnv,
    panel_x: int,
    y: int,
    font: pygame.font.Font,
) -> int:
    draw_text(screen, "COVERAGE:", panel_x + 10, y, font, (180, 200, 255))
    y += 18
    pct = env.coverage_in_percent * 100
    draw_text(
        screen,
        f" {env.coverage_in_pixels} / {env.total_cells} ({pct:.1f}%)",
        panel_x + 10,
        y,
        font,
        (100, 220, 100),
    )
    y += 15
    draw_text(
        screen,
        f" collisions: {env.num_collisions}",
        panel_x + 10,
        y,
        font,
        (200, 80, 80),
    )
    y += 15
    bar_w = PANEL_W - 20
    draw_bar(screen, panel_x + 10, y, bar_w, 12, pct, 100, (100, 220, 100))
    draw_text(screen, f"{pct:.1f}%", panel_x + bar_w + 14, y - 1, font, (255, 255, 255))
    y += 20

    cells_missed = env.total_cells - env.coverage_in_pixels
    draw_text(
        screen,
        f" cells missed: {cells_missed} (threshold: {CELLS_MISSED_THRESHOLD})",
        panel_x + 10,
        y,
        font,
        (200, 200, 200),
    )
    y += 22
    return y


def _draw_state(
    screen: pygame.Surface,
    env: RobotCoverageEnv,
    panel_x: int,
    y: int,
    font: pygame.font.Font,
) -> int:
    draw_text(screen, "STATE:", panel_x + 10, y, font, (180, 200, 255))
    y += 18
    gx_px, gy_px = env._m_to_grid_px(env.agent_pos_m)
    draw_text(
        screen,
        f" pos: ({env.agent_pos_m[0]:.2f}, {env.agent_pos_m[1]:.2f})",
        panel_x + 10,
        y,
        font,
        (200, 200, 200),
    )
    y += 15
    draw_text(
        screen, f" grid_px: ({gx_px}, {gy_px})", panel_x + 10, y, font, (150, 180, 255)
    )
    y += 15
    draw_text(
        screen,
        f" heading: {math.degrees(env.agent_heading):.1f} deg",
        panel_x + 10,
        y,
        font,
        (200, 200, 200),
    )
    y += 15
    draw_text(
        screen,
        f" grid: {env.grid_size_p}x{env.grid_size_p}px ({env.grid_size_m:.1f}m)",
        panel_x + 10,
        y,
        font,
        (200, 200, 200),
    )
    y += 15
    dil_k = ROBOT_RADIUS_PX
    draw_text(
        screen,
        f" collision dil: {dil_k * 2 + 1}x{dil_k * 2 + 1}",
        panel_x + 10,
        y,
        font,
        (150, 150, 150),
    )
    y += 22
    return y


def _draw_reward(
    screen: pygame.Surface,
    reward: float,
    panel_x: int,
    y: int,
    font: pygame.font.Font,
) -> int:
    draw_text(screen, "REWARD:", panel_x + 10, y, font, (180, 200, 255))
    y += 18
    draw_text(screen, f" last: {reward:+.4f}", panel_x + 10, y, font, (255, 255, 180))
    y += 15
    draw_text(
        screen,
        f" base: {REWARD_BASE_PENALTY:+.4f}",
        panel_x + 10,
        y,
        font,
        (255, 120, 120),
    )
    y += 22
    return y


def _draw_overlays(
    screen: pygame.Surface,
    toggles: dict[str, bool],
    panel_x: int,
    y: int,
    font: pygame.font.Font,
) -> int:
    draw_text(screen, "OVERLAYS:", panel_x + 10, y, font, (180, 200, 255))
    y += 18
    for key, label in [
        ("dilated", "D: collision dil"),
        ("stamped", "B: stamp bbox"),
        ("footprint", "F: local footprint"),
        ("rays", "G: ray pixel steps"),
        ("coverable", "V: coverable field"),
    ]:
        state = "ON" if toggles[key] else "off"
        color = (100, 220, 100) if toggles[key] else (100, 100, 100)
        draw_text(screen, f" {label}: {state}", panel_x + 10, y, font, color)
        y += 15
    return y + 10


def _draw_status(
    screen: pygame.Surface,
    paused: bool,
    auto_sample: bool,
    panel_x: int,
    font: pygame.font.Font,
) -> None:
    status = []
    if paused:
        status.append("PAUSED")
    if auto_sample:
        status.append("AUTO")
    if status:
        draw_text(
            screen, " ".join(status), panel_x + PANEL_W - 120, 8, font, (255, 100, 100)
        )


# ------------------------------------------------------------------
# Event handling
# ------------------------------------------------------------------


def _handle_event(
    event: pygame.event.Event,
    env: RobotCoverageEnv,
    toggles: dict[str, bool],
) -> tuple[bool, bool, bool, bool]:
    """Process a single pygame event.

    Returns ``(running, paused, auto_sample, show_help)`` updated flags.
    """
    paused = False
    auto_sample = False
    show_help = False
    running = True

    if event.type == pygame.QUIT:
        return False, False, False, False

    if event.type != pygame.KEYDOWN:
        return True, False, False, False

    if event.key == pygame.K_ESCAPE:
        return False, False, False, False
    elif event.key == pygame.K_r:
        env.reset()
    elif event.key == pygame.K_SPACE:
        paused = True  # caller toggles
    elif event.key == pygame.K_t:
        auto_sample = True  # caller toggles
    elif event.key == pygame.K_h:
        show_help = True  # caller toggles
    elif event.key == pygame.K_d:
        toggles["dilated"] = not toggles["dilated"]
    elif event.key == pygame.K_b:
        toggles["stamped"] = not toggles["stamped"]
    elif event.key == pygame.K_f:
        toggles["footprint"] = not toggles["footprint"]
    elif event.key == pygame.K_g:
        toggles["rays"] = not toggles["rays"]
    elif event.key == pygame.K_v:
        toggles["coverable"] = not toggles["coverable"]
    elif event.key in (
        pygame.K_1,
        pygame.K_2,
        pygame.K_3,
        pygame.K_4,
        pygame.K_5,
        pygame.K_6,
        pygame.K_7,
        pygame.K_8,
    ):
        env.set_phase(int(chr(event.key)))
        env.reset()
    elif event.key == pygame.K_c:
        env.reset()

    return running, paused, auto_sample, show_help


# ------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------


def main() -> None:
    """Run the teleoperation / debug viewer."""
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Robot Teleop & Debug Viewer v3")
    clock = pygame.time.Clock()
    font_sm = pygame.font.SysFont("monospace", 12)
    font_lg = pygame.font.SysFont("monospace", 16, bold=True)

    env = RobotCoverageEnv(render_mode="rgb_array", phase=1)
    env.window_size = ENV_SIZE
    obs, info = env.reset()

    auto_sample = False
    paused = False
    show_help = True
    toggles: dict[str, bool] = {
        "dilated": False,
        "stamped": False,
        "footprint": False,
        "rays": False,
        "coverable": False,
    }

    fps_smooth = 0.0
    step_times: deque[float] = deque(maxlen=60)
    running = True

    while running:
        for event in pygame.event.get():
            r, p, a, h = _handle_event(event, env, toggles)
            if not r:
                running = False
            if p:
                paused = not paused
            if a:
                auto_sample = not auto_sample
            if h:
                show_help = not show_help

        keys = pygame.key.get_pressed()
        if not auto_sample:
            throttle = 0.0
            steering = 0.0
            if keys[pygame.K_w] or keys[pygame.K_UP]:
                throttle = 1.0
            if keys[pygame.K_s] or keys[pygame.K_DOWN]:
                throttle = -1.0
            if keys[pygame.K_a] or keys[pygame.K_LEFT]:
                steering = 1.0
            if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                steering = -1.0
            steering *= TELEOP_ROT_SCALE
            action = np.array([throttle, steering], dtype=np.float32)
        else:
            action = env.action_space.sample()

        t0 = time.perf_counter()
        if not paused:
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()
        dt = time.perf_counter() - t0
        step_times.append(dt)
        avg_dt = sum(step_times) / len(step_times)
        instant_fps = 1.0 / dt if dt > 0 else 0
        fps_smooth = FPS_SMOOTHING * fps_smooth + (1 - FPS_SMOOTHING) * instant_fps

        # --- draw ---
        screen.fill((20, 20, 20))
        field_img = env.render(toggles=toggles)
        field_surf = pygame.surfarray.make_surface(np.transpose(field_img, (1, 0, 2)))
        screen.blit(field_surf, (0, 0))

        panel_x = ENV_SIZE
        pygame.draw.rect(screen, (35, 35, 40), (panel_x, 0, PANEL_W, WINDOW_H))

        y = 8
        y = _draw_header(screen, env, panel_x, y, fps_smooth, avg_dt, font_lg)
        if show_help:
            y = _draw_help(screen, panel_x, y, font_sm)
        y = _draw_sensors(screen, obs, panel_x, y, font_sm)
        y = _draw_coverage(screen, env, panel_x, y, font_sm)
        y = draw_obs_maps(screen, obs, panel_x, y, font_sm)
        if toggles["footprint"]:
            y = draw_footprint_inset(screen, env, panel_x, y, font_sm)
        y = _draw_state(screen, env, panel_x, y, font_sm)
        y = _draw_reward(screen, reward, panel_x, y, font_sm)
        y = _draw_overlays(screen, toggles, panel_x, y, font_sm)
        _draw_status(screen, paused, auto_sample, panel_x, font_lg)

        pygame.display.flip()
        clock.tick(60)

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
