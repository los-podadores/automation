import math

import cv2
import numpy as np
import pygame
from robot_env import (
    MAP_SIZE,
    PHASES,
    REWARD_BASE_PENALTY,
    RobotCoverageEnv,
)

ENV_SIZE = 700
PANEL_W = 660
WINDOW_W = ENV_SIZE + PANEL_W
WINDOW_H = 980

SENSOR_NAMES = [
    "ray_front_left",
    "ray_front_center",
    "ray_front_right",
    "ray_side_left",
    "ray_side_right",
    "ray_rear",
    "last_v",
    "last_w",
]

RAY_LABELS = ["FL", "FC", "FR", "SL", "SR", "RR"]
RAY_COLORS = [
    (255, 165, 0),
    (0, 255, 200),
    (255, 165, 0),
    (100, 100, 255),
    (100, 100, 255),
    (200, 200, 200),
]
MAP_NAMES = ["Coverage", "Obstacles", "Frontier"]
MAP_COLORS = [(100, 220, 100), (220, 80, 80), (220, 120, 255)]


def draw_text(surface, text, x, y, font, color=(220, 220, 220)):
    surface.blit(font.render(text, True, color), (x, y))


def draw_bar(surface, x, y, w, h, value, max_val, color, bg=(50, 50, 50)):
    pygame.draw.rect(surface, bg, (x, y, w, h))
    fill_w = max(0, min(w, int(w * value / max_val)))
    pygame.draw.rect(surface, color, (x, y, fill_w, h))
    pygame.draw.rect(surface, (100, 100, 100), (x, y, w, h), 1)


def draw_env_view(surface, env, x0, y0):
    canvas = pygame.Surface((ENV_SIZE, ENV_SIZE))
    canvas.fill((30, 30, 30))

    if env.field is None:
        surface.blit(canvas, (x0, y0))
        return

    pad = 5.0
    minx, miny, maxx, maxy = env.field.bounds
    width = (maxx - minx) + 2 * pad
    height = (maxy - miny) + 2 * pad
    scl = ENV_SIZE / max(width, height)
    off = np.array([minx - pad, miny - pad])

    def to_screen(wx, wy):
        px = int((wx - off[0]) * scl)
        py = int(ENV_SIZE - (wy - off[1]) * scl)
        return px, py

    img = np.full((env.grid_size_p, env.grid_size_p, 3), 30, dtype=np.uint8)
    field_mask = env.field_grid > 0
    img[field_mask] = [220, 220, 220]
    cov_mask = env.coverage_map > 0
    img[cov_mask] = [80, 160, 80]
    obs_mask = env.obstacle_map > 0
    img[obs_mask] = [200, 80, 80]

    img = cv2.resize(img, (ENV_SIZE, ENV_SIZE), interpolation=cv2.INTER_NEAREST)
    img = cv2.cvtColor(img[::-1], cv2.COLOR_BGR2RGB)
    screen_arr = np.transpose(img, (1, 0, 2))
    surf = pygame.surfarray.make_surface(screen_arr)
    canvas.blit(surf, (0, 0))

    ext_points = [to_screen(x, y) for x, y in env.field.exterior.coords]
    pygame.draw.polygon(canvas, (0, 0, 0), ext_points, 2)
    for interior in env.field.interiors:
        in_points = [to_screen(x, y) for x, y in interior.coords]
        pygame.draw.polygon(canvas, (255, 255, 255), in_points)
        pygame.draw.polygon(canvas, (255, 0, 0), in_points, 1)

    corners = env._get_square_corners(env.agent_pos_m, env.agent_heading)
    corner_pg = [to_screen(c[0], c[1]) for c in corners]
    pygame.draw.polygon(canvas, (50, 50, 200), corner_pg)

    hx = env.agent_pos_m[0] + 0.5 * math.cos(env.agent_heading)
    hy = env.agent_pos_m[1] + 0.5 * math.sin(env.agent_heading)
    pygame.draw.line(
        canvas, (0, 255, 0), to_screen(*env.agent_pos_m), to_screen(hx, hy), 2
    )

    a = 1.0
    theta = env.agent_heading
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    def local_pt(lx, ly):
        return (
            env.agent_pos_m[0] + lx * cos_t - ly * sin_t,
            env.agent_pos_m[1] + lx * sin_t + ly * cos_t,
        )

    origins = [
        local_pt(a / 2, a / 2),
        local_pt(a / 2, 0),
        local_pt(a / 2, -a / 2),
        local_pt(0, a / 2),
        local_pt(0, -a / 2),
        local_pt(-a / 2, 0),
    ]
    ray_angles = [math.pi / 4, 0.0, -math.pi / 4, math.pi / 2, -math.pi / 2, math.pi]
    sensors, _ = env._compute_sensors()

    for i, (origin, ang_off) in enumerate(zip(origins, ray_angles)):
        ang = theta + ang_off
        dist = sensors[i]
        end = (
            origin[0] + math.cos(ang) * dist,
            origin[1] + math.sin(ang) * dist,
        )
        pygame.draw.line(canvas, RAY_COLORS[i], to_screen(*origin), to_screen(*end), 2)
        pygame.draw.circle(canvas, (255, 255, 255), to_screen(*end), 3)
        pygame.draw.circle(canvas, RAY_COLORS[i], to_screen(*origin), 3)

    surface.blit(canvas, (x0, y0))


def draw_obs_maps(surface, obs, panel_x, y, font):
    draw_text(
        surface, "MULTI-SCALE MAPS (4 scales):", panel_x + 10, y, font, (180, 200, 255)
    )
    y += 18

    keys = ["coverage", "obstacles", "frontier"]
    map_px = 120
    num_scales = 4
    col_gap = 12
    row_gap = 6

    for scale in range(num_scales):
        draw_text(surface, f"S{scale}:", panel_x + 10, y, font, (150, 150, 150))
        for i, (key, name, color) in enumerate(zip(keys, MAP_NAMES, MAP_COLORS)):
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


def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Robot Teleop & Debug Viewer v3")
    clock = pygame.time.Clock()
    font_sm = pygame.font.SysFont("monospace", 12)
    font_md = pygame.font.SysFont("monospace", 14)
    font_lg = pygame.font.SysFont("monospace", 16, bold=True)

    env = RobotCoverageEnv(render_mode="rgb_array", phase=1)
    obs, info = env.reset()

    auto_sample = False
    paused = False
    show_help = True

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_t:
                    auto_sample = not auto_sample
                elif event.key == pygame.K_h:
                    show_help = not show_help
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
                    obs, info = env.reset()
                elif event.key == pygame.K_c:
                    obs, info = env.reset()

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
            action = np.array([throttle, steering], dtype=np.float32)
        else:
            action = env.action_space.sample()

        if not paused:
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()

        screen.fill((20, 20, 20))
        draw_env_view(screen, env, 0, 0)

        panel_x = ENV_SIZE
        pygame.draw.rect(screen, (35, 35, 40), (panel_x, 0, PANEL_W, WINDOW_H))

        y = 8
        draw_text(
            screen,
            f"Phase: {env.phase}  Step: {env.current_step}",
            panel_x + 10,
            y,
            font_lg,
            (255, 220, 100),
        )
        y += 22

        if show_help:
            draw_text(screen, "CONTROLS:", panel_x + 10, y, font_md, (180, 200, 255))
            y += 18
            draw_text(
                screen,
                " W/S: throttle fwd/back  A/D: steer",
                panel_x + 10,
                y,
                font_sm,
                (160, 160, 160),
            )
            y += 15
            draw_text(
                screen,
                " R/C: reset  1-8: phase  SPACE: pause",
                panel_x + 10,
                y,
                font_sm,
                (160, 160, 160),
            )
            y += 15
            draw_text(
                screen,
                " H: toggle help   T: toggle auto   ESC: quit",
                panel_x + 10,
                y,
                font_sm,
                (160, 160, 160),
            )
            y += 20

        sensors = obs["sensors"]
        draw_text(screen, "SENSORS:", panel_x + 10, y, font_md, (180, 200, 255))
        y += 18

        for i, name in enumerate(SENSOR_NAMES[:6]):
            val = sensors[i]
            color = RAY_COLORS[i]
            draw_text(
                screen,
                f" {RAY_LABELS[i]:>2s}: {val:.3f}",
                panel_x + 10,
                y,
                font_sm,
                color,
            )
            draw_bar(screen, panel_x + 110, y + 2, 100, 10, val, 1.0, color)
            y += 16

        draw_text(
            screen,
            f" last_v:  {sensors[6]:.3f}",
            panel_x + 10,
            y,
            font_sm,
            (200, 200, 200),
        )
        y += 16
        draw_text(
            screen,
            f" last_w:  {sensors[7]:.3f}",
            panel_x + 10,
            y,
            font_sm,
            (200, 200, 200),
        )
        y += 22

        draw_text(screen, "COVERAGE:", panel_x + 10, y, font_md, (180, 200, 255))
        y += 18
        pct = env.coverage_in_percent * 100
        draw_text(
            screen,
            f" {env.coverage_in_pixels} / {env.total_cells} ({pct:.1f}%)",
            panel_x + 10,
            y,
            font_sm,
            (100, 220, 100),
        )
        y += 15
        draw_text(
            screen,
            f" collisions: {env.num_collisions}",
            panel_x + 10,
            y,
            font_sm,
            (200, 80, 80),
        )
        y += 15
        bar_w = PANEL_W - 20
        draw_bar(screen, panel_x + 10, y, bar_w, 12, pct, 100, (100, 220, 100))
        draw_text(
            screen, f"{pct:.1f}%", panel_x + bar_w + 14, y - 1, font_sm, (255, 255, 255)
        )
        y += 20

        goal = PHASES[env.phase]["goal"] * 100
        draw_text(
            screen, f" goal: {goal:.0f}%", panel_x + 10, y, font_sm, (200, 200, 200)
        )
        y += 22

        y = draw_obs_maps(screen, obs, panel_x, y, font_sm)

        draw_text(screen, "STATE:", panel_x + 10, y, font_md, (180, 200, 255))
        y += 18
        draw_text(
            screen,
            f" pos: ({env.agent_pos_m[0]:.2f}, {env.agent_pos_m[1]:.2f})",
            panel_x + 10,
            y,
            font_sm,
            (200, 200, 200),
        )
        y += 15
        draw_text(
            screen,
            f" heading: {math.degrees(env.agent_heading):.1f} deg",
            panel_x + 10,
            y,
            font_sm,
            (200, 200, 200),
        )
        y += 15
        draw_text(
            screen,
            f" grid: {env.grid_size_p}x{env.grid_size_p}px ({env.grid_size_m:.1f}m)",
            panel_x + 10,
            y,
            font_sm,
            (200, 200, 200),
        )
        y += 22

        draw_text(screen, "REWARD:", panel_x + 10, y, font_md, (180, 200, 255))
        y += 18
        draw_text(
            screen, f" last: {reward:+.4f}", panel_x + 10, y, font_sm, (255, 255, 180)
        )
        y += 15
        draw_text(
            screen,
            f" base: {REWARD_BASE_PENALTY:+.4f}",
            panel_x + 10,
            y,
            font_sm,
            (255, 120, 120),
        )
        y += 22

        status = []
        if paused:
            status.append("PAUSED")
        if auto_sample:
            status.append("AUTO")
        if status:
            draw_text(
                screen,
                " ".join(status),
                panel_x + PANEL_W - 120,
                8,
                font_lg,
                (255, 100, 100),
            )

        pygame.display.flip()
        clock.tick(60)

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
