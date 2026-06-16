import math

import numpy as np
import pygame

from robot_env import (
    REWARD_BASE_PENALTY,
    REWARD_FORWARD,
    REWARD_FRONTIER_BREADCRUMB,
    ROBOT_SPEED_V,
    RobotCoverageEnv,
)

ENV_SIZE = 700
PANEL_W = 660
WINDOW_W = ENV_SIZE + PANEL_W
WINDOW_H = 820
CELL_PX = 8
GRID_SIZE = 64
GRID_PX = CELL_PX * GRID_SIZE

SENSOR_NAMES = [
    "ray_front_left",
    "ray_front_center",
    "ray_front_right",
    "ray_side_left",
    "ray_side_right",
    "ray_rear",
    "min_front",
    "asymmetry",
    "wall_angle",
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
CHANNEL_NAMES = ["Obstacles", "Visited", "Frontiers"]
CHANNEL_COLORS = [(255, 80, 80), (100, 220, 100), (80, 160, 255)]


def draw_text(surface, text, x, y, font, color=(220, 220, 220)):
    surface.blit(font.render(text, True, color), (x, y))


def draw_bar(surface, x, y, w, h, value, max_val, color, bg=(50, 50, 50)):
    pygame.draw.rect(surface, bg, (x, y, w, h))
    fill_w = max(0, min(w, int(w * value / max_val)))
    pygame.draw.rect(surface, color, (x, y, fill_w, h))
    pygame.draw.rect(surface, (100, 100, 100), (x, y, w, h), 1)


def draw_channel(surface, channel_data, x0, y0, color):
    flipped = np.fliplr(np.flipud(channel_data))
    for gy in range(GRID_SIZE):
        for gx in range(GRID_SIZE):
            if flipped[gy, gx] > 0:
                rect = (x0 + gx * CELL_PX, y0 + gy * CELL_PX, CELL_PX, CELL_PX)
                pygame.draw.rect(surface, color, rect)


def draw_env_view(surface, env, x0, y0):
    canvas = pygame.Surface((ENV_SIZE, ENV_SIZE))
    canvas.fill((30, 30, 30))

    if env.field is None:
        surface.blit(canvas, (x0, y0))
        return

    scale = env.render_scale
    off_x = env.render_offset_x
    off_y = env.render_offset_y

    def to_screen(wx, wy):
        px = int((wx - off_x) * scale)
        py = int(ENV_SIZE - (wy - off_y) * scale)
        return px, py

    ex_points = [to_screen(x, y) for x, y in env.field.exterior.coords]
    pygame.draw.polygon(canvas, (220, 220, 220), ex_points)

    for gx, gy in env.visited_grid:
        px = gx * env.grid_resolution
        py = (gy + 1) * env.grid_resolution
        pg = to_screen(px, py)
        rect_size = max(1, int(env.grid_resolution * scale))
        pygame.draw.rect(canvas, (80, 160, 80), (pg[0], pg[1], rect_size, rect_size))

    frontiers = env._get_frontiers()
    for gx, gy in frontiers:
        px = gx * env.grid_resolution
        py = (gy + 1) * env.grid_resolution
        pg = to_screen(px, py)
        rect_size = max(1, int(env.grid_resolution * scale))
        pygame.draw.rect(canvas, (80, 120, 220), (pg[0], pg[1], rect_size, rect_size))

    for gx, gy in env.obstacle_grid:
        px = gx * env.grid_resolution
        py = (gy + 1) * env.grid_resolution
        pg = to_screen(px, py)
        rect_size = max(1, int(env.grid_resolution * scale))
        pygame.draw.rect(canvas, (200, 80, 80), (pg[0], pg[1], rect_size, rect_size))

    pygame.draw.polygon(canvas, (0, 0, 0), ex_points, 2)
    for interior in env.field.interiors:
        in_points = [to_screen(x, y) for x, y in interior.coords]
        pygame.draw.polygon(canvas, (255, 255, 255), in_points)
        pygame.draw.polygon(canvas, (255, 0, 0), in_points, 1)

    body = env._get_agent_polygon()
    body_points = [to_screen(x, y) for x, y in body.exterior.coords]
    pygame.draw.polygon(canvas, (50, 50, 200), body_points)

    hx = env.agent_pos[0] + (env.a / 2) * math.cos(env.agent_theta)
    hy = env.agent_pos[1] + (env.a / 2) * math.sin(env.agent_theta)
    pygame.draw.line(canvas, (0, 255, 0), to_screen(*env.agent_pos), to_screen(hx, hy), 2)

    a, b = env.a, env.b
    theta = env.agent_theta
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    def local_to_global(lx, ly):
        return (
            env.agent_pos[0] + lx * cos_t - ly * sin_t,
            env.agent_pos[1] + lx * sin_t + ly * cos_t,
        )

    origins = [
        local_to_global(a / 2, b / 2),
        local_to_global(a / 2, 0),
        local_to_global(a / 2, -b / 2),
        local_to_global(0, b / 2),
        local_to_global(0, -b / 2),
        local_to_global(-a / 2, 0),
    ]
    ray_angles = [math.pi / 4, 0.0, -math.pi / 4, math.pi / 2, -math.pi / 2, math.pi]

    for i, (origin, ang_off) in enumerate(zip(origins, ray_angles)):
        ang = theta + ang_off
        hit_dist = None
        boundary = env.field.boundary
        from shapely.geometry import LineString, Point

        dx, dy = math.cos(ang), math.sin(ang)
        end_pt = Point(origin[0] + dx * env.max_ray_dist, origin[1] + dy * env.max_ray_dist)
        ray = LineString([Point(origin), end_pt])
        intersection = ray.intersection(boundary)
        if not intersection.is_empty:
            if isinstance(intersection, Point):
                hit_dist = math.hypot(intersection.x - origin[0], intersection.y - origin[1])
            else:
                try:
                    pts = list(intersection.geoms)
                    closest = min(pts, key=lambda pt: origin[0].distance(pt))
                    hit_dist = math.hypot(closest.x - origin[0], closest.y - origin[1])
                except AttributeError:
                    pass

        if hit_dist is not None:
            end = (
                origin[0] + math.cos(ang) * hit_dist,
                origin[1] + math.sin(ang) * hit_dist,
            )
            pygame.draw.line(canvas, RAY_COLORS[i], to_screen(*origin), to_screen(*end), 2)
            hit_screen = to_screen(*end)
            pygame.draw.circle(canvas, (255, 255, 255), hit_screen, 3)
        else:
            end = (
                origin[0] + math.cos(ang) * env.max_ray_dist,
                origin[1] + math.sin(ang) * env.max_ray_dist,
            )
            pygame.draw.line(canvas, RAY_COLORS[i], to_screen(*origin), to_screen(*end), 1)

        pygame.draw.circle(canvas, RAY_COLORS[i], to_screen(*origin), 3)

    surface.blit(canvas, (x0, y0))


def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Robot Teleop & Debug Viewer")
    clock = pygame.time.Clock()
    font_sm = pygame.font.SysFont("monospace", 12)
    font_md = pygame.font.SysFont("monospace", 14)
    font_lg = pygame.font.SysFont("monospace", 16, bold=True)

    env = RobotCoverageEnv(a=2.0, b=1.0, render_mode="rgb_array", phase=1)
    obs, info = env.reset()

    action = np.array([0.0, 0.0], dtype=np.float32)
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
                elif event.key == pygame.K_a:
                    auto_sample = not auto_sample
                elif event.key == pygame.K_h:
                    show_help = not show_help
                elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                    env.phase = int(chr(event.key))
                    obs, info = env.reset()
                elif event.key == pygame.K_c:
                    obs, info = env.reset()

        keys = pygame.key.get_pressed()
        if not auto_sample:
            v_cmd = 0.0
            w_cmd = 0.0
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                v_cmd = 1.0
            if keys[pygame.K_DOWN] or keys[pygame.K_s]:
                v_cmd = -1.0
            if keys[pygame.K_LEFT] or keys[pygame.K_q]:
                w_cmd = 1.0
            if keys[pygame.K_RIGHT] or keys[pygame.K_e]:
                w_cmd = -1.0
            action = np.array([v_cmd, w_cmd], dtype=np.float32)
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
        draw_text(screen, f"Phase: {env.phase}  Step: {env.current_step}", panel_x + 10, y, font_lg, (255, 220, 100))
        y += 22

        if show_help:
            draw_text(screen, "CONTROLS:", panel_x + 10, y, font_md, (180, 200, 255))
            y += 18
            draw_text(screen, " Arrows/WASD: drive  SPACE: pause", panel_x + 10, y, font_sm, (160, 160, 160))
            y += 15
            draw_text(screen, " R/C: reset   1/2/3: phase   A: auto", panel_x + 10, y, font_sm, (160, 160, 160))
            y += 15
            draw_text(screen, " H: toggle help   ESC: quit", panel_x + 10, y, font_sm, (160, 160, 160))
            y += 20

        sensors = obs["sensors"]
        draw_text(screen, "SENSORS:", panel_x + 10, y, font_md, (180, 200, 255))
        y += 18

        for i, name in enumerate(SENSOR_NAMES[:6]):
            val = sensors[i]
            color = RAY_COLORS[i]
            draw_text(screen, f" {RAY_LABELS[i]:>2s}: {val:.3f}", panel_x + 10, y, font_sm, color)
            draw_bar(screen, panel_x + 110, y + 2, 100, 10, val, env.max_ray_dist, color)
            y += 16

        draw_text(screen, f" min_frt: {sensors[6]:.3f}", panel_x + 10, y, font_sm, (200, 200, 200))
        y += 16
        draw_text(screen, f" asymm:   {sensors[7]:.3f}", panel_x + 10, y, font_sm, (200, 200, 200))
        y += 16
        draw_text(screen, f" wall_a:  {sensors[8]:.3f}", panel_x + 10, y, font_sm, (200, 200, 200))
        y += 16
        draw_text(screen, f" last_v:  {sensors[9]:.3f}", panel_x + 10, y, font_sm, (200, 200, 200))
        y += 16
        draw_text(screen, f" last_w:  {sensors[10]:.3f}", panel_x + 10, y, font_sm, (200, 200, 200))
        y += 22

        draw_text(screen, "REWARD BREAKDOWN:", panel_x + 10, y, font_md, (180, 200, 255))
        y += 18

        v_actual = (sensors[9] + 1.0) / 2.0 * ROBOT_SPEED_V
        r_base = REWARD_BASE_PENALTY
        r_fwd = REWARD_FORWARD * v_actual
        total = r_base + r_fwd

        draw_text(screen, f" base_penalty: {r_base:+.4f}", panel_x + 10, y, font_sm, (255, 120, 120))
        y += 15
        draw_text(screen, f" forward:      {r_fwd:+.4f}", panel_x + 10, y, font_sm, (120, 255, 120))
        y += 15

        if env._is_in_visited_cell():
            frontiers = env._get_frontiers()
            if frontiers and env.prev_frontier_dist is not None:
                fd = []
                for fx, fy in frontiers:
                    wx = (fx + 0.5) * env.grid_resolution
                    wy = (fy + 0.5) * env.grid_resolution
                    fd.append(math.hypot(env.agent_pos[0] - wx, env.agent_pos[1] - wy))
                cur_dist = min(fd)
                if cur_dist < env.prev_frontier_dist:
                    draw_text(screen, f" frontier_brd: +{REWARD_FRONTIER_BREADCRUMB:.4f}", panel_x + 10, y, font_sm, (120, 180, 255))
                    total += REWARD_FRONTIER_BREADCRUMB
                else:
                    draw_text(screen, " frontier_brd: +0.0000", panel_x + 10, y, font_sm, (100, 100, 100))
            else:
                draw_text(screen, " frontier_brd: +0.0000", panel_x + 10, y, font_sm, (100, 100, 100))
        else:
            draw_text(screen, " frontier_brd: +0.0000 (not in visited)", panel_x + 10, y, font_sm, (100, 100, 100))
        y += 15

        draw_text(screen, f" TOTAL:        {total:+.4f}", panel_x + 10, y, font_md, (255, 255, 180))
        y += 24

        draw_text(screen, "COVERAGE:", panel_x + 10, y, font_md, (180, 200, 255))
        y += 18
        total_cells = env._get_total_cells()
        visited = len(env.visited_grid)
        frontiers = env._get_frontiers()
        obstacles = len(env.obstacle_grid)
        pct = visited / total_cells * 100 if total_cells > 0 else 0

        draw_text(screen, f" visited:   {visited} / {total_cells} ({pct:.1f}%)", panel_x + 10, y, font_sm, (100, 220, 100))
        y += 15
        draw_text(screen, f" frontiers: {len(frontiers)}", panel_x + 10, y, font_sm, (80, 120, 220))
        y += 15
        draw_text(screen, f" obstacles: {obstacles}", panel_x + 10, y, font_sm, (200, 80, 80))
        y += 15

        bar_w = PANEL_W - 20
        draw_bar(screen, panel_x + 10, y, bar_w, 12, pct, 100, (100, 220, 100))
        draw_text(screen, f"{pct:.1f}%", panel_x + bar_w + 14, y - 1, font_sm, (255, 255, 255))
        y += 20

        if pct >= 95:
            draw_text(screen, " 95% MILESTONE REACHED!", panel_x + 10, y, font_md, (255, 255, 0))
            y += 20

        y += 8
        draw_text(screen, "STATE:", panel_x + 10, y, font_md, (180, 200, 255))
        y += 18
        draw_text(screen, f" pos: ({env.agent_pos[0]:.2f}, {env.agent_pos[1]:.2f})", panel_x + 10, y, font_sm, (200, 200, 200))
        y += 15
        draw_text(screen, f" theta: {math.degrees(env.agent_theta):.1f} deg", panel_x + 10, y, font_sm, (200, 200, 200))
        y += 15
        draw_text(screen, f" in_visited: {env._is_in_visited_cell()}", panel_x + 10, y, font_sm, (200, 200, 200))
        y += 15
        draw_text(screen, f" grid_res: {env.grid_resolution:.3f}", panel_x + 10, y, font_sm, (200, 200, 200))
        y += 24

        draw_text(screen, "VISUAL CHANNELS:", panel_x + 10, y, font_md, (180, 200, 255))
        y += 18

        ch_w = (PANEL_W - 40) // 3
        for ch in range(3):
            cx = panel_x + 10 + ch * (ch_w + 5)
            draw_text(screen, CHANNEL_NAMES[ch], cx, y, font_sm, CHANNEL_COLORS[ch])
            sub = pygame.Surface((ch_w, ch_w))
            sub.fill((20, 20, 25))
            scale_factor = ch_w / GRID_SIZE
            for gy in range(GRID_SIZE):
                for gx in range(GRID_SIZE):
                    if obs["visual"][ch, gy, gx] > 0:
                        rect = (int(gx * scale_factor), int(gy * scale_factor), max(1, int(scale_factor)), max(1, int(scale_factor)))
                        pygame.draw.rect(sub, CHANNEL_COLORS[ch], rect)
            pygame.draw.rect(sub, (80, 80, 80), (0, 0, ch_w, ch_w), 1)
            screen.blit(sub, (cx, y + 16))

        status = []
        if paused:
            status.append("PAUSED")
        if auto_sample:
            status.append("AUTO")
        if status:
            draw_text(screen, " ".join(status), panel_x + PANEL_W - 120, 8, font_lg, (255, 100, 100))

        pygame.display.flip()
        clock.tick(30)

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
