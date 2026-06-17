import cv2
import numpy as np
import pygame
from robot_env import RobotCoverageEnv, MAP_SIZE

CELL_PX = 10
MAP_PX = CELL_PX * MAP_SIZE
MARGIN = 30
ENV_WINDOW = 800
# 3 map types (coverage, obstacles, frontier) + sensors bar
PANEL_W = MAP_PX * 3 + MARGIN * 4
WINDOW_W = ENV_WINDOW + PANEL_W
WINDOW_H = max(ENV_WINDOW, MAP_PX + MARGIN * 2 + 80)

MAP_NAMES = ["Coverage", "Obstacles", "Frontier"]
MAP_COLORS = [
    (100, 220, 100),
    (220, 80, 80),
    (220, 120, 255),
]


def draw_map(surface, data, x0, y0, color):
    img = np.zeros((MAP_SIZE, MAP_SIZE, 3), dtype=np.uint8)
    mask = data > 0.01
    intensity = np.clip(data, 0, 1)
    for c in range(3):
        img[:, :, c] = (intensity * color[c]).astype(np.uint8)
    img[~mask] = 0
    img = cv2.resize(img, (MAP_PX, MAP_PX), interpolation=cv2.INTER_NEAREST)
    img = np.transpose(img, (1, 0, 2))
    surf = pygame.surfarray.make_surface(img)
    surface.blit(surf, (x0, y0))


def draw_panel(surface, obs, font):
    panel_x = ENV_WINDOW
    pygame.draw.rect(surface, (30, 30, 30), (panel_x, 0, PANEL_W, WINDOW_H))

    title = font.render("Multi-Scale Observations (scale 0, finest)", True, (220, 220, 220))
    surface.blit(title, (panel_x + MARGIN, 10))

    keys = ["coverage", "obstacles", "frontier"]
    for i, (key, name, color) in enumerate(zip(keys, MAP_NAMES, MAP_COLORS)):
        x0 = panel_x + MARGIN + i * (MAP_PX + MARGIN)
        y0 = MARGIN + 20

        label = font.render(name, True, (180, 180, 180))
        surface.blit(label, (x0, y0 - 2))

        border = (x0 - 1, y0 + 18, MAP_PX + 2, MAP_PX + 2)
        pygame.draw.rect(surface, (80, 80, 80), border, 1)

        data = obs[key][0]  # scale 0 (finest)
        draw_map(surface, data, x0, y0 + 19, color)

    # Draw sensor bar
    sensors = obs["sensors"]
    sx = panel_x + MARGIN
    sy = MARGIN + MAP_PX + 50
    label = font.render("Sensors (6 rays + derived)", True, (180, 180, 180))
    surface.blit(label, (sx, sy))
    sy += 20
    bar_w = 120
    bar_h = 14
    ray_labels = ["FL", "FC", "FR", "SL", "SR", "RE", "minF", "asym", "wAng", "lastV", "lastW"]
    max_vals = [3.5] * 6 + [1.0, 1.0, 1.0, 1.0, 1.0]
    for j in range(min(11, len(sensors))):
        val = sensors[j]
        mx = max_vals[j]
        ratio = max(0.0, min(1.0, (val - (-1.0 if mx < 0 else 0.0)) / (mx - (-1.0 if mx < 0 else 0.0))))
        lbl = font.render(f"{ray_labels[j]}", True, (150, 150, 150))
        surface.blit(lbl, (sx, sy))
        bx = sx + 40
        pygame.draw.rect(surface, (60, 60, 60), (bx, sy, bar_w, bar_h))
        fill_w = int(bar_w * ratio)
        bar_color = (0, 200, 100) if j < 6 else (200, 200, 0)
        pygame.draw.rect(surface, bar_color, (bx, sy, fill_w, bar_h))
        val_txt = font.render(f"{val:.2f}", True, (200, 200, 200))
        surface.blit(val_txt, (bx + bar_w + 5, sy))
        sy += bar_h + 4


def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Debug: Multi-Scale Observations v2")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 13)

    env = RobotCoverageEnv(render_mode="rgb_array")

    obs, _ = env.reset()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    obs, _ = env.reset()

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        rgb = env.render()
        if rgb is not None:
            env_img = pygame.surfarray.make_surface(rgb.transpose(1, 0, 2))
            env_img = pygame.transform.smoothscale(env_img, (ENV_WINDOW, ENV_WINDOW))
            screen.blit(env_img, (0, 0))

        draw_panel(screen, obs, font)

        pygame.display.flip()

        if terminated or truncated:
            obs, _ = env.reset()

        clock.tick(10)

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
