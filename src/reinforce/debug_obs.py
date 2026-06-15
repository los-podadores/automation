import numpy as np
import pygame
from robot_env import RobotCoverageEnv

CHANNEL_LABELS = ["Visited", "Current", "Observed"]
CHANNEL_COLORS = [(100, 220, 100), (80, 160, 255), (255, 80, 80)]
CELL_PX = 12
GRID_SIZE = 64
GRID_PX = CELL_PX * GRID_SIZE
MARGIN = 40
ENV_WINDOW = 800
OBS_PANEL_W = GRID_PX * 3 + MARGIN * 4
WINDOW_W = ENV_WINDOW + OBS_PANEL_W
WINDOW_H = max(ENV_WINDOW, GRID_PX + MARGIN * 2)


def draw_obs_channel(surface, channel_data, x0, y0, color):
    flipped = np.fliplr(np.flipud(channel_data))
    for gy in range(GRID_SIZE):
        for gx in range(GRID_SIZE):
            if flipped[gy, gx] > 0:
                rect = (x0 + gx * CELL_PX, y0 + gy * CELL_PX, CELL_PX, CELL_PX)
                pygame.draw.rect(surface, color, rect)


def draw_obs_panel(surface, visual_obs, font):
    panel_x = ENV_WINDOW
    pygame.draw.rect(surface, (30, 30, 30), (panel_x, 0, OBS_PANEL_W, WINDOW_H))

    title = font.render("Visual Observation Channels", True, (220, 220, 220))
    surface.blit(title, (panel_x + MARGIN, 10))

    for ch in range(3):
        x0 = panel_x + MARGIN + ch * (GRID_PX + MARGIN)
        y0 = MARGIN + 10

        label = font.render(f"Ch {ch}: {CHANNEL_LABELS[ch]}", True, (180, 180, 180))
        surface.blit(label, (x0, y0 - 2))

        border_rect = (x0 - 1, y0 + 22, GRID_PX + 2, GRID_PX + 2)
        pygame.draw.rect(surface, (80, 80, 80), border_rect, 1)

        draw_obs_channel(surface, visual_obs[ch], x0, y0 + 23, CHANNEL_COLORS[ch])


def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Debug: Random Agent Visual Observations")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 14)

    env = RobotCoverageEnv(a=2.0, b=1.0, render_mode="rgb_array")

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
        env_img = pygame.surfarray.make_surface(rgb.transpose(1, 0, 2))
        env_img = pygame.transform.smoothscale(env_img, (ENV_WINDOW, ENV_WINDOW))
        screen.blit(env_img, (0, 0))

        draw_obs_panel(screen, obs["visual"], font)

        pygame.display.flip()

        if terminated or truncated:
            obs, _ = env.reset()

        clock.tick(10)

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
