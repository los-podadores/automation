import pygame
from robot_env import RobotCoverageEnv


def main():
    pygame.init()
    screen = pygame.display.set_mode((800, 800))
    pygame.display.set_caption("Phase Debug Viewer (press 1/2/3 to switch phase)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 18)

    env = RobotCoverageEnv(a=2.0, b=1.0, render_mode="human", phase=1)
    obs, _ = env.reset()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                    new_phase = int(chr(event.key))
                    env.phase = new_phase
                    obs, _ = env.reset()
                    print(f"Switched to phase {new_phase}")
                elif event.key == pygame.K_r:
                    obs, _ = env.reset()

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        env.render()

        label = font.render(f"Phase: {env.phase}", True, (0, 0, 0))
        screen.blit(label, (10, 10))

        pygame.display.flip()

        if terminated or truncated:
            obs, _ = env.reset()

        clock.tick(60)

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
