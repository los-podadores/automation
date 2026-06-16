import os

from robot_env import RobotCoverageEnv
from stable_baselines3 import PPO

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "models", "ppo_robot_final.zip"
)
NUM_EPISODES = 20
MAX_STEPS = 10000


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"No model found at {MODEL_PATH}")
        print("Train first with: python train.py")
        return

    model = PPO.load(MODEL_PATH)
    env = RobotCoverageEnv(a=2.0, b=1.0, render_mode="human")

    try:
        for ep in range(NUM_EPISODES):
            obs, _ = env.reset()
            total_reward = 0.0
            steps = 0

            for _ in range(MAX_STEPS):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                env.render()
                total_reward += reward
                steps += 1

                if terminated or truncated:
                    break

            coverage = env.unwrapped.visited_grid
            print(
                f"Episode {ep + 1}: {steps} steps | {len(coverage)} cells covered | reward {total_reward:.1f}"
            )

            env.close_display()
    finally:
        env.close()


if __name__ == "__main__":
    main()
