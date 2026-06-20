"""Visualise a trained PPO agent by running episodes in human render mode."""

from __future__ import annotations

import argparse

from stable_baselines3 import PPO

from robot_env import RobotCoverageEnv
from utils import seed_everything


def main() -> None:
    """Load a model and render its behaviour."""
    parser = argparse.ArgumentParser(description="Visualize trained PPO agent")
    parser.add_argument(
        "--model", type=str, required=True, help="Path to trained model"
    )
    parser.add_argument("--phase", type=int, default=1, help="Phase to evaluate (1-8)")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes")
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    if args.seed is not None:
        seed_everything(args.seed)

    model = PPO.load(args.model)
    env = RobotCoverageEnv(render_mode="human", phase=args.phase)

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed)
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            env.render()

        print(
            f"Episode {ep + 1}: {steps} steps, "
            f"coverage={info['coverage_percent']:.2%}, "
            f"collisions={info['num_collisions']}, "
            f"total_reward={total_reward:.2f}"
        )

    env.close()


if __name__ == "__main__":
    main()
